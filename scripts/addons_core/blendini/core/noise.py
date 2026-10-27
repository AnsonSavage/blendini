# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Vectorized hash-based random numbers and Perlin noise."""

import numpy as np

_perm = np.random.RandomState(1337).permutation(256)
_P = np.concatenate([_perm, _perm]).astype(np.int64)

_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1],
], dtype=np.float64)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin3(p):
    """Signed improved Perlin noise, p is (N, 3). Returns (N,) in about [-1, 1]."""
    p = np.asarray(p, dtype=np.float64)
    pi = np.floor(p)
    pf = p - pi
    pi = pi.astype(np.int64) & 255
    u = _fade(pf)
    X, Y, Z = pi[:, 0], pi[:, 1], pi[:, 2]
    x, y, z = pf[:, 0], pf[:, 1], pf[:, 2]

    A = _P[X] + Y
    AA = _P[A] + Z
    AB = _P[A + 1] + Z
    B = _P[X + 1] + Y
    BA = _P[B] + Z
    BB = _P[B + 1] + Z

    def g(h, dx, dy, dz):
        gr = _GRAD3[_P[h] & 15]
        return gr[:, 0] * dx + gr[:, 1] * dy + gr[:, 2] * dz

    def lerp(t, a, b):
        return a + t * (b - a)

    n000 = g(AA, x, y, z)
    n100 = g(BA, x - 1, y, z)
    n010 = g(AB, x, y - 1, z)
    n110 = g(BB, x - 1, y - 1, z)
    n001 = g(AA + 1, x, y, z - 1)
    n101 = g(BA + 1, x - 1, y, z - 1)
    n011 = g(AB + 1, x, y - 1, z - 1)
    n111 = g(BB + 1, x - 1, y - 1, z - 1)
    return lerp(u[:, 2],
                lerp(u[:, 1], lerp(u[:, 0], n000, n100), lerp(u[:, 0], n010, n110)),
                lerp(u[:, 1], lerp(u[:, 0], n001, n101), lerp(u[:, 0], n011, n111)))


_OFFSETS = np.array([[0.0, 0.0, 0.0], [31.416, -47.853, 12.679], [-93.989, 11.512, 67.731], [53.12, 77.7, -21.3]])


def snoise(p, dims=1):
    """Signed noise. dims=1 returns (N,), dims=3 returns (N, 3)."""
    p = np.asarray(p, dtype=np.float64)
    if dims == 1:
        return perlin3(p)
    return np.stack([perlin3(p + _OFFSETS[i]) for i in range(dims)], axis=1)


def noise(p, dims=1):
    """Houdini style noise() centred at 0.5, roughly in [0, 1]."""
    return 0.5 + 0.5 * snoise(p, dims)


def curlnoise(p, eps=1e-3):
    p = np.asarray(p, dtype=np.float64)

    def psi(q):
        return snoise(q, 3)

    ex = np.array([eps, 0, 0])
    ey = np.array([0, eps, 0])
    ez = np.array([0, 0, eps])
    dx = (psi(p + ex) - psi(p - ex)) / (2 * eps)
    dy = (psi(p + ey) - psi(p - ey)) / (2 * eps)
    dz = (psi(p + ez) - psi(p - ez)) / (2 * eps)
    # curl = (dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy)
    return np.stack([
        dy[:, 2] - dz[:, 1],
        dz[:, 0] - dx[:, 2],
        dx[:, 1] - dy[:, 0],
    ], axis=1)


def fbm(p, octaves=3, lacunarity=2.0, roughness=0.5, dims=1):
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for _ in range(max(1, int(octaves))):
        total = total + snoise(p * freq, dims) * amp
        norm += amp
        amp *= roughness
        freq *= lacunarity
    return total / norm


# ----------------------------------------------------------------------------
# Hashing


def _hash_u32(x):
    x = x.astype(np.uint64) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7feb352d)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846ca68b)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    return x


def _float_bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)


def hash_seed(seed):
    """Combine seed array (N,) or (N, k) into uint64 hash (N,)."""
    seed = np.asarray(seed)
    if seed.ndim == 1:
        seed = seed[:, None]
    h = np.full(seed.shape[0], 0x9E3779B9, dtype=np.uint64)
    for i in range(seed.shape[1]):
        h = _hash_u32(h ^ _float_bits(seed[:, i]) ^ np.uint64(i * 0x85EBCA6B & 0xFFFFFFFF))
    return h


def rand(seed, dims=1):
    """Deterministic random in [0, 1) from a seed (N,) or (N, k)."""
    h = hash_seed(seed)
    outs = []
    for i in range(dims):
        hi = _hash_u32(h + np.uint64(i * 0x632BE5AB & 0xFFFFFFFF))
        outs.append(hi.astype(np.float64) / 4294967296.0)
    if dims == 1:
        return outs[0]
    return np.stack(outs, axis=1)
