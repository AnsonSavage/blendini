# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
SIMD VEX interpreter.

A wrangle snippet is executed once for *all* elements: every value is a numpy array with
one entry per element ("lane"). Control flow is handled with lane masks, the same way a
GPU (or Houdini's own VEX VM) executes divergent branches:

- ``if`` runs both branches with complementary masks,
- loops keep iterating while any lane is still active,
- ``break`` / ``continue`` / ``return`` remove lanes from the active mask.
"""

import numpy as np

from .parser import parse, VexSyntaxError
from .values import (
    V, VexError, SHAPES, SCALARS, VECTORS, MATRICES, VEC_BY_SIZE,
    is_array, elem_type, const, zeros, full, mwhere, convert, truthy, lane_mask, type_of_array, dtype_for,
)
from . import builtins as _builtins

_COMPONENTS = {'x': 0, 'y': 1, 'z': 2, 'w': 3, 'r': 0, 'g': 1, 'b': 2, 'a': 3, 'u': 0, 'v': 1}

_MAX_LOOP_ITERATIONS = 1000000


class _Func:
    __slots__ = ('rtype', 'ret', 'returned')

    def __init__(self, rtype, n):
        self.rtype = rtype
        self.ret = None
        self.returned = np.zeros(n, dtype=bool)


class _Loop:
    __slots__ = ('cont',)

    def __init__(self, n):
        self.cont = np.zeros(n, dtype=bool)


class Interpreter:
    def __init__(self, program, ctx, n):
        self.program = program
        self.functions = program.functions
        self.ctx = ctx
        self.n = n
        self.scopes = [_constants(), {}]
        self.attrs = {}       # binding name -> V
        self.attr_types = {}  # binding name -> type
        self.written = set()
        self.loops = []
        self.funcs = [_Func('void', n)]
        self.cur_line = 0

    # ------------------------------------------------------------------
    # Variables

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope
        return None

    def declare(self, t, name, value, mask):
        scope = self.scopes[-1]
        if value is None:
            value = zeros(t, self.n)
        else:
            value = convert(value, t)
        old = scope.get(name)
        if old is not None and old.t == t:
            scope[name] = V(t, mwhere(mask, value.d, old.d))
        else:
            scope[name] = V(t, np.array(full(value, self.n), copy=True))

    # ------------------------------------------------------------------
    # Statements

    def run(self):
        mask = np.ones(self.n, dtype=bool)
        with np.errstate(all='ignore'):
            try:
                self.exec_block(self.program.body, mask)
            except VexError as ex:
                if str(ex).startswith("Line "):
                    raise
                raise VexError("Line %d: %s" % (self.cur_line, ex)) from None
            except (TypeError, ValueError, IndexError, KeyError, AttributeError) as ex:
                raise VexError("Line %d: %s" % (self.cur_line, ex)) from None

    def exec_block(self, stmts, mask):
        for s in stmts:
            if not mask.any():
                break
            mask = self.exec(s, mask)
        return mask

    def exec(self, node, mask):
        k = node.kind
        if node.line:
            self.cur_line = node.line
        if k == 'expr':
            self.ev(node.expr, mask)
            return mask
        if k == 'decl':
            for t, name, init in node.decls:
                value = None
                if init is not None:
                    value = self.ev(init, mask, hint=t)
                self.declare(t, name, value, mask)
            return mask
        if k == 'block':
            self.scopes.append({})
            try:
                return self.exec_block(node.body, mask)
            finally:
                self.scopes.pop()
        if k == 'if':
            c = truthy(self.ev(node.cond, mask)) & mask
            m_then = c
            m_else = mask & ~c
            out = np.zeros_like(mask)
            if m_then.any():
                out |= self.exec(node.then, m_then)
            if m_else.any():
                if node.other is not None:
                    out |= self.exec(node.other, m_else)
                else:
                    out |= m_else
            return out
        if k == 'for':
            return self.exec_for(node, mask)
        if k == 'foreach':
            return self.exec_foreach(node, mask)
        if k == 'break':
            return np.zeros_like(mask)
        if k == 'continue':
            if self.loops:
                self.loops[-1].cont |= mask
            return np.zeros_like(mask)
        if k == 'return':
            f = self.funcs[-1]
            if node.value is not None:
                rt = f.rtype if f.rtype != 'void' else None
                v = self.ev(node.value, mask, hint=rt)
                if rt is None:
                    rt = v.t
                    f.rtype = rt
                v = convert(v, rt)
                if f.ret is None:
                    f.ret = zeros(rt, self.n)
                f.ret = V(rt, mwhere(mask, v.d, f.ret.d))
            f.returned |= mask
            return np.zeros_like(mask)
        raise VexError("Unknown statement %s" % k)

    def exec_for(self, node, mask):
        self.scopes.append({})
        try:
            if node.init is not None:
                self.exec(node.init, mask)
            loop = _Loop(self.n)
            self.loops.append(loop)
            active = mask.copy()
            first = True
            it = 0
            while True:
                if node.cond is not None and not (node.do and first):
                    active &= truthy(self.ev(node.cond, active))
                if not active.any():
                    break
                loop.cont[:] = False
                out = self.exec(node.body, active)
                active = out | loop.cont
                if node.step is not None and active.any():
                    self.ev(node.step, active)
                first = False
                it += 1
                if it > _MAX_LOOP_ITERATIONS:
                    raise VexError("Loop exceeded %d iterations (infinite loop?)" % _MAX_LOOP_ITERATIONS)
            self.loops.pop()
        finally:
            self.scopes.pop()
        return mask & ~self.funcs[-1].returned

    def exec_foreach(self, node, mask):
        arr = self.ev(node.array, mask)
        if not is_array(arr.t):
            raise VexError("foreach needs an array")
        et = elem_type(arr.t)
        vt = node.vtype or et
        data = full(arr, self.n)
        lengths = np.array([len(x) for x in data], dtype=np.int64)
        maxlen = int(lengths[mask].max()) if mask.any() else 0
        self.scopes.append({})
        try:
            loop = _Loop(self.n)
            self.loops.append(loop)
            active = mask.copy()
            for i in range(maxlen):
                active &= lengths > i
                if not active.any():
                    break
                elem = _gather_array_elem(data, np.full(self.n, i), et, active)
                self.declare(vt, node.var, elem, np.ones(self.n, dtype=bool))
                if node.index:
                    self.declare('int', node.index, V('int', np.full(self.n, i, dtype=np.int64)), np.ones(self.n, dtype=bool))
                loop.cont[:] = False
                out = self.exec(node.body, active)
                active = out | loop.cont
            self.loops.pop()
        finally:
            self.scopes.pop()
        return mask & ~self.funcs[-1].returned

    # ------------------------------------------------------------------
    # Expressions

    def ev(self, node, mask, hint=None):
        k = node.kind
        if k == 'num':
            return const(node.type, node.value)
        if k == 'str':
            return const('string', node.value)
        if k == 'var':
            scope = self.lookup(node.name)
            if scope is None:
                raise VexError("Line %d: undefined variable '%s'" % (node.line, node.name))
            return scope[node.name]
        if k == 'attr':
            return self.read_attr(node)
        if k == 'binop':
            return self.binop(node, mask)
        if k == 'unop':
            v = self.ev(node.x, mask)
            if node.op == '-':
                if v.t == 'string' or is_array(v.t):
                    raise VexError("Can't negate %s" % v.t)
                return V(v.t, -v.d)
            if node.op == '!':
                return V('int', (~truthy(v)).astype(np.int64))
            if node.op == '~':
                return V('int', ~convert(v, 'int').d)
        if k == 'assign':
            return self.assign(node, mask)
        if k == 'incdec':
            old = self.ev(node.target, mask)
            delta = 1 if node.op == '++' else -1
            new = V(old.t, old.d + delta)
            self.store(node.target, new, mask)
            return old if node.post else new
        if k == 'ternary':
            c = truthy(self.ev(node.cond, mask))
            a = self.ev(node.yes, mask & c, hint)
            b = self.ev(node.no, mask & ~c, hint)
            t = _promote(a.t, b.t) if a.t != b.t else a.t
            a = convert(a, t)
            b = convert(b, t)
            return V(t, mwhere(c, full(a, self.n), full(b, self.n)))
        if k == 'member':
            v = self.ev(node.x, mask)
            idx = _swizzle(node.name, v.t, node.line)
            if v.t in MATRICES:
                raise VexError("Line %d: member access on matrix" % node.line)
            d = v.d[:, idx]
            if len(idx) == 1:
                return V('float', d[:, 0])
            return V(VEC_BY_SIZE[len(idx)], d)
        if k == 'index':
            v = self.ev(node.x, mask)
            i = convert(self.ev(node.i, mask), 'int')
            if is_array(v.t):
                return _gather_array_elem(full(v, self.n), full(i, self.n), elem_type(v.t), mask)
            if v.t in VECTORS:
                k2 = VECTORS[v.t]
                ii = np.clip(full(i, self.n), 0, k2 - 1)
                d = full(v, self.n)
                return V('float', d[np.arange(self.n), ii])
            if v.t in MATRICES:
                k2 = MATRICES[v.t]
                ii = np.clip(full(i, self.n), 0, k2 - 1)
                d = full(v, self.n)
                return V(VEC_BY_SIZE[k2], d[np.arange(self.n), ii])
            if v.t == 'string':
                d = full(v, self.n)
                ii = full(i, self.n)
                out = np.empty(self.n, dtype=object)
                for j in range(self.n):
                    s = d[j]
                    out[j] = s[ii[j]] if -len(s) <= ii[j] < len(s) else ''
                return V('string', out)
            raise VexError("Line %d: can't index %s" % (node.line, v.t))
        if k == 'cast':
            return self.cast(node, mask)
        if k == 'brace':
            return self.brace(node, mask, hint)
        if k == 'call':
            return self.call(node, mask, hint)
        if k == 'seq':
            v = None
            for item in node.items:
                v = self.ev(item, mask)
            return v
        raise VexError("Unknown expression %s" % k)

    def brace(self, node, mask, hint):
        items = [self.ev(x, mask) for x in node.items]
        if hint and is_array(hint):
            et = elem_type(hint)
            return _make_array(et, [convert(x, et) for x in items], self.n)
        if all(x.t in SCALARS for x in items) and len(items) in (2, 3, 4, 9, 16):
            t = {2: 'vector2', 3: 'vector', 4: 'vector4', 9: 'matrix3', 16: 'matrix'}[len(items)]
            if hint in MATRICES and len(items) == MATRICES[hint] ** 2:
                t = hint
            return _assemble(t, items, self.n)
        if all(x.t in VECTORS for x in items) and len(items) in (2, 3, 4):
            k2 = len(items)
            t = {2: 'matrix2', 3: 'matrix3', 4: 'matrix'}[k2]
            rows = [convert(x, VEC_BY_SIZE[k2]) for x in items]
            n = max(r.d.shape[0] for r in rows)
            d = np.stack([full(r, n) for r in rows], axis=1)
            return V(t, d)
        et = items[0].t if items else 'float'
        return _make_array(et, [convert(x, et) for x in items], self.n)

    def cast(self, node, mask):
        t = node.type
        args = node.a.get('args')
        if args is not None and len(args) > 1:
            vals = [self.ev(a, mask) for a in args]
            if is_array(t):
                et = elem_type(t)
                return _make_array(et, [convert(x, et) for x in vals], self.n)
            return _assemble(t, vals, self.n)
        if args is not None and len(args) == 0:
            return zeros(t, 1)
        v = self.ev(node.x, mask, hint=t)
        if t in VECTORS and v.t in SCALARS:
            return convert(v, t)
        return convert(v, t)

    def binop(self, node, mask):
        op = node.op
        if op == '&&':
            a = truthy(self.ev(node.l, mask))
            b = truthy(self.ev(node.r, mask & a))
            return V('int', (a & b).astype(np.int64))
        if op == '||':
            a = truthy(self.ev(node.l, mask))
            b = truthy(self.ev(node.r, mask & ~a))
            return V('int', (a | b).astype(np.int64))
        a = self.ev(node.l, mask)
        b = self.ev(node.r, mask)
        return arith(op, a, b, node.line)

    def assign(self, node, mask):
        op = node.op
        target = node.target
        hint = self.target_type(target)
        value = self.ev(node.value, mask, hint=hint)
        if op != '=':
            cur = self.ev(target, mask)
            value = arith(op[:-1], cur, value, node.line)
        self.store(target, value, mask)
        return value

    def target_type(self, target):
        k = target.kind
        if k == 'var':
            scope = self.lookup(target.name)
            return scope[target.name].t if scope else None
        if k == 'attr':
            return self.binding_type(target)
        if k == 'member':
            return 'float' if len(target.name) == 1 else VEC_BY_SIZE.get(len(target.name))
        if k == 'index':
            bt = self.target_type(target.x)
            if bt and is_array(bt):
                return elem_type(bt)
            return 'float'
        return None

    def store(self, target, value, mask):
        k = target.kind
        if k == 'var':
            scope = self.lookup(target.name)
            if scope is None:
                raise VexError("Line %d: undefined variable '%s'" % (target.line, target.name))
            old = scope[target.name]
            value = convert(value, old.t)
            scope[target.name] = V(old.t, mwhere(mask, value.d, old.d))
            return
        if k == 'attr':
            t = self.binding_type(target)
            old = self.read_attr(target)
            value = convert(value, t)
            name = target.name
            self.attrs[name] = V(t, mwhere(mask, value.d, full(old, self.n)))
            self.written.add(name)
            return
        if k == 'member':
            base = self.ev(target.x, mask)
            idx = _swizzle(target.name, base.t, target.line)
            src = convert(value, 'float' if len(idx) == 1 else VEC_BY_SIZE[len(idx)])
            d = np.array(full(base, self.n), dtype=np.float64, copy=True)
            sd = full(src, self.n)
            if len(idx) == 1:
                d[:, idx[0]] = sd
            else:
                d[:, idx] = sd
            self.store(target.x, V(base.t, d), mask)
            return
        if k == 'index':
            base = self.ev(target.x, mask)
            i = full(convert(self.ev(target.i, mask), 'int'), self.n)
            if is_array(base.t):
                et = elem_type(base.t)
                src = full(convert(value, et), self.n)
                d = np.array(full(base, self.n), dtype=object, copy=True)
                for j in np.nonzero(mask)[0]:
                    arr = d[j]
                    ii = int(i[j])
                    if et == 'string':
                        arr = list(arr)
                        if ii < 0:
                            ii += len(arr)
                        while len(arr) <= ii:
                            arr.append('')
                        arr[ii] = src[j]
                    else:
                        arr = np.array(arr, copy=True)
                        if ii < 0:
                            ii += len(arr)
                        if ii >= len(arr):
                            grow = np.zeros((ii + 1 - len(arr),) + arr.shape[1:], dtype=arr.dtype)
                            arr = np.concatenate([arr, grow])
                        arr[ii] = src[j]
                    d[j] = arr
                self.store(target.x, V(base.t, d), mask)
                return
            if base.t in VECTORS:
                k2 = VECTORS[base.t]
                src = full(convert(value, 'float'), self.n)
                d = np.array(full(base, self.n), copy=True)
                ii = np.clip(i, 0, k2 - 1)
                lanes = np.nonzero(mask)[0]
                d[lanes, ii[lanes]] = src[lanes]
                self.store(target.x, V(base.t, d), mask)
                return
            if base.t in MATRICES:
                k2 = MATRICES[base.t]
                src = full(convert(value, VEC_BY_SIZE[k2]), self.n)
                d = np.array(full(base, self.n), copy=True)
                ii = np.clip(i, 0, k2 - 1)
                lanes = np.nonzero(mask)[0]
                d[lanes, ii[lanes]] = src[lanes]
                self.store(target.x, V(base.t, d), mask)
                return
        raise VexError("Line %d: can't assign to expression" % target.line)

    # ------------------------------------------------------------------
    # Attribute bindings

    def binding_type(self, node):
        name = node.name
        t = self.attr_types.get(name)
        if t is not None:
            return t
        t = node.type or self.program.bindings.get(name) or self.ctx.implicit_type(name)
        self.attr_types[name] = t
        return t

    def read_attr(self, node):
        name = node.name
        v = self.attrs.get(name)
        if v is not None:
            return v
        t = self.binding_type(node)
        v = self.ctx.bind_attr(name, t, self.n)
        v = convert(v, t)
        self.attrs[name] = v
        return v

    # ------------------------------------------------------------------
    # Functions

    def call(self, node, mask, hint):
        name = node.name
        fn = self.functions.get(name)
        if fn is not None:
            return self.call_user(fn, node, mask)
        impl = _builtins.FUNCTIONS.get(name)
        if impl is None:
            raise VexError("Line %d: unknown function '%s'" % (node.line, name))
        args = [self.ev(a, mask) for a in node.args]
        try:
            result = impl(self, args, mask, hint)
        except IndexError:
            if any(len(a.d) == 0 for a in args):
                raise
            raise VexError("Line %d: wrong number of arguments to %s()" % (node.line, name)) from None
        if name in _builtins.INPLACE:
            # Functions like append() / rotate() modify their first argument.
            new_val, result = result
            self.store(node.args[0], new_val, mask)
        elif name in _builtins.OUTARGS:
            # Functions with output parameters, e.g. xyzdist(geo, pos, &prim, &uv).
            result, outs = result
            for idx, val in outs.items():
                if idx < len(node.args) and node.args[idx].kind in ('var', 'attr', 'member', 'index'):
                    self.store(node.args[idx], val, mask)
        if result is None:
            return const('int', 0)
        return result

    def call_user(self, fn, node, mask):
        if len(node.args) != len(fn.params):
            raise VexError("Line %d: %s() expects %d arguments" % (node.line, fn.name, len(fn.params)))
        scope = {}
        for (pt, pname), arg in zip(fn.params, node.args):
            v = convert(self.ev(arg, mask, hint=pt), pt)
            scope[pname] = V(pt, np.array(full(v, self.n), copy=True))
        saved = self.scopes
        self.scopes = [saved[0], scope]
        f = _Func(fn.rtype, self.n)
        self.funcs.append(f)
        if len(self.funcs) > 200:
            raise VexError("Recursion too deep in %s()" % fn.name)
        try:
            self.exec(fn.body, mask)
        finally:
            self.funcs.pop()
            self.scopes = saved
        # VEX passes arguments by reference.
        for (pt, pname), arg in zip(fn.params, node.args):
            if arg.kind in ('var', 'attr', 'member', 'index'):
                self.store(arg, scope[pname], mask)
        if fn.rtype == 'void':
            return const('int', 0)
        return f.ret if f.ret is not None else zeros(fn.rtype, self.n)


# ----------------------------------------------------------------------
# Helpers

def _constants():
    """Constants from Houdini's math.h (which wrangles include by default)."""
    import math
    vals = {
        'M_PI': math.pi, 'PI': math.pi, 'M_PI_2': math.pi / 2, 'M_PI_4': math.pi / 4, 'M_TWO_PI': 2 * math.pi,
        'M_1_PI': 1 / math.pi, 'M_2_PI': 2 / math.pi, 'M_E': math.e, 'M_SQRT2': math.sqrt(2),
        'M_SQRT1_2': math.sqrt(0.5), 'M_LN2': math.log(2), 'M_LN10': math.log(10), 'M_LOG2E': 1 / math.log(2),
        'M_LOG10E': 1 / math.log(10), 'FLT_MAX': 3.402823466e+38, 'FLT_MIN': 1.175494351e-38,
        'FLT_EPSILON': 1.192092896e-07,
    }
    out = {k: const('float', v) for k, v in vals.items()}
    out['INT_MAX'] = const('int', 2147483647)
    out['INT_MIN'] = const('int', -2147483648)
    return out


def _swizzle(name, t, line):
    if t not in VECTORS:
        raise VexError("Line %d: '.%s' needs a vector, got %s" % (line, name, t))
    k = VECTORS[t]
    try:
        idx = [_COMPONENTS[c] for c in name]
    except KeyError:
        raise VexError("Line %d: bad component '.%s'" % (line, name))
    if any(i >= k for i in idx):
        raise VexError("Line %d: component '.%s' out of range for %s" % (line, name, t))
    return idx


def _assemble(t, vals, n_lanes):
    if t in VECTORS or t in MATRICES:
        size = VECTORS.get(t) or MATRICES[t] ** 2
        comps = []
        for v in vals:
            if v.t in SCALARS:
                comps.append(v.d.astype(np.float64)[:, None])
            elif v.t in VECTORS:
                comps.append(v.d)
            elif v.t in MATRICES:
                comps.append(v.d.reshape(v.d.shape[0], -1))
        n = max(c.shape[0] for c in comps)
        comps = [np.broadcast_to(c, (n, c.shape[1])) for c in comps]
        d = np.concatenate(comps, axis=1)
        if d.shape[1] < size:
            d = np.concatenate([d, np.zeros((n, size - d.shape[1]))], axis=1)
        d = d[:, :size]
        if t in MATRICES:
            k = MATRICES[t]
            d = d.reshape(n, k, k)
        return V(t, d)
    return convert(vals[0], t)


def _make_array(et, vals, n):
    out = np.empty(n, dtype=object)
    if not vals:
        from .values import _empty_array
        for i in range(n):
            out[i] = _empty_array(et)
        return V(et + '[]', out)
    datas = [full(v, n) for v in vals]
    if et == 'string':
        for i in range(n):
            out[i] = [d[i] for d in datas]
    else:
        stacked = np.stack(datas, axis=1)
        for i in range(n):
            out[i] = stacked[i]
    return V(et + '[]', out)


def _gather_array_elem(data, idx, et, mask):
    n = len(data)
    if et == 'string':
        out = np.empty(n, dtype=object)
        out[:] = ''
    else:
        out = np.zeros((n,) + SHAPES.get(et, ()), dtype=dtype_for(et))
    for j in np.nonzero(mask)[0]:
        arr = data[j]
        ii = int(idx[j])
        ln = len(arr)
        if ii < 0:
            ii += ln
        if 0 <= ii < ln:
            out[j] = arr[ii]
    return V(et, out)


_RANK = {'int': 0, 'float': 1, 'vector2': 2, 'vector': 3, 'vector4': 4}


def _promote(ta, tb):
    if ta == tb:
        return ta
    if ta == 'string' or tb == 'string':
        return 'string'
    if ta in MATRICES or tb in MATRICES:
        return ta if ta in MATRICES else tb
    return ta if _RANK.get(ta, 0) >= _RANK.get(tb, 0) else tb


def arith(op, a, b, line=0):
    ta, tb = a.t, b.t
    if is_array(ta) or is_array(tb):
        if op in ('==', '!='):
            n = max(a.n, b.n)
            da, db = full(a, n), full(b, n)
            eq = np.array([np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(da, db)])
            return V('int', (eq if op == '==' else ~eq).astype(np.int64))
        raise VexError("Line %d: operator %s not supported on arrays" % (line, op))
    if ta == 'string' or tb == 'string':
        n = max(a.n, b.n)
        sa = full(convert(a, 'string'), n)
        sb = full(convert(b, 'string'), n)
        if op == '+':
            out = np.empty(n, dtype=object)
            for i in range(n):
                out[i] = sa[i] + sb[i]
            return V('string', out)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            fn = {'==': lambda x, y: x == y, '!=': lambda x, y: x != y, '<': lambda x, y: x < y,
                  '>': lambda x, y: x > y, '<=': lambda x, y: x <= y, '>=': lambda x, y: x >= y}[op]
            return V('int', np.array([fn(x, y) for x, y in zip(sa, sb)], dtype=np.int64))
        raise VexError("Line %d: operator %s not supported on strings" % (line, op))

    # Matrix products
    if op == '*' and (ta in MATRICES or tb in MATRICES):
        if ta in MATRICES and tb in MATRICES:
            if ta != tb:
                b = convert(b, ta)
            return V(ta, np.matmul(a.d, b.d))
        if ta in VECTORS and tb in MATRICES:
            return V(ta, _vec_mat(a.d, b.d))
        if ta in MATRICES and tb in VECTORS:
            # m * v == v * transpose(m) in VEX
            return V(tb, _vec_mat(b.d, np.swapaxes(a.d, 1, 2)))
        if ta in SCALARS:
            return V(tb, a.d.astype(np.float64)[:, None, None] * b.d)
        return V(ta, a.d * b.d.astype(np.float64)[:, None, None])

    if op in ('==', '!=', '<', '>', '<=', '>='):
        t = _promote(ta, tb)
        da = convert(a, t).d
        db = convert(b, t).d
        if da.ndim > 1 or db.ndim > 1:
            n = max(len(da), len(db))
            da = np.broadcast_to(da, (n,) + da.shape[1:]).reshape(n, -1)
            db = np.broadcast_to(db, (n,) + db.shape[1:]).reshape(n, -1)
            if op == '==':
                return V('int', np.all(da == db, axis=1).astype(np.int64))
            if op == '!=':
                return V('int', np.any(da != db, axis=1).astype(np.int64))
            # VEX compares vectors by length for ordering.
            da = np.linalg.norm(da, axis=1)
            db = np.linalg.norm(db, axis=1)
        r = {'==': np.equal, '!=': np.not_equal, '<': np.less, '>': np.greater,
             '<=': np.less_equal, '>=': np.greater_equal}[op](da, db)
        return V('int', r.astype(np.int64))

    if op in ('&', '|', '^', '<<', '>>'):
        da = convert(a, 'int').d
        db = convert(b, 'int').d
        f = {'&': np.bitwise_and, '|': np.bitwise_or, '^': np.bitwise_xor,
             '<<': np.left_shift, '>>': np.right_shift}[op]
        return V('int', f(da, db))

    # Arithmetic
    if ta in SCALARS and tb in SCALARS:
        t = 'float' if 'float' in (ta, tb) else 'int'
        da, db = a.d, b.d
    elif ta in SCALARS:
        t = tb
        da, db = a.d.astype(np.float64)[:, None], b.d
    elif tb in SCALARS:
        t = ta
        da, db = a.d, b.d.astype(np.float64)[:, None]
    else:
        t = ta if _RANK[ta] >= _RANK[tb] else tb
        da, db = convert(a, t).d, convert(b, t).d
    if op == '+':
        r = da + db
    elif op == '-':
        r = da - db
    elif op == '*':
        r = da * db
    elif op == '/':
        if t == 'int':
            safe = np.where(db == 0, 1, db)
            r = np.where(db == 0, 0, np.trunc(da / safe)).astype(np.int64)
        else:
            r = np.where(db == 0, 0.0, da / np.where(db == 0, 1, db))
    elif op == '%':
        safe = np.where(db == 0, 1, db)
        r = np.where(db == 0, 0, np.fmod(da, safe))
        if t == 'int':
            r = r.astype(np.int64)
    else:
        raise VexError("Line %d: unknown operator %s" % (line, op))
    return V(t, r)


def _vec_mat(v, m):
    """Row vector times matrix with homogeneous coordinate for 4x4 matrices."""
    k = v.shape[1]
    mk = m.shape[1]
    if k == mk:
        return np.einsum('ni,nij->nj', *_bcast(v, m))
    if k == 3 and mk == 4:
        h = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
        r = np.einsum('ni,nij->nj', *_bcast(h, m))
        w = r[:, 3:4]
        w = np.where(w == 0, 1, w)
        return r[:, :3] / w
    if k == 3 and mk == 3:
        return np.einsum('ni,nij->nj', *_bcast(v, m))
    raise VexError("Can't multiply vector%d by matrix%d" % (k, mk))


def _bcast(v, m):
    n = max(v.shape[0], m.shape[0])
    return (np.broadcast_to(v, (n,) + v.shape[1:]), np.broadcast_to(m, (n,) + m.shape[1:]))


def compile_vex(src):
    return parse(src)
