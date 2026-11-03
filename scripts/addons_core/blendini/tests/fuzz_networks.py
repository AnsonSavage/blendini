# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Random network fuzzer: chains random SOPs with random parameter variants and reports any
exception other than a CookError, plus cooks slower than 5 s (or stuck for 20 s).

    python fuzz_networks.py SEED COUNT          (pure Python SOPs only)
    blender --background --python fuzz_networks.py -- SEED COUNT   (with the Blender-backed SOPs)
"""
import random, traceback, sys, collections
import numpy as np
sys.path.insert(0, __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')))
from blendini.sops import REGISTRY, default_parms, CookContext, CookError, cook_sop
args = [a for a in sys.argv[1:] if a.lstrip('-').isdigit()]
seed = int(args[0]) if args else 0
rng = random.Random(seed)
skip = {'file', 'objectmerge', 'font', 'attribfrommap', 'blendermodifier', 'filecache'}
ids = [k for k in REGISTRY if k not in skip]
bases = [cook_sop('box'), cook_sop('grid'), cook_sop('sphere'), cook_sop('line', points=8),
         cook_sop('scatter', [cook_sop('sphere')], npts=40), cook_sop('volume', voxelsize=0.25),
         cook_sop('voronoifracture', [cook_sop('box')]) if 'voronoifracture' in REGISTRY else cook_sop('box'),
         cook_sop('pack', [cook_sop('box')])]
from blendini.core.geometry import Geometry
bases.append(Geometry())
fails = collections.Counter()
examples = {}
def variant(sid):
    p = default_parms(sid)
    for parm in REGISTRY[sid].parms:
        r = rng.random()
        if parm.type == 'enum' and r < 0.5:
            p[parm.name] = rng.choice(parm.items)[0]
        elif parm.type == 'bool' and r < 0.3:
            p[parm.name] = not parm.default
        elif parm.type in ('float', 'int') and r < 0.15:
            v = parm.default * rng.choice([0, 0.5, 2, -1]) if parm.default else rng.choice([0, 1, -1])
            if parm.min is not None: v = max(v, parm.min)
            if parm.max is not None: v = min(v, parm.max)
            p[parm.name] = int(v) if parm.type == 'int' else float(v)
    return p
N = int(args[1]) if len(args) > 1 else 300
for it in range(N):
    geo = rng.choice(bases)
    for depth in range(rng.randint(1, 4)):
        sid = rng.choice(ids)
        sop = REGISTRY[sid]
        ins = [geo] + [rng.choice(bases) for _ in range(max(sop.inputs - 1, 0))]
        p = variant(sid)
        ctx = CookContext(frame=rng.choice([1, 3, 10]), time=0.1, state={})
        ctx.channels, ctx.ramps = {}, {}
        import signal, time as _t
        class Slow(Exception): pass
        def _alarm(*a): raise Slow()
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(20)
        t0 = _t.time()
        try:
            out = sop.cook(ins[:max(sop.inputs, 1)], p, ctx)
            signal.alarm(0)
            if _t.time() - t0 > 5:
                print('SLOW', sid, round(_t.time() - t0, 1), {k: v for k, v in p.items() if k != 'snippet'}, 'in', ins[0].num_points, flush=True)
            if isinstance(out, list):
                out = out[0]
            if out is None:
                raise RuntimeError("returned None")
            geo = out
        except Slow:
            print('TIMEOUT', sid, {k: v for k, v in p.items() if k != 'snippet'}, 'in pts', ins[0].num_points, 'prims', ins[0].num_prims, 'vols', len(ins[0].volumes), flush=True)
            break
        except CookError:
            signal.alarm(0)
            break
        except Exception as ex:
            key = (sid, type(ex).__name__, str(ex)[:80])
            fails[key] += 1
            if key not in examples:
                examples[key] = (dict(p), traceback.format_exc().splitlines()[-4:])
            break
print('seed', seed, 'iterations', N, 'distinct failures', len(fails))
for k, c in fails.most_common(30):
    print(c, k)
    print('    ', examples[k][1][-3:])
