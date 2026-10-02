"""`flowscope run train.py [args...]`: run an unmodified training script with FlowScope attached.

After the first optimizer.step(), it finds the model that owns the optimizer's parameters and attaches.
"""
import argparse, atexit, gc, os, runpy, sys, time, warnings

import torch.nn as nn
from torch.optim.optimizer import register_optimizer_step_post_hook

from flowscope.scope import FlowScope
from flowscope.server import LiveServer


def find_model(opt):
    """The module that owns the most of the optimizer's parameters (the root model, not a submodule)."""
    params = {id(p) for g in opt.param_groups for p in g["params"]}
    best = None
    with warnings.catch_warnings():  # isinstance on some deprecated torch objects emits FutureWarnings
        warnings.simplefilter("ignore")
        modules = [o for o in gc.get_objects() if _is_module(o)]
    for o in modules:
        ids = [id(p) for p in o.parameters()]
        overlap = sum(i in params for i in ids)
        key = (overlap, -len(ids))
        if overlap and (best is None or key > best[0]):
            best = (key, o)
    return best[1] if best else None


def _is_module(o):
    try:
        return isinstance(o, nn.Module)
    except Exception:
        return False


def autoattach(sink, every, label, record=None):
    state = {"done": False}

    def hook(opt, args, kwargs):
        if state["done"]:
            return
        state["done"] = True
        model = find_model(opt)
        if model is None:
            print("FlowScope: couldn't find a model for the optimizer; not attaching", file=sys.stderr)
            return
        try:
            scope = FlowScope(model, opt, every=every, label=label, sink=sink, record=record)
            scope.steps = 1  # attached after the script's first step; keep step numbers aligned with the script
            print(f"FlowScope: attached to {type(model).__name__}", flush=True)
        except ValueError as e:
            print(e, file=sys.stderr)

    return register_optimizer_step_post_hook(hook)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flowscope", description="Live forward/backward view of a PyTorch training run.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a training script with FlowScope attached")
    run.add_argument("--every", type=int, default=50, help="redraw every N optimizer steps (default 50)")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--host", default="127.0.0.1",
                     help="interface to serve on; 0.0.0.0 makes it reachable from other devices on your network")
    run.add_argument("--record", help="append every tour frame to this JSONL file (replayable in GrandTourVision)")
    run.add_argument("--label", help="name for this run in the comparison chart (default: script + args)")
    run.add_argument("script")
    run.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    server = LiveServer.shared(a.port, a.host)
    autoattach(server, a.every, a.label or " ".join([os.path.basename(a.script)] + a.args), a.record)
    atexit.register(time.sleep, 1.0)  # let open pages receive the last frame before the process exits
    sys.argv = [a.script] + a.args
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.script)))
    runpy.run_path(a.script, run_name="__main__")
