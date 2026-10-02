"""What does RL through coding-agent harnesses change inside a model?

Runs the same tokens through LFM2.5-2.6B before and after the FineEnvs multi-harness training
(https://huggingface.co/spaces/AdithyaSK/multi-harness-rl) and writes a FlowScope recording that GrandTourVision
opens as one frame per checkpoint.

Inputs come from FineEnvs/SmolDataEnvs-multiharness-sft, pre-tokenized for LFM2.5: tasks that were run in every
harness (claude-code, codex, opencode, mini-swe-agent), the first assistant turn of each. The points are the first
tokens the assistant writes (its tool call), seen with the harness's full system prompt and tools before them, so the
task is held fixed and only the harness differs.

    HF_HOME=/Volumes/Samsung_T5/hf uv run --extra compare python examples/harness_compare.py
"""
import argparse, json, os, pathlib

import torch

from flowscope.compare import capture, frames

HARNESSES = ["claude-code", "codex", "opencode", "mini-swe-agent"]
MODELS = [
    ("base", "LiquidAI/LFM2.5-2.6B"),
    ("multi-harness SFT", "FineEnvs/LFM2.5-2.6B-multiharness-SFT"),
    ("multi-harness RL", "FineEnvs/LFM2.5-2.6B-multiharness-RL"),
    ("OpenCode-only RL", "FineEnvs/LFM2.5-2.6B-opencode-RL"),
]
DATASET = "FineEnvs/SmolDataEnvs-multiharness-sft"


def pick_inputs(tasks, window, max_tokens):
    """The first assistant turn of `tasks` tasks that every harness ran, with at least `window` assistant tokens."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = pathlib.Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["lfm25_2_6b/*/*.parquet"]))
    meta = {}  # (harness, task) -> best row: first turn of the rollout, short enough
    for h in HARNESSES:
        # skip macOS "._" sidecar files (the cache may live on an exFAT drive)
        for f in sorted(f for f in (root / "lfm25_2_6b" / h).glob("*.parquet") if not f.name.startswith("._")):
            t = pq.read_table(f, columns=["task_id", "rollout_id", "num_tokens", "num_supervised_tokens", "common_task"])
            for i, (task, n, sup, common) in enumerate(zip(*(t[c].to_pylist() for c in ["task_id", "num_tokens", "num_supervised_tokens", "common_task"]))):
                if common and sup >= window and n <= max_tokens:
                    key = (h, task)
                    if key not in meta or n < meta[key][0]:
                        meta[key] = (n, f, i)
    shared = sorted(set.intersection(*({t for (hh, t) in meta if hh == h} for h in HARNESSES)))
    chosen = shared[:tasks]
    if len(chosen) < tasks:
        raise SystemExit(f"only {len(chosen)} tasks run in every harness fit the limits")
    rows = []
    for task in chosen:
        for h in HARNESSES:
            _, f, i = meta[(h, task)]
            r = pq.read_table(f, columns=["input_ids", "labels"]).slice(i, 1).to_pylist()[0]
            start = next(j for j, lab in enumerate(r["labels"]) if lab != -100)
            rows.append(dict(harness=h, task=task, ids=r["input_ids"], start=start))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--window", type=int, default=32, help="assistant tokens per (task, harness)")
    ap.add_argument("--max-tokens", type=int, default=17000, help="skip turns with longer contexts (Claude Code first turns are ~15k)")
    ap.add_argument("--k", type=int, default=128, help="principal directions kept from the hidden size")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/lfm25-harness-compare.jsonl")
    a = ap.parse_args()

    rows = pick_inputs(a.tasks, a.window, a.max_tokens)
    # order points by harness, then task, so each sequence of `window` tokens stays contiguous
    rows.sort(key=lambda r: (HARNESSES.index(r["harness"]), r["task"]))
    print("inputs:", [(r["harness"], r["task"], len(r["ids"]), r["start"]) for r in rows], flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[0][1])
    tokens = [i for r in rows for i in r["ids"][r["start"]:r["start"] + a.window]]
    texts = [tok.decode([i]) for i in tokens]
    groups = [HARNESSES.index(r["harness"]) for r in rows for _ in range(a.window)]

    captures, layer_types = {}, None
    for name, repo in MODELS:
        print(f"capturing {name} ({repo}) on {a.device}", flush=True)
        captures[name], layer_types = capture(repo, [r["ids"] for r in rows], [(r["start"], a.window) for r in rows], a.device)
        for other_name, other in captures.items():  # every checkpoint must see the exact same token positions
            assert other[0].shape == captures[name][0].shape, (other_name, name)

    kinds = ["attn" if t == "full_attention" else "conv" for t in (layer_types or [])]
    layer_names = ["emb"] + [f"b{i + 1}" + ("·attn" if k == "attn" else "") for i, k in enumerate(kinds or range(len(captures["base"]) - 1))]
    out = frames(captures, layer_names, tokens=tokens, texts=texts, groups=groups, group_names=HARNESSES, T=a.window,
                 title="LFM2.5-2.6B · harness training", k=a.k)

    path = pathlib.Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(f) + "\n" for f in out))
    print(f"\nwrote {path} ({os.path.getsize(path) // 1024} KB), top {a.k} directions keep {out[0]['retained']:.1%} of the spread")
    print("\nper layer vs base: linear CKA (1 = same representation) / median relative change of each token's vector")
    print("layer        " + "  ".join(f"{f['name'][:19]:>19}" for f in out[1:]))
    for li, ln in enumerate(layer_names):
        print(f"{ln:<12} " + "  ".join(f"{f['similarity'][li]:>9.3f} / {f['change'][li]:>6.1%}" for f in out[1:]))


if __name__ == "__main__":
    main()
