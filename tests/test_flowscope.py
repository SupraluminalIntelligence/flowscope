import base64, importlib.util, json, pathlib, urllib.request

import pytest
import torch

from flowscope import FlowScope
from flowscope import cli
from flowscope.server import LiveServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpt_example", ROOT / "examples" / "gpt.py")
gpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt)

TINY = ["--embd", "32", "--heads", "2", "--block-size", "16", "--batch-size", "8"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FlowScope.runs.clear()
    (tmp_path / "input.txt").write_text("to be or not to be, that is the question. " * 200)


def train(flags, steps=21, every=10):
    cfg = gpt.parse_args(TINY + flags)
    torch.manual_seed(0)
    model = gpt.GPT(cfg, vocab_size=40)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scope = FlowScope(model, opt, every=every, label=" ".join(flags) or "baseline", sink=None)
    for _ in range(steps):
        x = torch.randint(0, 40, (8, 16))
        _, loss = model(x, torch.roll(x, -1, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():  # eval / generate forwards must not be mistaken for training steps
        model(torch.randint(0, 40, (1, 16)))
    return scope


def test_gradient_blockage_is_detected_and_residuals_remove_it():
    healthy = train(["--layers", "8"])
    blocked = train(["--layers", "8", "--no-residual", "--no-layernorm"])

    assert [h["step"] for h in healthy.hist] == [0, 10, 20]
    emb_grad = lambda s: dict(s.hist[-1]["profile"])[0]  # log10 of gradient at embeddings vs last block
    assert emb_grad(healthy) > -1
    assert emb_grad(blocked) < -4

    html = blocked.to_html()
    assert "BLOCKED" in html and "<svg" in html
    assert "--layers 8" in html  # the healthy run is drawn as a comparison line
    assert "healthy" in healthy.to_html().split("</b>")[1][:400]


def test_cli_attaches_to_unmodified_script_and_serves_live_frames(tmp_path):
    cli.main(["run", "--port", "8931", "--every", "5", "--record", str(tmp_path / "rec.jsonl"), str(ROOT / "examples" / "gpt.py"),
              "--iters", "12", "--data", str(tmp_path / "input.txt"), *TINY])
    server = LiveServer.shared(8931)

    assert "FlowScope ·" in server.frame and "step 10" in server.frame
    page = urllib.request.urlopen(server.url + "/").read().decode()
    assert "EventSource('/events')" in page
    with urllib.request.urlopen(server.url + "/events", timeout=5) as events:
        first = events.readline().decode()
    msg = json.loads(first[6:])
    assert first.startswith("data: ") and "step 10" in msg["html"]
    assert [l["name"] for l in msg["tour"]["layers"]] == ["emb", "b1", "b2", "b3", "b4"]
    assert "FlowTour" in page
    saved = json.loads((tmp_path / ".flowscope" / "runs.json").read_text())
    assert any(label.startswith("gpt.py --iters 12") for label in saved)
    frames = [json.loads(line) for line in (tmp_path / "rec.jsonl").read_text().splitlines()]
    assert [f["step"] for f in frames] == [5, 10] and frames[0]["label"].startswith("gpt.py")  # script's step numbers


def test_tour_shares_one_basis_and_shows_rank_collapse():
    healthy = train(["--layers", "8"], steps=1, every=1)
    blocked = train(["--layers", "8", "--no-residual"], steps=1, every=1)

    p = healthy.tour_payload
    assert (p["N"], p["C"], p["T"], len(p["tokens"])) == (128, 32, 16, 128)
    assert all(len(base64.b64decode(l["data"])) == p["N"] * p["C"] for l in p["layers"])
    dims = lambda s: [l["dim"] for l in s.tour_payload["layers"]]
    assert dims(healthy)[-1] > 0.7 * dims(healthy)[0]  # residual stream keeps its dimensionality at init
    assert dims(blocked)[-1] < 0.35 * dims(blocked)[0]  # without residuals it collapses toward a line


def test_compare_frames_share_one_basis_and_report_similarity(tmp_path):
    transformers = pytest.importorskip("transformers")
    from flowscope.compare import capture, frames, linear_cka

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=4)
    for name in ("a", "b"):
        transformers.LlamaModel(cfg).save_pretrained(tmp_path / name)
    seqs, windows = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 8, 7, 6, 5, 4]], [(2, 4), (1, 4)]
    a, _ = capture(str(tmp_path / "a"), seqs, windows, "cpu", dtype=torch.float32)
    b, _ = capture(str(tmp_path / "b"), seqs, windows, "cpu", dtype=torch.float32)
    assert len(a) == 4 and a[0].shape == (8, 32)  # embeddings + 3 layers, 2 windows x 4 tokens

    out = frames({"a": a, "a again": a, "b": b}, ["emb", "b1", "b2", "b3"], tokens=list(range(8)), texts=["x"] * 8,
                 groups=[0] * 4 + [1] * 4, group_names=["g0", "g1"], T=4, title="t", k=6)
    assert [f["name"] for f in out] == ["a", "a again", "b"] and all(f["C"] == 6 and f["N"] == 8 for f in out)
    assert out[0]["layers"][2]["data"] == out[1]["layers"][2]["data"]  # one shared basis: same input, same coordinates
    assert out[1]["similarity"] == [1.0] * 4 and min(out[2]["similarity"][1:]) < 0.999
    assert out[1]["change"] == [0.0] * 4 and min(out[2]["change"][1:]) > 0.1  # a different model moves the vectors
    assert abs(linear_cka(a[1], a[1] @ torch.linalg.qr(torch.randn(32, 32))[0] * 3) - 1) < 1e-6  # rotation/scale invariant
