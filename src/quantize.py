import argparse
import copy
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data.sql_runner import same_result
from src.evaluate import DB_BUILDERS, predict
from src.train import ROOT, load_jsonl, load_model


def quantize_weights(model, bits):
    """Round every nn.Linear weight to `bits` bits, with one scale per output row (symmetric)."""
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                w = m.weight
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
                m.weight.copy_((w / scale).round().clamp(-qmax, qmax) * scale)


def size_mb(model, bits=None):
    """Size the model would have if Linear weights were stored in `bits` bits (rest stays fp32)."""
    linear_weights, scales = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear_weights += m.weight.numel()
            scales += m.weight.shape[0]
    total = sum(p.numel() for p in model.parameters())
    if bits is None:
        return total * 4 / 1e6
    other = total - linear_weights
    return (linear_weights * bits / 8 + scales * 4 + other * 4) / 1e6


def accuracy(model, tok, rows, conn, device):
    correct = 0
    for r in rows:
        pred = predict(model, tok, r["question"], device)
        correct += same_result(conn, r["sql"], pred)
    return correct / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="v2aug3")
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(ROOT / "checkpoints" / args.run_name / "best.pt", device)
    conn = DB_BUILDERS["v2"]()

    test_rows = random.Random(0).sample(load_jsonl("test", "v2"), 200)   # the same 200 as the API run
    para_rows = load_jsonl("paraphrase", "v2")
    print(f"device: {device} | {len(test_rows)} test + {len(para_rows)} paraphrase questions per variant")

    results = []
    for bits in [None] + args.bits:
        m = copy.deepcopy(model)
        if bits:
            quantize_weights(m, bits)
        m.eval()
        rec = {"label": "fp32" if bits is None else f"int{bits}", "bits": bits or 32,
               "size_mb": size_mb(m, bits),
               "test_acc": accuracy(m, tok, test_rows, conn, device),
               "para_acc": accuracy(m, tok, para_rows, conn, device)}
        results.append(rec)
        print(f"{rec['label']:>5s} | {rec['size_mb']:5.2f} MB | test {rec['test_acc']:.1%} | paraphrase {rec['para_acc']:.1%}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "quantization.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["size_mb"] for r in results], [r["test_acc"] * 100 for r in results], marker="o")
    for r in results:
        ax.annotate(r["label"], (r["size_mb"], r["test_acc"] * 100), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("model size (MB)")
    ax.set_ylabel("test execution accuracy (%)")
    ax.set_title("Quantization: size vs accuracy (from-scratch model)")
    fig.savefig(ROOT / "results" / "quantization.png", dpi=150, bbox_inches="tight")
    print("saved results/quantization.json and results/quantization.png")


if __name__ == "__main__":
    main()