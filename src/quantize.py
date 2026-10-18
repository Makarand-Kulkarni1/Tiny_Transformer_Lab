import argparse
import copy
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sql_runner import same_result
from src.dataset import SQLDataset, collate
from src.evaluate import DB_BUILDERS, predict
from src.train import ROOT, evaluate, load_jsonl, load_model


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


def weight_error(original, quantized):
    """Relative size of the rounding error over the quantized weights."""
    num = den = 0.0
    for (name, a), (_, b) in zip(original.named_parameters(), quantized.named_parameters()):
        if a.ndim == 2 and "emb" not in name:
            num += (a - b).pow(2).sum().item()
            den += a.pow(2).sum().item()
    return (num / den) ** 0.5 if den else 0.0


def accuracy(model, tok, rows, conn, device):
    correct = 0
    for r in rows:
        pred = predict(model, tok, r["question"], device)
        correct += same_result(conn, r["sql"], pred)
    return correct / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="v2aug3")
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(ROOT / "checkpoints" / args.run_name / "best.pt", device)
    conn = DB_BUILDERS["v2"]()

    test_rows = random.Random(0).sample(load_jsonl("test", "v2"), 200)   # the same 200 as the API run
    para_rows = load_jsonl("paraphrase", "v2")
    val_loader = DataLoader(SQLDataset(load_jsonl("val", "v2"), tok), batch_size=64,
                            collate_fn=lambda b: collate(b, tok.pad_id))
    print(f"device: {device} | {len(test_rows)} test + {len(para_rows)} paraphrase questions per variant")

    results = []
    for bits in [None] + args.bits:
        m = copy.deepcopy(model)
        if bits:
            quantize_weights(m, bits)
        val = evaluate(m, val_loader, device)        # note: evaluate() leaves the model in train mode
        m.eval()                                     # so switch back before generating
        rec = {"label": "fp32" if bits is None else f"int{bits}", "bits": bits or 32,
               "size_mb": size_mb(m, bits), "weight_error": weight_error(model, m), "val_loss": val,
               "test_acc": accuracy(m, tok, test_rows, conn, device),
               "para_acc": accuracy(m, tok, para_rows, conn, device)}
        results.append(rec)
        print(f"{rec['label']:>5s} | {rec['size_mb']:5.2f} MB | weight error {rec['weight_error']:5.1%} | "
              f"val loss {rec['val_loss']:.4f} | test {rec['test_acc']:.1%} | paraphrase {rec['para_acc']:.1%}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "quantization.json", "w") as f:
        json.dump(results, f, indent=2)

    labels = [r["label"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(labels, [r["test_acc"] * 100 for r in results], marker="o", label="test (200 questions)")
    ax1.plot(labels, [r["para_acc"] * 100 for r in results], marker="o", label="paraphrase (108 questions)")
    ax1.set_xlabel("weight precision")
    ax1.set_ylabel("execution accuracy (%)")
    ax1.set_title("Accuracy holds, then collapses")
    ax1.legend()
    ax2.plot(labels, [r["val_loss"] for r in results], marker="o", color="tab:red")
    ax2.set_yscale("log")
    ax2.set_xlabel("weight precision")
    ax2.set_ylabel("validation loss (log scale)")
    ax2.set_title("Validation loss: the early warning")
    fig.savefig(ROOT / "results" / "quantization.png", dpi=150, bbox_inches="tight")
    print("saved results/quantization.json and results/quantization.png")


if __name__ == "__main__":
    main()