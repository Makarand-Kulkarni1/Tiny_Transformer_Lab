import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.data.make_dataset import skeleton
from src.data.make_db import build_db
from src.data.make_db_v2 import build_db_v2
from src.data.sql_runner import run_query, same_result
from src.train import ROOT, load_jsonl, load_model

DB_BUILDERS = {"v1": build_db, "v2": build_db_v2}


@torch.no_grad()
def predict(model, tok, question, device, max_new_tokens=260):
    """Greedy-decode the SQL for one question."""
    prompt = torch.tensor([tok.encode(question) + [tok.sep_id]], device=device)
    out = model.generate(prompt, max_new_tokens=max_new_tokens, eos_id=tok.eos_id)
    return tok.decode(out[0, prompt.shape[1]:].tolist())


def evaluate_split(model, tok, rows, conn, device):
    """Generate SQL for every row and score it against the gold SQL."""
    results = []
    for i, row in enumerate(rows, start=1):
        start = time.time()
        pred = predict(model, tok, row["question"], device)
        seconds = time.time() - start
        results.append({
            "question": row["question"],
            "gold": row["sql"],
            "predicted": pred,
            "exact": pred == row["sql"],
            "valid": run_query(conn, pred) is not None,
            "exec_correct": same_result(conn, row["sql"], pred),
            "seconds": seconds,
            "n_tokens": len(pred) + 1,       # characters generated, plus <eos>
        })
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} done")
    return results


def summarize(results):
    n = len(results)
    total_seconds = sum(r["seconds"] for r in results)
    total_tokens = sum(r["n_tokens"] for r in results)
    return {
        "n": n,
        "exec_accuracy": sum(r["exec_correct"] for r in results) / n,
        "exact_match": sum(r["exact"] for r in results) / n,
        "valid_sql": sum(r["valid"] for r in results) / n,
        "avg_seconds": total_seconds / n,
        "tokens_per_second": total_tokens / total_seconds,
    }


def per_template(results):
    groups = defaultdict(list)
    for r in results:
        groups[skeleton(r["gold"])].append(r)
    return {key: summarize(group) for key, group in sorted(groups.items())}


def short(key):
    """Drop the repeated JOIN conditions so template labels are readable."""
    return (key.replace(" ON orders.customer_id = customers.id", "")
               .replace(" ON orders.product_id = products.id", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="base")
    parser.add_argument("--data", default="v1", choices=["v1", "v2"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "paraphrase"])
    parser.add_argument("--ckpt", default=None, help="checkpoint path (default: best.pt of the run)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate a random subset of this size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / args.run_name / "best.pt"
    model, tok = load_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = ckpt_path.stat().st_size / 1e6
    print(f"device: {device} | parameters: {n_params} | checkpoint: {ckpt_path}")

    # IMPORTANT: this must be the same database the dataset was generated against
    conn = DB_BUILDERS[args.data]()
    rows = load_jsonl(args.split, args.data)
    if args.limit and args.limit < len(rows):
        rows = random.Random(0).sample(rows, args.limit)
    print(f"evaluating {len(rows)} examples from the {args.data} {args.split} split")

    results = evaluate_split(model, tok, rows, conn, device)
    summary = summarize(results)
    templates = per_template(results)
    summary.update({
        "macro_exec_accuracy": sum(s["exec_accuracy"] for s in templates.values()) / len(templates),
        "parameters": n_params, "checkpoint_mb": size_mb, "run_name": args.run_name,
        "data": args.data, "split": args.split, "device": device,
    })

    print("\n=== overall ===")
    print(f"execution accuracy:       {summary['exec_accuracy']:.1%}")
    print(f"macro (per-template avg): {summary['macro_exec_accuracy']:.1%}")
    print(f"exact match:              {summary['exact_match']:.1%}")
    print(f"valid SQL:                {summary['valid_sql']:.1%}")
    print(f"speed:                    {summary['tokens_per_second']:.0f} tokens/s "
          f"({summary['avg_seconds']:.2f} s per question)")

    print("\n=== per template ===")
    print(f"{'template':95s} {'n':>4s} {'exec':>7s}")
    for key, s in templates.items():
        print(f"{short(key)[:95]:95s} {s['n']:4d} {s['exec_accuracy']:7.1%}")

    failures = [r for r in results if not r["exec_correct"]]
    print(f"\n=== failures: {len(failures)} (showing up to 5) ===")
    for r in failures[:5]:
        print("Q:    ", r["question"])
        print("gold: ", r["gold"])
        print("pred: ", r["predicted"])
        print()

    out = ROOT / "results" / f"{args.run_name}_{args.split}_eval.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_template": templates, "examples": results}, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()