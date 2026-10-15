import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import PeftModel

from src.data.sql_runner import run_query, same_result
from src.evaluate import DB_BUILDERS, per_template, short, summarize
from src.lora_finetune import BASE_MODEL, PROMPT, load_base
from src.train import ROOT, load_jsonl


@torch.no_grad()
def generate_batch(model, tok, questions, device, max_new_tokens=200):
    """Greedy-decode SQL for a batch of questions. Returns (texts, n_generated_tokens)."""
    prompts = [PROMPT.format(question=q) for q in questions]
    tok.padding_side = "left"                       # generation needs left padding
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    texts, counts = [], []
    for row in out[:, enc["input_ids"].shape[1]:].tolist():
        if tok.eos_token_id in row:
            row = row[: row.index(tok.eos_token_id)]
        texts.append(tok.decode(row, skip_special_tokens=True).strip())
        counts.append(len(row) + 1)                 # generated tokens, plus <eos>
    return texts, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="lora")
    parser.add_argument("--data", default="v2", choices=["v1", "v2"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "paraphrase"])
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_base(args.base, device)
    adapter_dir = ROOT / "checkpoints" / args.run_name
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()   # bake the adapter in
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    adapter_mb = sum(f.stat().st_size for f in adapter_dir.glob("adapter_model*")) / 1e6
    print(f"device: {device} | parameters: {n_params} | adapter: {adapter_mb:.1f} MB")

    conn = DB_BUILDERS[args.data]()
    rows = load_jsonl(args.split, args.data)
    if args.limit and args.limit < len(rows):
        rows = random.Random(0).sample(rows, args.limit)
    print(f"evaluating {len(rows)} examples from the {args.data} {args.split} split")

    results = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        start = time.time()
        texts, counts = generate_batch(model, tok, [r["question"] for r in batch], device)
        per_example = (time.time() - start) / len(batch)
        for row, pred, n_tok in zip(batch, texts, counts):
            results.append({
                "question": row["question"], "gold": row["sql"], "predicted": pred,
                "exact": pred == row["sql"],
                "valid": run_query(conn, pred) is not None,
                "exec_correct": same_result(conn, row["sql"], pred),
                "seconds": per_example, "n_tokens": n_tok,
            })
        print(f"  {len(results)}/{len(rows)} done")

    summary = summarize(results)
    templates = per_template(results)
    summary.update({
        "macro_exec_accuracy": sum(s["exec_accuracy"] for s in templates.values()) / len(templates),
        "parameters": n_params, "checkpoint_mb": adapter_mb, "run_name": args.run_name,
        "data": args.data, "split": args.split, "device": device, "batch_size": args.batch_size,
    })

    print("\n=== overall ===")
    print(f"execution accuracy:       {summary['exec_accuracy']:.1%}")
    print(f"macro (per-template avg): {summary['macro_exec_accuracy']:.1%}")
    print(f"exact match:              {summary['exact_match']:.1%}")
    print(f"valid SQL:                {summary['valid_sql']:.1%}")
    print(f"speed (batched, {args.batch_size}):    {summary['tokens_per_second']:.0f} tokens/s")

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