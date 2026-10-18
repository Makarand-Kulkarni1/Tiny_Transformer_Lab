import argparse
import json
import random
import statistics
import time

import torch

from src.evaluate import predict
from src.train import ROOT, load_jsonl, load_model


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(run, questions, device):
    """Call run(question) -> n_tokens for each question; the first two calls are warm-up."""
    run(questions[0])
    run(questions[1])
    seconds, tokens = [], 0
    for q in questions[2:]:
        sync(device)
        start = time.perf_counter()
        tokens += run(q)
        sync(device)
        seconds.append(time.perf_counter() - start)
    return seconds, tokens


def bench_scratch(device, questions):
    ckpt = ROOT / "checkpoints" / "v2aug3" / "best.pt"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(ckpt, device)
    seconds, tokens = timed(lambda q: len(predict(model, tok, q, device)) + 1, questions, device)
    return {"name": "From scratch (852k)", "params": sum(p.numel() for p in model.parameters()),
            "disk_mb": ckpt.stat().st_size / 1e6, "seconds": seconds, "tokens": tokens,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None}


def bench_lora(device, questions):
    from peft import PeftModel
    from src.evaluate_lm import generate_batch
    from src.lora_finetune import BASE_MODEL, load_base

    adapter_dir = ROOT / "checkpoints" / "lora"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok, model = load_base(BASE_MODEL, device)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    adapter_mb = sum(f.stat().st_size for f in adapter_dir.glob("adapter_model*")) / 1e6
    seconds, tokens = timed(lambda q: generate_batch(model, tok, [q], device)[1][0], questions, device)
    return {"name": "LoRA Qwen2.5-0.5B", "params": n_params,
            "disk_mb": n_params * 2 / 1e6 + adapter_mb,       # base weights stored in 16 bits + adapter
            "seconds": seconds, "tokens": tokens,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--which", default="both", choices=["scratch", "lora", "both"])
    parser.add_argument("--n", type=int, default=20, help="number of timed questions")
    args = parser.parse_args()

    pool = random.Random(0).sample(load_jsonl("test", "v2"), 200)      # same questions for every model
    questions = [r["question"] for r in pool[: args.n + 2]]

    runs = []
    if args.which in ("scratch", "both"):
        runs.append(bench_scratch(args.device, questions))
    if args.which in ("lora", "both"):
        runs.append(bench_lora(args.device, questions))

    out_path = ROOT / "results" / f"benchmark_{args.device}.json"
    saved = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"\n=== speed on {args.device}: one question at a time, {args.n} timed questions ===")
    for r in runs:
        mean = statistics.mean(r["seconds"])
        info = {"params": r["params"], "disk_mb": r["disk_mb"], "mean_s": mean,
                "median_s": statistics.median(r["seconds"]), "tokens_per_s": r["tokens"] / sum(r["seconds"]),
                "peak_gpu_mb": r["peak_gpu_mb"], "n": args.n}
        saved[r["name"]] = info
        gpu = f" | peak GPU {info['peak_gpu_mb']:.0f} MB" if info["peak_gpu_mb"] else ""
        print(f"{r['name']:22s} {mean * 1000:8.0f} ms/question | {info['tokens_per_s']:6.0f} tokens/s | "
              f"disk {info['disk_mb']:7.1f} MB{gpu}")
    out_path.write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()