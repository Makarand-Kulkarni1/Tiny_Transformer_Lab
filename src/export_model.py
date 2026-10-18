import argparse
import random
from pathlib import Path

import torch

from src.evaluate import predict
from src.serve import load_slim
from src.train import ROOT, load_jsonl, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="v2aug3")
    parser.add_argument("--ckpt", default=None, help="path to the training checkpoint (default: saved_models/<run>/best.pt)")
    args = parser.parse_args()

    src_path = Path(args.ckpt) if args.ckpt else ROOT / "saved_models" / args.run_name / "best.pt"
    if not src_path.exists():
        raise SystemExit(f"checkpoint not found: {src_path}\n"
                         "Download v2aug3/best.pt from Drive into saved_models/v2aug3/")

    ckpt = torch.load(src_path, map_location="cpu")
    slim = {
        "model_state": {k: v for k, v in ckpt["model_state"].items() if not k.endswith(".mask")},
        "model_config": ckpt["model_config"],
        "chars": ckpt["chars"],
    }
    out = ROOT / "saved_models" / "tiny_sql.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(slim, out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB; original {src_path.stat().st_size / 1e6:.2f} MB)")

    # safety check: the slim model must write exactly the same SQL as the original
    full_model, full_tok = load_model(src_path, "cpu")
    slim_model, slim_tok = load_slim(out, "cpu")
    questions = [r["question"] for r in random.Random(0).sample(load_jsonl("test", "v2"), 20)]
    same = sum(predict(full_model, full_tok, q, "cpu") == predict(slim_model, slim_tok, q, "cpu")
               for q in questions)
    print(f"slim model matches the original on {same} of {len(questions)} questions")
    assert same == len(questions), "the slim model differs from the original"


if __name__ == "__main__":
    main()