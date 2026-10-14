import argparse
import json
import math
import random
import string
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import SQLDataset, collate, IGNORE
from src.model import GPTConfig, TinyGPT
from src.tokenizer import CharTokenizer, SPECIALS

ROOT = Path(__file__).resolve().parents[1]   # the project folder
DATA_DIRS = {"v1": "raw", "v2": "v2"}         # which folder under data/ each version lives in


@dataclass
class TrainConfig:
    run_name: str = "run"
    data: str = "v1"
    max_steps: int = 3000
    batch_size: int = 32
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 100
    patience: int = 10        # stop if val loss has not improved for this many checks
    seed: int = 0
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


def load_jsonl(name, data="v1"):
    path = ROOT / "data" / DATA_DIRS[data] / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_lr(step, cfg):
    """Linear warmup, then cosine decay from max_lr down to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device):
    """Average loss per target token over a whole dataset."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        n = (yb != IGNORE).sum().item()      # how many real targets in this batch
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / total_tokens


def save_checkpoint(path, model, model_cfg, tok, cfg, step, val_loss):
    torch.save({
        "model_state": model.state_dict(),
        "model_config": asdict(model_cfg),
        "chars": tok.tokens[len(SPECIALS):],   # lets us rebuild the identical tokenizer later
        "train_config": asdict(cfg),
        "step": step,
        "val_loss": val_loss,
    }, path)


def load_model(path, device="cpu"):
    """Rebuild a trained model and its tokenizer from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    tok = CharTokenizer(ckpt["chars"])
    model = TinyGPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


def train(cfg):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_rows = load_jsonl("train", cfg.data)
    val_rows = load_jsonl("val", cfg.data)
    test_rows = load_jsonl("test", cfg.data)   # only used to size the context window

    # the context window must fit the longest example in any split
    longest = max(len(r["question"]) + len(r["sql"]) + 2 for r in train_rows + val_rows + test_rows)
    block_size = ((longest + 15) // 16) * 16
    print(f"data: {cfg.data} | train {len(train_rows)}, val {len(val_rows)} | "
          f"longest example {longest} -> block_size {block_size}")

    # the tokenizer is built from training text plus every lowercase letter and digit,
    # so a held-out value with a rare letter never becomes <unk>
    texts = [r["question"] for r in train_rows] + [r["sql"] for r in train_rows]
    tok = CharTokenizer.build(texts + [string.ascii_lowercase + string.digits])

    collate_fn = lambda batch: collate(batch, tok.pad_id)
    train_loader = DataLoader(SQLDataset(train_rows, tok), batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SQLDataset(val_rows, tok), batch_size=64,
                            shuffle=False, collate_fn=collate_fn)

    model_cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size, n_layer=cfg.n_layer,
                          n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout)
    model = TinyGPT(model_cfg).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)

    ckpt_dir = ROOT / "checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    history = []
    best_val, bad_evals = float("inf"), 0
    running, running_n = 0.0, 0
    step, stop = 0, False
    start = time.time()
    model.train()

    while step < cfg.max_steps and not stop:
        for xb, yb in train_loader:
            if step >= cfg.max_steps:
                break

            lr = get_lr(step, cfg)
            for group in opt.param_groups:
                group["lr"] = lr

            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running += loss.item()
            running_n += 1
            step += 1

            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                val_loss = evaluate(model, val_loader, device)
                train_loss = running / running_n
                running, running_n = 0.0, 0

                if val_loss < best_val:
                    best_val, bad_evals = val_loss, 0
                    save_checkpoint(ckpt_dir / "best.pt", model, model_cfg, tok, cfg, step, val_loss)
                    mark = "  <- best"
                else:
                    bad_evals += 1
                    mark = ""

                history.append({"step": step, "train_loss": train_loss,
                                "val_loss": val_loss, "lr": lr})
                with open(results_dir / f"{cfg.run_name}_history.json", "w") as f:
                    json.dump(history, f, indent=2)

                print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                      f"lr {lr:.2e} | {time.time() - start:.0f}s{mark}")

                if bad_evals >= cfg.patience:
                    print(f"no improvement for {cfg.patience} checks, stopping early")
                    stop = True
                    break

    print(f"done. best val loss: {best_val:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    for name, default in asdict(TrainConfig()).items():
        parser.add_argument(f"--{name}", type=type(default), default=default)
    cfg = TrainConfig(**vars(parser.parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()