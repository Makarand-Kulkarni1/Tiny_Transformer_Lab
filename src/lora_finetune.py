import argparse
import json
import random
import time
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train import ROOT, get_lr, load_jsonl

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = "Question: {question}\nSQL:"
IGNORE = -100


def load_base(name=BASE_MODEL, device="cuda"):
    """Download the pretrained model (in fp32) and its tokenizer."""
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except TypeError:                       # older transformers versions call it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    return tok, model.float().to(device)


class LMDataset(Dataset):
    """Each item is the prompt plus the SQL plus <eos>; the loss covers only the SQL part."""

    def __init__(self, rows, tok):
        self.items = []
        for r in rows:
            prompt_ids = tok(PROMPT.format(question=r["question"]), add_special_tokens=False).input_ids
            sql_ids = tok(" " + r["sql"], add_special_tokens=False).input_ids + [tok.eos_token_id]
            self.items.append((prompt_ids + sql_ids, [IGNORE] * len(prompt_ids) + sql_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    """Pad a batch to one rectangle. Padding gets no loss and is hidden by the mask."""
    max_len = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_id)
    labels = torch.full((len(batch), max_len), IGNORE)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        mask[i, :len(ids)] = 1
    return input_ids, mask, labels


@torch.no_grad()
def val_loss(model, loader, device):
    """Average loss per target token over the validation set."""
    model.eval()
    total, count = 0.0, 0
    for input_ids, mask, labels in loader:
        input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        n = (labels[:, 1:] != IGNORE).sum().item()      # real targets (the model shifts labels by one)
        total += loss.item() * n
        count += n
    model.train()
    return total / count


def train(args):
    assert torch.cuda.is_available(), "this needs the GPU runtime (Runtime -> Change runtime type -> T4 GPU)"
    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok, model = load_base(args.base, device)
    lora = LoraConfig(
        r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_rows, val_rows = load_jsonl("train", args.data), load_jsonl("val", args.data)
    collate_fn = lambda batch: collate(batch, tok.pad_token_id)
    train_loader = DataLoader(LMDataset(train_rows, tok), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(LMDataset(val_rows, tok), batch_size=32, shuffle=False, collate_fn=collate_fn)

    max_steps = args.epochs * len(train_loader)
    schedule = SimpleNamespace(max_lr=args.lr, min_lr=args.lr / 10, warmup_steps=args.warmup, max_steps=max_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")          # keeps fp16 gradients from underflowing
    print(f"train {len(train_rows)}, val {len(val_rows)} | {max_steps} steps "
          f"({args.epochs} epochs x {len(train_loader)})")

    ckpt_dir = ROOT / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    history, best_val = [], float("inf")
    running, running_n, step = 0.0, 0, 0
    start = time.time()
    model.train()

    for epoch in range(args.epochs):
        for input_ids, mask, labels in train_loader:
            lr = get_lr(step, schedule)
            for group in opt.param_groups:
                group["lr"] = lr

            input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            running_n += 1
            step += 1

            if step % args.eval_every == 0 or step == max_steps:
                v = val_loss(model, val_loader, device)
                train_loss = running / running_n
                running, running_n = 0.0, 0
                mark = ""
                if v < best_val:
                    best_val, mark = v, "  <- best"
                    model.save_pretrained(ckpt_dir)          # saves only the small adapter
                history.append({"step": step, "train_loss": train_loss, "val_loss": v, "lr": lr})
                with open(ROOT / "results" / f"{args.run_name}_history.json", "w") as f:
                    json.dump(history, f, indent=2)
                print(f"step {step:5d} | train {train_loss:.4f} | val {v:.4f} | "
                      f"lr {lr:.2e} | {time.time() - start:.0f}s{mark}")

    print(f"done. best val loss: {best_val:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="lora")
    parser.add_argument("--data", default="v2", choices=["v1", "v2"])
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()