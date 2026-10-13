import sys, json
import torch
from torch.utils.data import DataLoader
from src.tokenizer import CharTokenizer
from src.dataset import SQLDataset, collate, IGNORE

def load(name):
    return [json.loads(line) for line in open(f"data/raw/{name}.jsonl", encoding="utf-8")]

print("loading data...")
train, val, test = load("train"), load("val"), load("test")
print("sizes:", len(train), len(val), len(test))

print("building tokenizer...")
tok = CharTokenizer.build([r["question"] for r in train] + [r["sql"] for r in train])

print("building dataset...")
ds = SQLDataset(train, tok)
x, y = ds[0]
print("x:", x.shape, x.dtype)

print("building batch...")
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_id))
xb, yb = next(iter(loader))
print("batch shapes:", xb.shape, yb.shape)
print("done")