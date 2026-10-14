import random
import re
import string

import torch
from torch.utils.data import Dataset

IGNORE = -100    # PyTorch's cross_entropy skips targets with this value

QUOTED = re.compile(r"'([^']*)'")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CONSONANTS = "bdfghjklmnprstvyz"
VOWELS = "aeiou"


def random_word(rng):
    """A fresh 3 to 12 letter word: half pronounceable, half random letters."""
    length = rng.randint(3, 12)
    if rng.random() < 0.5:
        return "".join(rng.choice(CONSONANTS if i % 2 == 0 else VOWELS) for i in range(length))
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def augment_example(question, sql, rng, p=0.8):
    """Swap each quoted word value (not dates) for a random word, in both texts.
    The question and the SQL are changed consistently, so the SQL stays a correct copy."""
    for value in sorted(set(QUOTED.findall(sql))):      # sorted: reproducible order
        if DATE.fullmatch(value) or rng.random() > p:
            continue
        new = random_word(rng)
        question = re.sub(rf"\b{re.escape(value)}\b", new, question)
        sql = sql.replace(f"'{value}'", f"'{new}'")
    return question, sql


class SQLDataset(Dataset):
    def __init__(self, rows, tok, augment_p=0.0):
        self.rows = rows            # list of {"question": ..., "sql": ...}
        self.tok = tok
        self.augment_p = augment_p  # 0 = no augmentation (used for val and test)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        question, sql = row["question"], row["sql"]
        if self.augment_p > 0:
            question, sql = augment_example(question, sql, random, self.augment_p)
        ids, sql_start = self.tok.encode_example(question, sql)
        x = ids[:-1]                      # input: everything except the last token
        y = ids[1:]                       # target: the same sequence shifted by one
        for j in range(sql_start - 1):    # question positions: no loss
            y[j] = IGNORE
        return torch.tensor(x), torch.tensor(y)


def collate(batch, pad_id):
    """Pad a list of (x, y) pairs into two rectangular tensors."""
    B = len(batch)
    max_len = max(len(x) for x, _ in batch)
    xs = torch.full((B, max_len), pad_id)    # inputs padded with the pad id
    ys = torch.full((B, max_len), IGNORE)    # targets padded with "ignore"
    for i, (x, y) in enumerate(batch):
        xs[i, :len(x)] = x
        ys[i, :len(y)] = y
    return xs, ys