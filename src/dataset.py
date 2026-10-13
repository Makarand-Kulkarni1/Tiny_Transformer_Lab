import torch
from torch.utils.data import Dataset

IGNORE = -100    # PyTorch's cross_entropy skips targets with this value


class SQLDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows      # list of {"question": ..., "sql": ...}
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids, sql_start = self.tok.encode_example(row["question"], row["sql"])
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