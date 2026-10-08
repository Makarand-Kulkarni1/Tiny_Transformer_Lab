PAD, UNK, SEP, EOS = "<pad>", "<unk>", "<sep>", "<eos>"
SPECIALS = [PAD, UNK, SEP, EOS]


class CharTokenizer:
    def __init__(self, chars):
        # vocabulary = special tokens first, then every character in a fixed (sorted) order
        self.tokens = SPECIALS + sorted(chars)
        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}   # token -> id
        self.itos = {i: tok for i, tok in enumerate(self.tokens)}   # id -> token

    @classmethod
    def build(cls, texts):
        """Make a tokenizer from a list of strings."""
        return cls(set("".join(texts)))

    @property
    def vocab_size(self):
        return len(self.tokens)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def unk_id(self):
        return self.stoi[UNK]

    @property
    def sep_id(self):
        return self.stoi[SEP]

    @property
    def eos_id(self):
        return self.stoi[EOS]

    def encode(self, text):
        """Text -> list of ids. Unknown characters become <unk>."""
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        """List of ids -> text. Special tokens are skipped."""
        return "".join(self.itos[i] for i in ids if self.itos[i] not in SPECIALS)

    def encode_example(self, question, sql):
        """Question + <sep> + SQL + <eos> as ids, plus the index where the SQL starts."""
        ids = self.encode(question) + [self.sep_id] + self.encode(sql) + [self.eos_id]
        sql_start = len(question) + 1      # one id per question character, plus the <sep>
        return ids, sql_start