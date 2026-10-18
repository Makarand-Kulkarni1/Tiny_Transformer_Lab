import torch

from src.model import GPTConfig, TinyGPT
from src.tokenizer import CharTokenizer


def load_slim(path, device="cpu"):
    """Load the small deployable checkpoint made by export_model.py (attention masks are rebuilt, not stored)."""
    ckpt = torch.load(path, map_location=device)
    tok = CharTokenizer(ckpt["chars"])
    model = TinyGPT(GPTConfig(**ckpt["model_config"])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if unexpected or any(not k.endswith(".mask") for k in missing):
        raise RuntimeError(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    model.eval()
    return model, tok