# -*- coding: utf-8 -*-
"""載入已儲存的兩個模型並生成文字，示範模型持久化與還原。"""

import os

import torch

from mlp_lm import CharMLPLM, generate as mlp_generate
from ngram_lm import NgramLM

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PROMPTS = ["早安", "今天", "我", "朋友", "夜晚"]


def demo_ngram() -> None:
    lm = NgramLM.load(os.path.join(OUT_DIR, "ngram_model.pkl"))
    print("== n-gram LM（從 outputs/ngram_model.pkl 還原）==")
    for p in PROMPTS:
        print(f"  「{p}」→ {lm.generate(p, temperature=0.6, top_k=5, seed=7)}")


def demo_mlp() -> None:
    data = torch.load(os.path.join(OUT_DIR, "mlp_model.pt"), map_location="cpu", weights_only=False)
    cfg = data["config"]
    model = CharMLPLM(len(data["itos"]), emb_dim=cfg["emb_dim"],
                      ctx=cfg["ctx"], hidden=cfg["hidden"])
    model.load_state_dict(data["state_dict"])
    print("\n== MLP LM（從 outputs/mlp_model.pt 還原）==")
    for p in PROMPTS:
        print(f"  「{p}」→ {mlp_generate(model, p, data['vocab'], data['itos'], temperature=0.7, top_k=5, seed=7)}")


if __name__ == "__main__":
    demo_ngram()
    demo_mlp()
