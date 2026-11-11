# -*- coding: utf-8 -*-
"""
從零實作的小型 MLP 神經語言模型（學習示範，PyTorch CPU）。

模型架構（參考經典的 n-gram MLP LM / makemore 思路）：
    前 C 個字 → 查表取字向量 Embedding → 攤平 → 隱藏層(tanh) → Softmax → 下一個字

對比 n-gram：神經模型用稠密的字向量表示相似性，不會因為「沒見過這組合」
就直接失效；但資料很少時仍會背誦語料、泛化有限。
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from common import EOS, SOS, build_vocab, decode, encode, load_sentences


class CharMLPLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, ctx: int = 4, hidden: int = 128):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(vocab_size, emb_dim)          # 字向量查表
        self.fc1 = nn.Linear(ctx * emb_dim, hidden)           # 隱藏層
        self.fc2 = nn.Linear(hidden, vocab_size)               # 輸出詞表分數

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) 的 token id
        e = self.emb(x)                                        # (B, C, emb)
        e = e.view(x.size(0), -1)                             # (B, C*emb)
        h = torch.tanh(self.fc1(e))
        return self.fc2(h)


def build_dataset(sentences: list[str], vocab: dict[str, int], ctx: int):
    """把每句滑動成 (前文 C 字, 下一字) 的訓練樣本。"""
    sos_id, eos_id = vocab[SOS], vocab[EOS]
    X, Y = [], []
    for s in sentences:
        ids = encode(s, vocab)
        seq = [sos_id] * ctx + ids + [eos_id]
        for i in range(ctx, len(seq)):
            X.append(seq[i - ctx:i])
            Y.append(seq[i])
    return torch.tensor(X), torch.tensor(Y)


@torch.no_grad()
def perplexity(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
               loss_fn: nn.Module) -> float:
    model.eval()
    logits = model(X)
    loss = loss_fn(logits, Y)
    return float(torch.exp(loss))


def generate(model: CharMLPLM, prompt: str, vocab: dict[str, int], itos: list[str],
             max_len: int = 40, temperature: float = 1.0, top_k: int | None = None,
             seed: int | None = None) -> str:
    model.eval()
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    sos_id, eos_id = vocab[SOS], vocab[EOS]
    history = [sos_id] * model.ctx + encode(prompt, vocab)
    generated = encode(prompt, vocab)

    for _ in range(max_len):
        ctx_ids = history[-model.ctx:]
        x = torch.tensor([ctx_ids])
        logits = model(x)[0] / temperature
        if top_k is not None:
            v, idx = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float("-inf"))
            mask[idx] = 0.0
            logits = logits + mask
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1, generator=g).item())

        if nxt == eos_id:
            break
        generated.append(nxt)
        history.append(nxt)
        if len(generated) >= 6 and generated[-3:] == generated[-6:-3]:
            break
    return decode(generated, itos)


def main() -> None:
    ap = argparse.ArgumentParser(description="從零實作的小型 MLP 中文語言模型")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    sentences = load_sentences(args.corpus)
    vocab, itos = build_vocab(sentences)
    V = len(vocab)
    print(f"語料句子數：{len(sentences)}　詞表大小：{V}")

    # 留 10% 當驗證集，觀察是否背誦（過擬合）
    split = int(len(sentences) * 0.9)
    train_sents, val_sents = sentences[:split], sentences[split:]
    Xtr, Ytr = build_dataset(train_sents, vocab, args.ctx)
    Xva, Yva = build_dataset(val_sents, vocab, args.ctx)
    print(f"訓練樣本：{len(Xtr)}　驗證樣本：{len(Xva)}")

    model = CharMLPLM(V, emb_dim=args.emb, ctx=args.ctx, hidden=args.hidden)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    history = []
    n = len(Xtr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            logits = model(Xtr[idx])
            loss = loss_fn(logits, Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.detach().item() * len(idx)
        avg = total_loss / n
        history.append(avg)

        if epoch % 5 == 0 or epoch == 1:
            tr_ppl = perplexity(model, Xtr, Ytr, loss_fn)
            va_ppl = perplexity(model, Xva, Yva, loss_fn)
            print(f"epoch {epoch:3d}　loss {avg:.4f}　訓練困惑度 {tr_ppl:.2f}　驗證困惑度 {va_ppl:.2f}")

    # 存損失曲線圖
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump({"loss": history, "epochs": args.epochs}, f)
    tr_ppl = perplexity(model, Xtr, Ytr, loss_fn)
    va_ppl = perplexity(model, Xva, Yva, loss_fn)
    mpath = os.path.join(out_dir, "metrics.json")
    mdata = {}
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            mdata = json.load(f)
    mdata.update({"mlp_train_ppl": round(tr_ppl, 2), "mlp_val_ppl": round(va_ppl, 2),
                  "train_samples": int(len(Xtr)), "val_samples": int(len(Xva)),
                  "ctx": args.ctx, "emb": args.emb, "hidden": args.hidden,
                  "sentences": len(sentences), "vocab": V})
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(mdata, f, ensure_ascii=False, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(history, linewidth=1.5)
        plt.xlabel("epoch"); plt.ylabel("cross-entropy loss")
        plt.title("MLP LM training loss")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120)
    except Exception as e:
        print(f"（跳過損失曲線圖：{e}）")

    prompts = ["今天", "我", "天氣", "朋友", "這座", "夜晚"]
    print("\n生成範例：")
    for p in prompts:
        out = generate(model, p, vocab, itos, temperature=args.temperature,
                       top_k=args.top_k, seed=args.seed)
        print(f"  「{p}」→ {out}")

    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "itos": itos,
                "config": {"emb_dim": args.emb, "ctx": args.ctx, "hidden": args.hidden}},
               os.path.join(out_dir, "mlp_model.pt"))
    print(f"\n模型權重已儲存至 outputs/mlp_model.pt")


if __name__ == "__main__":
    main()
