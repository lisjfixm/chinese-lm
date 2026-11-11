# -*- coding: utf-8 -*-
"""
把訓練好的兩個模型匯出成「可內嵌進單一 HTML」的資料。

- MLP 權重：int8 對稱量化 + base64，體積約為 float32 的 1/4
- n-gram：bigram / trigram 的稀疏計數表（JSON）
- 其他：詞表、字的邊際次數、訓練曲線、指標數字
"""

import base64
import json
import os
from collections import defaultdict

import numpy as np
import torch

from common import SOS
from mlp_lm import CharMLPLM
from ngram_lm import NgramLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
TARGET = os.path.join(OUT_DIR, "web_model.json")


def quantize(t: torch.Tensor) -> dict:
    """int8 對稱量化：以最大絕對值定 scale，回傳 base64 與 scale。"""
    arr = t.detach().cpu().numpy().astype(np.float32)
    scale = float(np.abs(arr).max()) or 1.0
    scale /= 127.0
    q = np.rint(arr / scale).clip(-127, 127).astype(np.int8)
    return {"scale": scale, "data": base64.b64encode(q.tobytes()).decode("ascii")}


def export_ngram(lm: NgramLM) -> dict:
    """把二gram、三gram 的計數表轉成 {ctx: [[next, count], ...]}。"""
    def to_json(table):
        out = {}
        for ctx, counts in table.items():
            key = ",".join(str(c) for c in ctx)
            out[key] = [[int(n), int(c)] for n, c in counts.items() if c > 0]
        return out

    return {
        "bigram": to_json(lm.counts2),
        "trigram": to_json(lm.counts),
        "unigram": {str(k): int(v) for k, v in lm.unigram.items()},
        "total_uni": int(lm.total_uni),
    }


def main() -> None:
    mlp = torch.load(os.path.join(OUT_DIR, "mlp_model.pt"), map_location="cpu", weights_only=False)
    cfg = mlp["config"]
    model = CharMLPLM(len(mlp["itos"]), emb_dim=cfg["emb_dim"], ctx=cfg["ctx"], hidden=cfg["hidden"])
    model.load_state_dict(mlp["state_dict"])
    model.eval()

    ngram = NgramLM.load(os.path.join(OUT_DIR, "ngram_model.pkl"))

    with open(os.path.join(OUT_DIR, "train_history.json"), encoding="utf-8") as f:
        history = json.load(f)

    with open(os.path.join(OUT_DIR, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)

    # 位置實驗：同一批字排在不同槽位，下一個字的機率不同
    def probe(ctx_text: str) -> dict:
        ids = [mlp["vocab"].get(c, mlp["vocab"]["<UNK>"]) for c in ctx_text]
        hist = [mlp["vocab"][SOS]] * cfg["ctx"] + ids
        x = torch.tensor([hist[-cfg["ctx"]:]])
        with torch.no_grad():
            logits = model(x)[0]
        p = torch.softmax(logits, -1).numpy()
        top = p.argsort()[::-1][:3]
        return {"ctx": ctx_text,
                "top": [[mlp["itos"][int(i)], round(float(p[int(i)]) * 100, 1)] for i in top]}

    probes = [probe(c) for c in ["我的", "的我", "今天", "天今", "媽媽在", "在媽媽"]]

    # 對話模式的語料問答：頁面先找最接近的一句問法，再讓模型接出答句
    dialogs = []
    with open(os.path.join(HERE, "corpus.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "—" in line:
                q, a = line.split("—", 1)
                if q and a:
                    dialogs.append([q, a])

    data = {
        "meta": {
            "sentences": int(metrics["sentences"]),
            "vocab": len(mlp["itos"]),
            "ctx": cfg["ctx"],
            "emb": cfg["emb_dim"],
            "hidden": cfg["hidden"],
            "params": int(sum(p.numel() for p in model.parameters())),
            "ngram_bigram_ppl": metrics["ngram_2gram_ppl"],
            "ngram_trigram_ppl": metrics["ngram_3gram_ppl"],
            "mlp_train_ppl": metrics["mlp_train_ppl"],
            "mlp_val_ppl": metrics["mlp_val_ppl"],
            "train_samples": metrics["train_samples"],
            "val_samples": metrics["val_samples"],
            "epochs": int(metrics.get("epochs", len(history["loss"]) if False else 0)) or len(history["loss"]),
            "bigram_entries": len(NgramLM.load(os.path.join(OUT_DIR, "ngram_model.pkl")).counts2),
            "trigram_entries": len(ngram.counts),
            "probe": probes,
            "dialogs": dialogs,
        },
        "itos": list(mlp["itos"]),
        "sos": mlp["vocab"][SOS],
        "mlp": {
            "emb": quantize(model.emb.weight),
            "fc1w": quantize(model.fc1.weight),
            "fc1b": quantize(model.fc1.bias.reshape(-1, 1)),
            "fc2w": quantize(model.fc2.weight),
            "fc2b": quantize(model.fc2.bias.reshape(-1, 1)),
            "shape": {"emb": list(model.emb.weight.shape),
                      "fc1": list(model.fc1.weight.shape),
                      "fc2": list(model.fc2.weight.shape)},
        },
        "ngram": export_ngram(ngram),
        "loss": history["loss"],
    }

    # 詞表以 shift_jis 風格的 \uXXXX 轉義不必要，json 直接用 ensure_ascii=False
    with open(TARGET, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    size = os.path.getsize(TARGET)
    print(f"已匯出 {TARGET}　{size / 1024:.0f} KB")
    print(f"參數量 {data['meta']['params']}　bigram 條目 {len(data['ngram']['bigram'])}"
          f"　trigram 條目 {len(data['ngram']['trigram'])}")


if __name__ == "__main__":
    main()
