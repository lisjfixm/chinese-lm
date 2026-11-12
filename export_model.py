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
import re
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


def export_ngram(lm: NgramLM, bigram_topk: int = 10) -> dict:
    """把 n-gram 計數表轉成 {ctx: [[next, count], ...]}。

    網頁單檔體積有限：只帶 bigram（每個前文保留前 top-k 個後接字），
    trigram 表留在本機 pkl，不內嵌進 HTML。
    """
    def to_json(table, topk):
        out = {}
        for ctx, counts in table.items():
            key = ",".join(str(c) for c in ctx)
            items = sorted(counts.items(), key=lambda kv: -kv[1])[:topk]
            out[key] = [[int(n), int(c)] for n, c in items if c > 0]
        return out

    return {
        "bigram": to_json(lm.counts2, bigram_topk),
        "trigram": {},
        "unigram": {str(k): int(v) for k, v in lm.unigram.items()},
        "total_uni": int(lm.total_uni),
    }


def token_type(ch: str) -> str:
    if ch in ("<S>", "</S>", "<UNK>"):
        return "sp"
    if ch == " ":
        return "space"
    if re.match(r"[0-9]", ch):
        return "digit"
    if re.match(r"[a-z]", ch):
        return "en"
    if re.match(r"[一-鿿]", ch):
        return "zh"
    return "punct"


QWEN_DIR = os.path.join(HERE, "data-src", "qwen")
QWEN_BIN = os.path.join(QWEN_DIR, "token_embd_q4k.bin")
QWEN_VOCAB = os.path.join(QWEN_DIR, "vocab-full.txt")
QWEN_ROW_BYTES = 2016


def _own_pca(E: np.ndarray) -> np.ndarray:
    Ec = E - E.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    co = Ec @ Vt[:2].T
    return co / (np.abs(co).max() + 1e-9)


def _qwen_vocab_map() -> dict | None:
    if not os.path.exists(QWEN_VOCAB):
        return None
    m: dict[str, int] = {}
    for line in open(QWEN_VOCAB, encoding="utf-8"):
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                m[p[1]] = int(p[0])
            except ValueError:
                pass
    return m


def export_embmap(model: CharMLPLM, itos: list[str]):
    """每個字元的二維座標。

    優先使用本地 data-src/qwen 的 Qwen2.5-7B 嵌入（3584 維，PCA 投影二維）；
    對應不到的字回退到本模型自訓的 32 維向量 PCA。回傳 (embmap, 對應數, 來源)。
    """
    E = model.emb.weight.detach().cpu().numpy().astype("float64")
    own = _own_pca(E)

    qmap: dict | None = None
    n_match = 0
    source = "own-mlp"
    vmap = _qwen_vocab_map()
    if vmap and os.path.exists(QWEN_BIN):
        try:
            from gguf import dequantize, GGMLQuantizationType
            from opencc import OpenCC
            cc = OpenCC("t2s")
            resolve: dict[str, int] = {}
            for ch in itos:
                if ch in vmap:
                    resolve[ch] = vmap[ch]
                else:
                    simp = cc.convert(ch)
                    if simp in vmap:
                        resolve[ch] = vmap[simp]
            if resolve:
                raw = np.frombuffer(open(QWEN_BIN, "rb").read(), dtype=np.uint8)
                names = list(resolve)
                M = []
                for ch in names:
                    i = resolve[ch]
                    row = raw[i * QWEN_ROW_BYTES:(i + 1) * QWEN_ROW_BYTES]
                    M.append(np.asarray(dequantize(row, GGMLQuantizationType.Q4_K), dtype="float32"))
                M = np.stack(M)
                Mc = M - M.mean(0, keepdims=True)
                U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
                qco = Mc @ Vt[:2].T
                qco = qco / (np.abs(qco).max() + 1e-9)
                qmap = {names[i]: qco[i] for i in range(len(names))}
                n_match = len(names)
                source = "qwen2.5-7b"
                # Qwen 詞表沒有的生僻字：在自訓 32 維空間找最近鄰，借用其 Qwen 座標
                stoi = {c: i for i, c in enumerate(itos)}
                En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
                matched_idx = [stoi[c] for c in names]
                Mown = En[matched_idx]
                n_borrow = 0
                for i, ch in enumerate(itos):
                    if ch in qmap or token_type(ch) == "sp":
                        continue
                    sims = Mown @ En[i]
                    j = matched_idx[int(np.argmax(sims))]
                    qmap[ch] = qmap[itos[j]]
                    n_borrow += 1
        except Exception as e:  # 本機缺檔案或套件時退回自訓向量
            print("Qwen 嵌入讀取失敗，改用自訓向量：", e)
            qmap = None
            n_borrow = 0

    emb = []
    for i, ch in enumerate(itos):
        if qmap and ch in qmap:
            x, y = qmap[ch]
        else:
            x, y = own[i]
        emb.append([ch, round(float(x), 3), round(float(y), 3), token_type(ch)])
    return emb, n_match, source, n_borrow


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

    probes = [probe(c) for c in ["我的", "的我", "今天", "天今", "媽媽在", "在媽媽", "i l", "th", "the "]]

    # 對話模式的語料問答：頁面先找最接近的一句問法，再讓模型接出答句
    dialogs = []
    with open(os.path.join(HERE, "corpus.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "—" not in line:
                continue
            q, a = line.split("—", 1)
            q, a = q.strip(), a.strip().strip("—－-").strip()
            # 只保留真正的問答：問句以問號收尾，答話裡不再夾破折號
            if len(q) >= 2 and len(a) >= 1 and q[-1] in "？?" and "—" not in a:
                dialogs.append([q, a])

    embmap, qwen_matched, emb_source, qwen_borrowed = export_embmap(model, mlp["itos"])

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
            "emb_source": emb_source,
            "qwen_matched": qwen_matched,
            "qwen_borrowed": qwen_borrowed,
        },
        "itos": list(mlp["itos"]),
        "embmap": embmap,
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
