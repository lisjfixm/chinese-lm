# -*- coding: utf-8 -*-
"""
把訓練好的模型資料匯出成純文字檔「模型.txt」。

來源：
  outputs/metrics.json      訓練指標
  outputs/train_history.json 每輪損失
  outputs/ngram_model.pkl   單字 / 二字 / 三字統計表
  outputs/mlp_model.pt      MLP 權重（emb / fc1 / fc2）

本檔方便在 GitHub 上直接閱讀；要在 Python 裡重現推論，仍應使用
ngram_model.pkl 與 mlp_model.pt（或匯出用的 outputs/web_model.json）。
"""

import json
import os
import pickle

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def fmt(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def main() -> None:
    metrics = json.load(open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
    history = json.load(open(os.path.join(OUT, "train_history.json"), encoding="utf-8"))

    ng = pickle.load(open(os.path.join(OUT, "ngram_model.pkl"), "rb"))
    pt = torch.load(os.path.join(OUT, "mlp_model.pt"), map_location="cpu", weights_only=False)
    sd = pt["state_dict"]
    itos = pt["itos"]
    vocab = pt["vocab"]

    lines: list[str] = []
    w = lines.append

    w("模型資料（純文字匯出）")
    w("=" * 40)
    w("本檔由 dump_model_txt.py 從 outputs/ 下的訓練產物匯出，即「訓練好的數據」。")
    w("特殊 token：<S> 句首、</S> 句尾、<UNK> 未收錄字。")
    w("")

    # 一、總覽
    w("一、總覽")
    w("-" * 40)
    w(f"語料句數：{metrics['sentences']}")
    w(f"詞表大小：{metrics['vocab']}")
    w(f"MLP 上下文長度 ctx：{metrics['ctx']}")
    w(f"MLP 字向量維度 emb：{metrics['emb']}")
    w(f"MLP 隱藏層維度 hidden：{metrics['hidden']}")
    w(f"MLP 參數量：{sum(t.numel() for t in sd.values())}")
    w(f"訓練樣本數：{metrics['train_samples']}")
    w(f"驗證樣本數：{metrics['val_samples']}")
    w(f"MLP 訓練輪數 epochs：{history.get('epochs', 24)}")
    w(f"n-gram bigram 困惑度：{metrics['ngram_2gram_ppl']}")
    w(f"n-gram trigram 困惑度：{metrics['ngram_3gram_ppl']}")
    w(f"MLP 訓練困惑度：{metrics['mlp_train_ppl']}")
    w(f"MLP 驗證困惑度：{metrics['mlp_val_ppl']}")
    w("")
    w("訓練過程每輪損失（cross-entropy）：")
    for i, loss in enumerate(history["loss"], start=1):
        w(f"  epoch {i:2d}: {loss:.4f}")
    w("")

    # 二、詞表
    w("二、詞表（id → 字）")
    w("-" * 40)
    for i, ch in enumerate(itos):
        w(f"{i}\t{ch}")
    w("")

    # 三、n-gram 統計
    w("三、n-gram 統計表")
    w("-" * 40)
    w(f"平滑：add-k，k = {ng['k']}；線性插值權重 trigram 0.7 / bigram 0.2 / unigram 0.1。")
    w("詞表擴大後 n-gram 表很大，本檔只摘錄可讀的摘要；完整計數表在 outputs/ngram_model.pkl。")
    w("")
    w("3.1 單字次數（unigram）　格式：字 次數")
    for tid, cnt in sorted(ng["unigram"].items(), key=lambda kv: -kv[1]):
        w(f"{itos[tid]}\t{cnt}")
    w("")
    w("3.2 二字連接（bigram，每個前文取前 5 個後接字）　格式：前字 後字 次數")
    for (prev,), nxts in sorted(ng["counts2"].items()):
        top = sorted(nxts.items(), key=lambda kv: -kv[1])[:5]
        for nxt, cnt in top:
            w(f"{itos[prev]}\t{itos[nxt]}\t{cnt}")
    w("")
    # trigram 只列全局最常見的前 50 個三元組
    tri_all = []
    for (p2, p1), nxts in ng["counts"].items():
        for nxt, cnt in nxts.items():
            tri_all.append((cnt, p2, p1, nxt))
    tri_all.sort(reverse=True)
    w("3.3 三字連接（trigram，僅列全體最常見的前 50 個）　格式：前兩字 後字 次數")
    for cnt, p2, p1, nxt in tri_all[:50]:
        w(f"{itos[p2]}{itos[p1]}\t{itos[nxt]}\t{cnt}")
    w("")

    # 四、MLP 權重（摘錄：完整權重在 outputs/mlp_model.pt）
    w("四、MLP 權重（摘錄）")
    w("-" * 40)
    w("網路：前 ctx 字 id → Embedding 查表 → 攤平 (ctx × emb) → Linear+tanh → Linear → softmax。")
    w(f"完整權重共 {sum(t.numel() for t in sd.values())} 個參數，存於 outputs/mlp_model.pt；本檔只列矩陣形狀與前幾列樣本，數值保留四位小數。")
    w("")

    def sample_matrix(name, tensor, row_label, rows=5):
        t = tensor.detach().cpu()
        w(f"{row_label}　shape {list(t.shape)}")
        if t.dim() == 1:
            vals = t.tolist()[:rows]
            w(" ".join(fmt(float(x)) for x in vals) + (" …" if len(t) > rows else ""))
        else:
            for r in t.tolist()[:rows]:
                w(" ".join(fmt(float(x)) for x in r[:8]) + (" …" if len(r) > 8 else ""))
            if t.shape[0] > rows:
                w(f"…（其餘 {t.shape[0] - rows} 列略）")
        w("")

    sample_matrix("emb.weight", sd["emb.weight"], "4.1 emb.weight（每字的向量）")
    sample_matrix("fc1.weight", sd["fc1.weight"], "4.2 fc1.weight（隱藏層權重）")
    sample_matrix("fc1.bias", sd["fc1.bias"], "4.3 fc1.bias（隱藏層偏置）")
    sample_matrix("fc2.weight", sd["fc2.weight"], "4.4 fc2.weight（輸出層權重）")
    sample_matrix("fc2.bias", sd["fc2.bias"], "4.5 fc2.bias（輸出層偏置）")

    # 五、推論說明
    w("五、推論方式（對照本專案程式）")
    w("-" * 40)
    w("1. 把輸入句編成 id 序列，不足 ctx 格時前面補 <S>。")
    w("2. MLP：emb 查表 → 攤平 → fc1 → tanh → fc2 → softmax 得到下一字機率。")
    w("3. n-gram：trigram / bigram / unigram 三層機率按 0.7 / 0.2 / 0.1 插值。")
    w("4. 兩引擎可再融合取樣；解碼可調 temperature 與 top-k。")
    w("5. 瀏覽器版匯出時把權重做 int8 對稱量化（見 export_model.py），本檔保留原始浮點數。")

    out_path = os.path.join(HERE, "模型.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    size = os.path.getsize(out_path)
    print(f"已匯出 {out_path}　{size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
