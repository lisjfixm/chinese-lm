# -*- coding: utf-8 -*-
"""
從零實作的 n-gram 中文語言模型（學習示範）。

核心觀念：語言模型要估計「給定前文，下一個字出現的條件機率」
    P(下一個字 | 前文)
n-gram 的簡化假設：下一個字只跟前面 n-1 個字有關。

本檔只依賴 numpy，不使用現成的模型套件，方便看清每一步。
改進重點：
  * add-k 平滑：避免未出現過的組合機率變成 0
  * 階層回溯（back-off）：三gram 沒見過就退回二gram，再退回字的邊際機率
  * 循環偵測：生成進入重複迴圈時提前停止
"""

import argparse
import json
import math
import os
import pickle
from collections import defaultdict

import numpy as np

from common import EOS, SOS, build_vocab, decode, encode, load_sentences


class NgramLM:
    def __init__(self, n: int = 2, k: float = 0.1):
        self.n = n                      # 2 = bigram, 3 = trigram
        self.k = k                      # add-k 平滑常數
        self.counts: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.context_total: dict[tuple, int] = defaultdict(int)
        self.counts2: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.total2: dict[tuple, int] = defaultdict(int)
        self.unigram: dict[int, int] = defaultdict(int)
        self.total_uni = 0
        self.vocab: dict[str, int] = {}
        self.itos: list[str] = []
        self.V = 0

    def fit(self, sentences: list[str], vocab: dict[str, int], itos: list[str]) -> "NgramLM":
        """統計 unigram / bigram / trigram 次數，供階層回溯使用。"""
        self.vocab, self.itos, self.V = vocab, itos, len(vocab)
        sos_id, eos_id = vocab[SOS], vocab[EOS]

        for s in sentences:
            ids = encode(s, vocab)
            seq = [sos_id] * (self.n - 1) + ids + [eos_id]

            for tok in seq:  # 字的邊際出現次數
                self.unigram[tok] += 1
                self.total_uni += 1

            for i in range(1, len(seq)):  # 二gram：前一個字
                ctx = (seq[i - 1],)
                self.counts2[ctx][seq[i]] += 1
                self.total2[ctx] += 1

            for i in range(2, len(seq)):  # 三gram：前兩個字
                ctx = (seq[i - 2], seq[i - 1])
                self.counts[ctx][seq[i]] += 1
                self.context_total[ctx] += 1
        return self

    def _level_probs(self, order: int, ctx: tuple | None) -> np.ndarray:
        """單一階層的平滑機率分佈；order=1 表示不看前文的邊際分佈。"""
        V, k = self.V, self.k
        if order == 1:
            return np.array([(self.unigram.get(j, 0) + k) / (self.total_uni + k * V)
                             for j in range(V)])
        table, total = (self.counts2, self.total2) if order == 2 else (self.counts, self.context_total)
        counts = table.get(ctx)
        t = total.get(ctx, 0)
        if counts is None:
            return np.ones(V) / V
        return np.array([(counts.get(j, 0) + k) / (t + k * V) for j in range(V)])

    def _probs(self, ctx: tuple) -> np.ndarray:
        """階層回溯 + 線性插值：高階沒資訊時，把權重讓給低階模型。"""
        p1 = self._level_probs(1, None)
        if self.n == 2:
            p2 = self._level_probs(2, ctx[-1:])
            return 0.8 * p2 + 0.2 * p1
        p3 = self._level_probs(3, ctx)
        p2 = self._level_probs(2, ctx[-1:])
        return 0.7 * p3 + 0.2 * p2 + 0.1 * p1

    def _token_prob(self, order: int, ctx: tuple | None, j: int) -> float:
        """單一階層下某個目標字的平滑機率（稀疏查表，不建整個詞表陣列）。"""
        V, k = self.V, self.k
        if order == 1:
            return (self.unigram.get(j, 0) + k) / (self.total_uni + k * V)
        table, total = (self.counts2, self.total2) if order == 2 else (self.counts, self.context_total)
        counts = table.get(ctx)
        t = total.get(ctx, 0)
        if counts is None:
            return 1.0 / V
        return (counts.get(j, 0) + k) / (t + k * V)

    def _token_probs(self, ctx: tuple, j: int) -> float:
        """階層回溯 + 線性插值後，目標字 j 的機率。"""
        p1 = self._token_prob(1, None, j)
        if self.n == 2:
            p2 = self._token_prob(2, ctx[-1:], j)
            return 0.8 * p2 + 0.2 * p1
        p3 = self._token_prob(3, ctx, j)
        p2 = self._token_prob(2, ctx[-1:], j)
        return 0.7 * p3 + 0.2 * p2 + 0.1 * p1

    def sentence_logprob(self, ids: list[int]) -> tuple[float, int]:
        sos_id, eos_id = self.vocab[SOS], self.vocab[EOS]
        seq = [sos_id] * (self.n - 1) + list(ids) + [eos_id]
        logp, n_tok = 0.0, 0
        for i in range(self.n - 1, len(seq)):
            ctx = tuple(seq[i - self.n + 1: i])
            p = self._token_probs(ctx, seq[i])
            logp += math.log(p)
            n_tok += 1
        return logp, n_tok

    def perplexity(self, sentences: list[str]) -> float:
        """困惑度（Perplexity）：越小代表模型對語料預測得越準。"""
        total_lp, total_tok = 0.0, 0
        for s in sentences:
            lp, n_tok = self.sentence_logprob(encode(s, self.vocab))
            total_lp += lp
            total_tok += n_tok
        return math.exp(-total_lp / total_tok)

    def generate(self, prompt: str, max_len: int = 40, temperature: float = 1.0,
                 top_k: int | None = None, seed: int | None = None) -> str:
        """從提示詞逐字生成；temperature 越低越保守，top_k 只保留最可能的幾個字。"""
        rng = np.random.default_rng(seed)
        sos_id, eos_id = self.vocab[SOS], self.vocab[EOS]
        history = [sos_id] * (self.n - 1) + encode(prompt, self.vocab)
        generated = encode(prompt, self.vocab)

        for _ in range(max_len):
            ctx = tuple(history[-(self.n - 1):])
            p = self._probs(ctx)
            logp = np.log(p) / temperature
            logp -= logp.max()
            w = np.exp(logp)
            if top_k is not None and top_k < len(w):
                top_idx = np.argsort(w)[::-1][:top_k]
                mask = np.zeros_like(w, dtype=bool)
                mask[top_idx] = True
                w[~mask] = 0.0
            w /= w.sum()
            nxt = int(rng.choice(len(w), p=w))

            if nxt == eos_id:
                break
            generated.append(nxt)
            history.append(nxt)

            # 循環偵測：最後 3 個字連續重複兩次就停止，避免無意義繞圈
            if len(generated) >= 6 and generated[-3:] == generated[-6:-3]:
                break
        return decode(generated, self.itos)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"n": self.n, "k": self.k, "counts": dict(self.counts),
                         "context_total": dict(self.context_total),
                         "counts2": {k: dict(v) for k, v in self.counts2.items()},
                         "total2": dict(self.total2),
                         "unigram": dict(self.unigram), "total_uni": self.total_uni,
                         "vocab": self.vocab, "itos": self.itos}, f)

    @classmethod
    def load(cls, path: str) -> "NgramLM":
        with open(path, "rb") as f:
            data = pickle.load(f)
        lm = cls(n=data["n"], k=data["k"])
        lm.counts = defaultdict(lambda: defaultdict(int),
                                {k: defaultdict(int, v) for k, v in data["counts"].items()})
        lm.context_total = defaultdict(int, data["context_total"])
        lm.counts2 = defaultdict(lambda: defaultdict(int),
                                 {k: defaultdict(int, v) for k, v in data["counts2"].items()})
        lm.total2 = defaultdict(int, data["total2"])
        lm.unigram = defaultdict(int, data["unigram"])
        lm.total_uni = data["total_uni"]
        lm.vocab, lm.itos = data["vocab"], data["itos"]
        lm.V = len(lm.vocab)
        return lm


def _write_metrics(out_dir: str, patch: dict) -> None:
    path = os.path.join(out_dir, "metrics.json")
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data.update(patch)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="從零實作的 n-gram 中文語言模型")
    ap.add_argument("--corpus", default=None, help="自訂語料檔（每行一句）")
    ap.add_argument("--n", type=int, default=2, choices=[2, 3])
    ap.add_argument("--k", type=float, default=0.1, help="add-k 平滑常數")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sentences = load_sentences(args.corpus)
    vocab, itos = build_vocab(sentences, min_freq=2)
    print(f"語料句子數：{len(sentences)}　詞表大小：{len(vocab)}")

    metrics = {}
    eval_sample = sentences[:5000]  # 困惑度在抽樣上計算，避免大詞表下逐字建整個機率陣列
    for n in (2, 3):
        probe = NgramLM(n=n, k=args.k).fit(sentences, vocab, itos)
        ppl = probe.perplexity(eval_sample)
        metrics[f"ngram_{n}gram_ppl"] = round(ppl, 2)
        print(f"{n}-gram 訓練困惑度（5000 句抽樣）：{ppl:.2f}")

    lm = NgramLM(n=args.n, k=args.k).fit(sentences, vocab, itos)

    prompts = ["今天", "我", "天氣", "朋友", "這座", "夜晚"]
    print("\n生成範例：")
    for p in prompts:
        out = lm.generate(p, temperature=args.temperature, top_k=args.top_k, seed=args.seed)
        print(f"  「{p}」→ {out}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    lm.save(os.path.join(out_dir, "ngram_model.pkl"))
    _write_metrics(out_dir, metrics)
    print(f"\n模型已儲存至 outputs/ngram_model.pkl")


if __name__ == "__main__":
    main()
