# -*- coding: utf-8 -*-
"""
共用工具：語料載入與字級（char-level）斷詞。
中文沒有空白分词，這裡直接把「每一個字」當成一個 token，
好處是不需要外部分词套件，詞表也容易理解。
"""

import os

SOS = "<S>"   # 句子開頭（start of sentence）
EOS = "</S>"  # 句子結尾（end of sentence）
UNK = "<UNK>" # 未見過的字

SPECIAL_TOKENS = [SOS, EOS, UNK]


def load_sentences(path: str | None = None) -> list[str]:
    """讀取語料檔，一行視為一個句子；未指定路徑時讀取同目錄的 corpus.txt。"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")
    sentences: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                sentences.append(s)
    return sentences


def tokenize(sentence: str) -> list[str]:
    """把句子拆成字串串列；標點符號也各自成為一個 token。"""
    return list(sentence)


def build_vocab(sentences: list[str], min_freq: int = 1) -> tuple[dict[str, int], list[str]]:
    """由語料建立詞表：字 -> 編號、編號 -> 字。特殊 token 排在最前面。

    min_freq 以下的罕見字不收進詞表，編碼時對應到 <UNK>，避免詞表被冷僻字撐大。
    """
    from collections import Counter
    freq: Counter[str] = Counter()
    for s in sentences:
        freq.update(tokenize(s))
    chars = sorted(ch for ch, n in freq.items() if n >= min_freq)
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    itos = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return vocab, itos


def encode(sentence: str, vocab: dict[str, int]) -> list[int]:
    """句子 -> token id 串列，未登入的字映射到 <UNK>。"""
    unk_id = vocab[UNK]
    return [vocab.get(ch, unk_id) for ch in tokenize(sentence)]


def decode(ids: list[int], itos: list[str]) -> str:
    """token id 串列 -> 字串（去掉特殊 token）。"""
    out = []
    for i in ids:
        tok = itos[i]
        if tok in (SOS, EOS, UNK):
            continue
        out.append(tok)
    return "".join(out)
