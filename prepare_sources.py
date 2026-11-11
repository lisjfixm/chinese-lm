# -*- coding: utf-8 -*-
"""
清洗線上下載的真實語料，輸出 data-src/zh_clean.txt 與 data-src/en_clean.txt。

來源：
  data-src/sentences_CC0.csv  Tatoeba CC0 英文句（lang=eng）
  data-src/lot1.xlsx          AIxBlock 中文短句（Accepted），簡轉繁

規則：
  * 長度 4–60 字；去掉網址、郵件、外文字母（假名／諺文／西里爾等）、符號雜訊。
  * 英文轉小寫、空白收緊；中文經 OpenCC 簡轉繁，半形標點轉全形。
  * 去重；一行一句，供 build_corpus.py 取用。
"""

import csv
import os
import re

import openpyxl
from opencc import OpenCC

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data-src")
ZH_OUT = os.path.join(SRC, "zh_clean.txt")
EN_OUT = os.path.join(SRC, "en_clean.txt")

cc = OpenCC("s2t")

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
LATIN = re.compile(r"[a-z0-9]")
# 不該出現的文字系統：假名、諺文、西里爾、希臘、阿拉伯字母、emoji 與全形符號
BAD = re.compile(r"[぀-ヿ가-힣Ѐ-ӿΑ-Ωα-ω؀-ۿ😀-🿿]")
URL = re.compile(r"(https?://|www\.|@|\.com|\.cn|\.org)")

ZH_PUNCT = "，。？！、：；「」『』“”‘’（）《》—…·"
EN_ALLOW = re.compile(r"[a-z0-9 ,.!?;:'\"()-]")


def clean_zh(text: str) -> str | None:
    s = str(text).strip()
    if not s or URL.search(s):
        return None
    if BAD.search(s):
        return None
    s = cc.convert(s)
    # 半形標點轉全形；數字之間的小數點保留
    trans = {",": "，", "?": "？", "!": "！",
             ":": "：", ";": "；", "(": "（", ")": "）"}
    s = "".join(trans.get(ch, ch) for ch in s)
    s = re.sub(r"(?<!\d)\.(?!\d)", "。", s)
    s = re.sub(r"\s+", "", s)
    s = s.strip("\"'“”‘’，。、")
    if not (4 <= len(s) <= 60):
        return None
    if not CJK.search(s):
        return None
    # 只保留 CJK、容許的標點與少量拉丁字母（中英夾雜）
    for ch in s:
        if CJK.match(ch) or ch in ZH_PUNCT or LATIN.match(ch):
            continue
        return None
    return s


def clean_en(text: str) -> str | None:
    s = str(text).strip().lower()
    if not s or URL.search(s):
        return None
    if BAD.search(s):
        return None
    s = re.sub(r"\s+", " ", s)
    if not (3 <= len(s) <= 60):
        return None
    if not LATIN.search(s):
        return None
    for ch in s:
        if not EN_ALLOW.match(ch):
            return None
    return s


def main() -> None:
    # 中文：xlsx
    zh: list[str] = []
    seen_zh: set[str] = set()
    wb = openpyxl.load_workbook(os.path.join(SRC, "lot1.xlsx"), read_only=True)
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[2] != "Accepted":
            continue
        s = clean_zh(row[1])
        if s and s not in seen_zh:
            seen_zh.add(s)
            zh.append(s)

    # 英文：Tatoeba CC0 csv
    en: list[str] = []
    seen_en: set[str] = set()
    with open(os.path.join(SRC, "sentences_CC0.csv"), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3 or row[1] != "eng":
                continue
            s = clean_en(row[2])
            if s and s not in seen_en:
                seen_en.add(s)
                en.append(s)

    with open(ZH_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(zh) + "\n")
    with open(EN_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(en) + "\n")

    print(f"中文短句：{len(zh)} 句 → {ZH_OUT}")
    print(f"英文短句：{len(en)} 句 → {EN_OUT}")


if __name__ == "__main__":
    main()
