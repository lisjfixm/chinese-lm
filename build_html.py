# -*- coding: utf-8 -*-
"""把模板、模型資料、推論程式組成單一 HTML。"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
OUT_PATH = os.path.join(HERE, "活字盤-中英語言模型.html")


def main() -> None:
    with open(os.path.join(HERE, "page.tpl.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(HERE, "page.js"), encoding="utf-8") as f:
        js = f.read()
    with open(os.path.join(OUT_DIR, "web_model.json"), encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(HERE, "vendor", "echarts.min.js"), encoding="utf-8") as f:
        echarts = f.read()

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    block = ('<script id="model-data" type="application/json">' + payload + '</script>\n'
             '<script>\n' + js + '\n</script>')
    html = tpl.replace("<!-- __ECHARTS__ -->", "<script>\n" + echarts + "\n</script>")
    html = html.replace("<!-- __SCRIPT__ -->", block)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已產生 {OUT_PATH}　{os.path.getsize(OUT_PATH) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
