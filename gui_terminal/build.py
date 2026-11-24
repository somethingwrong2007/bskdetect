"""把 src/App.jsx 预编译成 src/App.js，让 gui_terminal 零构建、零 Node 也能跑。

原理：Babel standalone（MIT）只在构建期用一次，通过 headless Chrome 执行编译，
产物写回 src/App.js。运行时页面只加载 React + ReactDOM + Tailwind Play，
不需要 Node，也不需要浏览器端的 Babel。

用法：
    python gui_terminal/build.py            # 编译（缺 babel.min.js 时自动下载一次）
    python gui_terminal/build.py --check    # 只校验现有 App.js 是否与 App.jsx 一致

babel.min.js 与临时 harness 都在 .gitignore 里，不进仓库。
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src" / "App.jsx"
OUT = HERE / "src" / "App.js"
BABEL = HERE / "vendor" / "babel.min.js"
BABEL_URL = "https://unpkg.com/@babel/standalone@7.26.7/babel.min.js"
BABEL_VERSION = "@babel/standalone 7.26.7 (MIT)"

HEADER = (
    "/* 本文件由 gui_terminal/build.py 从同目录 App.jsx 预编译生成，请勿手改。\n"
    f"   编译器：{BABEL_VERSION}，preset=react\n"
    "   改界面只改 App.jsx，然后跑 python gui_terminal/build.py 重新生成。 */\n\n"
)

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
]


def find_browser() -> str:
    for name in ("chrome", "chromium", "chromium-browser", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    for cand in CHROME_CANDIDATES:
        if pathlib.Path(cand).exists():
            return cand
    raise SystemExit("找不到 Chrome/Chromium/Edge，无法预编译 JSX（或改用 Node：npx @babel/cli）")


def ensure_babel() -> None:
    if BABEL.exists():
        return
    print(f"[build] 下载 {BABEL_URL}（约 3MB，仅构建期使用，不入仓）")
    with urllib.request.urlopen(BABEL_URL, timeout=120) as resp:  # noqa: S310
        BABEL.write_bytes(resp.read())


def compile_jsx(source: str) -> str:
    harness = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<script src='{BABEL.as_uri()}'></script>"
        "</head><body><pre id='out'></pre><script>\n"
        f"var JSX_SOURCE = {json.dumps(source, ensure_ascii=False)};\n"
        "var COMPILED = Babel.transform(JSX_SOURCE, {presets: ['react'], filename: 'App.jsx'}).code;\n"
        "document.getElementById('out').textContent = COMPILED;\n"
        "</script></body></html>"
    )
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gui_terminal_build_")) / "harness.html"
    tmp.write_text(harness, encoding="utf-8")
    try:
        proc = subprocess.run(
            [find_browser(), "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=10000", "--dump-dom", tmp.as_uri()],
            capture_output=True, text=True, encoding="utf-8", timeout=180,
        )
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    m = re.search(r"<pre id=['\"]out['\"]>(.*?)</pre>", proc.stdout, re.S)
    if not m:
        raise SystemExit(
            "[build] 编译失败，浏览器没有输出结果。\nstderr:\n" + proc.stderr[-2000:])
    return html.unescape(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="只比对现有 App.js 是否与 App.jsx 的编译结果一致")
    args = ap.parse_args()

    source = SRC.read_text(encoding="utf-8")
    ensure_babel()
    compiled = compile_jsx(source)
    fresh = HEADER + compiled.rstrip("\r\n") + "\n"

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        same = current == fresh
        print("[check] App.js 与 App.jsx 一致" if same
              else "[check] 不一致：App.jsx 改过但没重新编译，请跑 python gui_terminal/build.py")
        return 0 if same else 1

    OUT.write_text(fresh, encoding="utf-8", newline="\n")
    print(f"[build] {SRC.name} {len(source)} 字符 -> {OUT.name} {len(fresh)} 字符")
    return 0


if __name__ == "__main__":
    sys.exit(main())
