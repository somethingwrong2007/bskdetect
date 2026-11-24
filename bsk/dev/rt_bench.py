"""实时接口性能基准：keep-alive 连发 N 帧，报中位时延与等效帧率。

用法（服务已在 8000 端口跑起来）：
  .venv/Scripts/python.exe -m bsk.dev.rt_bench --n 20 --start 3000 --width 960
  .venv/Scripts/python.exe -m bsk.dev.rt_bench --n 20 --layers hud,boxes,pose --quality accurate

为什么用中位数：首帧含模型预热/编译，偶发 GC 也会拉长个别帧，
用均值会被少数离群值带偏，判断"能不能 30fps"要看中位。
"""

from __future__ import annotations

import argparse
import http.client
import statistics
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vid", default="harden-55")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--layers", default="hud,boxes,pose")
    ap.add_argument("--quality", default="accurate")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    conn = http.client.HTTPConnection(args.host, args.port, timeout=60)

    def get(n: int) -> tuple[float, bytes]:
        path = (f"/api/videos/{args.vid}/rt_frame.jpg?n={n}&layers={args.layers}"
                f"&width={args.width}&quality={args.quality}")
        t0 = time.perf_counter()
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status} {body[:200]!r}")
        return time.perf_counter() - t0, body

    for i in range(args.warmup):
        get(args.start + i)
    times = []
    sizes = []
    for i in range(args.n):
        dt, body = get(args.start + args.warmup + i)
        times.append(dt)
        sizes.append(len(body))
    med = statistics.median(times)
    print(f"frames={args.n} width={args.width} layers={args.layers} quality={args.quality}")
    print(f"  median={med*1000:.1f}ms p90={sorted(times)[int(0.9*len(times))-1]*1000:.1f}ms "
          f"min={min(times)*1000:.1f}ms max={max(times)*1000:.1f}ms")
    print(f"  等效帧率 median={1.0/med:.1f}fps  平均 jpeg={sum(sizes)/len(sizes)/1024:.0f}KB")


if __name__ == "__main__":
    main()