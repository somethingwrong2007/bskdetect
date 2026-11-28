#!/usr/bin/env python3
"""独立运行示例：python detect_court_lines.py <图片路径>。

脚本读取 BGR 图片，输出红点（有效 Shi-Tomasi 特征点）和蓝线（鲁棒拟合边线）。
核心实现位于 bsk.perceive.court_line_features，不依赖模型权重或视频服务。
"""

from bsk.perceive.court_line_features import main


if __name__ == "__main__":
    main()
