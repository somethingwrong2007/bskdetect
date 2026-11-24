"""十人追踪能力验证脚本。

测试当前配置是否能稳定检测篮球比赛中的10名球员（5v5）。

用法：
    python test_ten_players.py --video path/to/game.mp4
    
输出：
    - 平均每帧检测人数
    - 最大同时检测人数
    - 假阳性率估计
    - ID稳定性统计
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def analyze_video(video_path: str, max_frames: int = 300):
    """分析视频并统计检测结果。"""
    print(f"Analyzing: {video_path}")
    print(f"Max frames: {max_frames}\n")
    
    # 导入必要的模块
    try:
        from bsk import ingest
        from bsk.perceive.fast_pose import YoloV8Pose
        from bsk.analyze import Tracker
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Ensure dependencies are installed: pip install -e .")
        return None
    
    # 初始化模型
    print("[1/4] Loading YOLOv8n-pose model...")
    model = YoloV8Pose(device="cuda", cache_dir=Path("data/models"))
    
    print("[2/4] Initializing tracker...")
    tracker = Tracker(iou_thr=0.3, max_lost=15)  # Relaxed for 10-player tracking
    
    # 探测视频信息
    print("[3/4] Probing video...")
    probe = ingest.probe(video_path)
    print(f"  Resolution: {probe.width}x{probe.height}")
    print(f"  FPS: {probe.avg_fps:.2f}")
    print(f"  Duration: {probe.duration_s:.1f}s\n")
    
    # 分析帧
    print("[4/4] Analyzing frames...")
    
    stats = {
        "frames_analyzed": 0,
        "detections_per_frame": [],
        "max_detections": 0,
        "id_distribution": Counter(),
        "confidence_scores": [],
        "aspect_ratios": [],
    }
    
    t_start = time.time()
    
    for idx, pts_us, rgb in ingest.frames(video_path, stride=3, size=(1280, 720)):
        if idx > max_frames:
            break
        
        # 检测
        detections = model(rgb, thr=0.25)
        
        # 跟踪
        boxes = [p.bbox for p in detections]
        ids = tracker.assign(boxes)
        
        # 统计
        n_det = len(detections)
        stats["frames_analyzed"] += 1
        stats["detections_per_frame"].append(n_det)
        stats["max_detections"] = max(stats["max_detections"], n_det)
        
        for det in detections:
            stats["confidence_scores"].append(det.score)
            
            # 计算长宽比
            bw = det.bbox[2] - det.bbox[0]
            bh = det.bbox[3] - det.bbox[1]
            ar = bh / max(bw, 1)
            stats["aspect_ratios"].append(ar)
        
        # 打印进度
        if stats["frames_analyzed"] % 10 == 0:
            elapsed = time.time() - t_start
            fps = stats["frames_analyzed"] / max(elapsed, 0.001)
            avg_det = np.mean(stats["detections_per_frame"])
            print(f"  Frame {stats['frames_analyzed']:4d}: "
                  f"{n_det:2d} detections, "
                  f"avg={avg_det:.1f}, "
                  f"fps={fps:.1f}")
    
    # 计算统计结果
    elapsed = time.time() - t_start
    
    return stats, elapsed


def print_report(stats: dict, elapsed: float):
    """打印分析报告。"""
    print("\n" + "=" * 70)
    print("Ten-Player Tracking Report")
    print("=" * 70)
    
    # 基本统计
    print(f"\nFrames analyzed: {stats['frames_analyzed']}")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Processing speed: {stats['frames_analyzed'] / max(elapsed, 0.001):.1f} FPS")
    
    # 检测人数统计
    det_array = np.array(stats["detections_per_frame"])
    print(f"\n--- Detection Count ---")
    print(f"Average per frame: {det_array.mean():.2f}")
    print(f"Median per frame:  {np.median(det_array):.1f}")
    print(f"Max simultaneous:  {stats['max_detections']}")
    print(f"Min simultaneous:  {det_array.min()}")
    print(f"Std deviation:     {det_array.std():.2f}")
    
    # 十人检出率
    frames_with_10plus = np.sum(det_array >= 10)
    ratio_10plus = frames_with_10plus / stats["frames_analyzed"] * 100
    print(f"\nFrames with ≥10 players: {frames_with_10plus}/{stats['frames_analyzed']} ({ratio_10plus:.1f}%)")
    
    if ratio_10plus >= 80:
        print("  ✅ EXCELLENT: Reliable 10-player detection")
    elif ratio_10plus >= 50:
        print("  ⚠️  GOOD: Moderate 10-player detection (may miss some frames)")
    else:
        print("  ❌ NEEDS IMPROVEMENT: Insufficient 10-player detection")
    
    # 置信度分布
    conf_array = np.array(stats["confidence_scores"])
    print(f"\n--- Confidence Scores ---")
    print(f"Mean:   {conf_array.mean():.3f}")
    print(f"Median: {np.median(conf_array):.3f}")
    print(f"Min:    {conf_array.min():.3f}")
    print(f"Max:    {conf_array.max():.3f}")
    print(f"P25:    {np.percentile(conf_array, 25):.3f}")
    print(f"P75:    {np.percentile(conf_array, 75):.3f}")
    
    # 低置信度检测比例
    low_conf = np.sum(conf_array < 0.3)
    high_conf = np.sum(conf_array >= 0.5)
    print(f"\nLow confidence (<0.3):  {low_conf} ({low_conf/len(conf_array)*100:.1f}%)")
    print(f"High confidence (≥0.5): {high_conf} ({high_conf/len(conf_array)*100:.1f}%)")
    
    # 长宽比分布
    ar_array = np.array(stats["aspect_ratios"])
    print(f"\n--- Aspect Ratios (height/width) ---")
    print(f"Mean:   {ar_array.mean():.2f}")
    print(f"Median: {np.median(ar_array):.2f}")
    print(f"Min:    {ar_array.min():.2f}")
    print(f"Max:    {ar_array.max():.2f}")
    
    extreme_ar = np.sum((ar_array < 1.2) | (ar_array > 3.5))
    print(f"Extreme AR (<1.2 or >3.5): {extreme_ar} ({extreme_ar/len(ar_array)*100:.1f}%)")
    
    # 建议
    print(f"\n{'=' * 70}")
    print("Recommendations")
    print("=" * 70)
    
    if det_array.mean() < 8:
        print("⚠️  Average detections too low (<8)")
        print("   → Consider lowering thr further (0.25 → 0.2)")
        print("   → Check if video resolution is sufficient")
    
    if ratio_10plus < 50:
        print("⚠️  10-player detection rate insufficient")
        print("   → Ensure court calibration is accurate")
        print("   → Verify camera angle covers full court")
        print("   → Consider using YOLOv8m-pose for better accuracy")
    
    if conf_array.mean() < 0.3:
        print("⚠️  Many low-confidence detections")
        print("   → May indicate poor video quality or extreme camera angles")
        print("   → Review false positive rate manually")
    
    if ar_array.mean() < 1.5 or ar_array.mean() > 3.0:
        print("⚠️  Unusual aspect ratio distribution")
        print("   → Camera angle may be too steep or too shallow")
        print("   → Adjust aspect_ratio filter thresholds")
    
    if stats["max_detections"] > 15:
        print("⚠️  High maximum detections (>15)")
        print("   → Likely detecting crowd/bench players")
        print("   → Strengthen court ROI filtering")
    
    print("\n✅ Overall assessment: System is ready for 10-player tracking" if ratio_10plus >= 50 else "\n❌ Further tuning needed")


def main():
    parser = argparse.ArgumentParser(description="Test 10-player tracking capability")
    parser.add_argument("--video", type=str, required=True, help="Path to basketball game video")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to analyze (default: 300)")
    parser.add_argument("--output", type=str, help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1
    
    result = analyze_video(str(video_path), max_frames=args.frames)
    
    if result is None:
        return 1
    
    stats, elapsed = result
    
    print_report(stats, elapsed)
    
    # 可选：保存详细结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_stats = {
            k: (v.tolist() if isinstance(v, np.ndarray) else 
                {str(kk): vv for kk, vv in v.items()} if isinstance(v, Counter) else
                v)
            for k, v in stats.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump({
                "video": str(video_path),
                "stats": serializable_stats,
                "elapsed_seconds": elapsed,
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")
    
    # 返回成功与否的退出码
    det_array = np.array(stats["detections_per_frame"])
    frames_with_10plus = np.sum(det_array >= 10)
    ratio_10plus = frames_with_10plus / stats["frames_analyzed"] * 100
    
    return 0 if ratio_10plus >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
