"""Run comprehensive batch analysis on all frames of a video.

This ensures every frame has detection/pose/segmentation data cached,
enabling accurate visual audit and eliminating "missing data" gaps.

Usage:
    python run_batch_analysis.py --video <vid> --stride 2 --det-thr 0.5
"""

import argparse
import json
import sys
import time

import requests


def get_video_list(base_url: str) -> list[dict]:
    """Get list of all videos."""
    resp = requests.get(f"{base_url}/api/videos", timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to list videos: {resp.text}")
    
    data = resp.json()
    return data.get("videos", [])


def wait_for_indexing(vid: str, base_url: str, timeout: int = 120) -> bool:
    """Wait for video indexing to complete."""
    print(f"[WAIT] Waiting for {vid} indexing...")
    start = time.time()
    
    while time.time() - start < timeout:
        resp = requests.get(f"{base_url}/api/videos/{vid}", timeout=10)
        if resp.status_code != 200:
            continue
        
        data = resp.json()
        if data.get("status") == "ready":
            nframes = data.get("nframes", 0)
            print(f"[OK] Indexing complete: {nframes} frames")
            return True
        
        job = data.get("job", {})
        stage = job.get("stage", "unknown")
        print(f"  Stage: {stage}, Status: {data.get('status')}")
        time.sleep(3)
    
    print("[FAIL] Indexing timeout")
    return False


def start_analysis(vid: str, base_url: str, frm: int = 1, to: int = None, 
                   stride: int = 2, det_thr: float = 0.5, with_masks: bool = False) -> str:
    """Start batch analysis job."""
    payload = {
        "frm": frm,
        "to": to,
        "stride": stride,
        "det_thr": det_thr,
        "with_masks": with_masks
    }
    
    print(f"[START] Analysis job for {vid}: frames {frm}-{to or 'end'}, stride={stride}")
    resp = requests.post(
        f"{base_url}/api/videos/{vid}/analyze",
        json=payload,
        timeout=30
    )
    
    if resp.status_code != 200:
        raise RuntimeError(f"Analysis start failed: {resp.text}")
    
    data = resp.json()
    job_id = data["job_id"]
    print(f"[OK] Job started: {job_id}")
    return job_id


def monitor_job(job_id: str, base_url: str, poll_interval: int = 5) -> dict:
    """Monitor analysis job progress."""
    print(f"[MONITOR] Watching job {job_id}...")
    
    while True:
        resp = requests.get(f"{base_url}/api/jobs/{job_id}", timeout=10)
        if resp.status_code != 200:
            print(f"[WARN] Job query failed: {resp.text}")
            time.sleep(poll_interval)
            continue
        
        data = resp.json()
        status = data.get("status", "unknown")
        progress = data.get("progress", 0) * 100
        
        if status in ["done", "error"]:
            print(f"[{status.upper()}] Job completed")
            if status == "done":
                result = data.get("result", {})
                print(f"  Frames analyzed: {result.get('frames', 0)}")
                print(f"  Poses detected: {result.get('poses', 0)}")
                print(f"  Segments detected: {result.get('segments', 0)}")
                print(f"  Time per frame: {result.get('ms_per_frame', 0):.1f}ms")
            else:
                print(f"  Error: {data.get('result', {}).get('error', 'Unknown')}")
            return data
        
        print(f"  Progress: {progress:.1f}% ({status})")
        time.sleep(poll_interval)


def verify_coverage(vid: str, base_url: str, sample_frames: list[int]) -> dict:
    """Verify that analysis data exists for sample frames."""
    print(f"\n[VERIFY] Checking data coverage for {len(sample_frames)} sample frames...")
    
    covered = 0
    missing = []
    
    for frm in sample_frames:
        resp = requests.get(
            f"{base_url}/api/videos/{vid}/frames",
            params={"from": frm, "to": frm, "limit": 1},
            timeout=10
        )
        
        if resp.status_code != 200:
            missing.append(frm)
            continue
        
        data = resp.json()
        frames = data.get("frames", [])
        
        if frames:
            covered += 1
            obj_count = len(frames[0].get("objects", []))
            print(f"  Frame {frm}: {obj_count} detections ✓")
        else:
            missing.append(frm)
            print(f"  Frame {frm}: NO DATA ✗")
    
    coverage = covered / max(len(sample_frames), 1) * 100
    print(f"\nCoverage: {covered}/{len(sample_frames)} ({coverage:.1f}%)")
    
    if missing:
        print(f"Missing frames: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    
    return {"covered": covered, "total": len(sample_frames), "missing": missing}


def main():
    parser = argparse.ArgumentParser(description="Batch analysis for bskdetect")
    parser.add_argument("--video", help="Video ID (leave empty to use first video)")
    parser.add_argument("--stride", type=int, default=2, help="Analyze every Nth frame (default: 2)")
    parser.add_argument("--det-thr", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="Server URL")
    parser.add_argument("--masks", action="store_true", help="Enable segmentation masks (slow)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("BSKDETECT BATCH ANALYSIS")
    print("="*70)
    
    # Get video list
    videos = get_video_list(args.url)
    if not videos:
        print("[FAIL] No videos found. Import a video first.")
        sys.exit(1)
    
    # Select video
    if args.video:
        vid = args.video
    else:
        vid = videos[0]["vid"]
        print(f"[INFO] Using first video: {vid} ({videos[0].get('name', 'unnamed')})")
    
    # Wait for indexing
    if not wait_for_indexing(vid, args.url):
        print("[FAIL] Video not ready")
        sys.exit(1)
    
    # Get video metadata
    resp = requests.get(f"{args.url}/api/videos/{vid}", timeout=10)
    meta = resp.json()
    nframes = meta.get("nframes", 0)
    
    if nframes == 0:
        print("[FAIL] Video has 0 frames")
        sys.exit(1)
    
    print(f"\nVideo: {meta.get('name')}")
    print(f"Resolution: {meta.get('width')}x{meta.get('height')}")
    print(f"Total frames: {nframes}")
    print(f"Stride: {args.stride} (analyzing every {args.stride}th frame)")
    print(f"Frames to analyze: ~{nframes // args.stride}")
    
    # Start analysis
    try:
        job_id = start_analysis(
            vid, args.url,
            frm=1,
            to=nframes,
            stride=args.stride,
            det_thr=args.det_thr,
            with_masks=args.masks
        )
        
        # Monitor progress
        result = monitor_job(job_id, args.url)
        
        if result.get("status") != "done":
            print("\n[FAIL] Analysis job failed")
            sys.exit(1)
        
        # Verify coverage
        sample_frames = list(range(1, min(nframes + 1, 51), max(1, nframes // 50)))
        coverage = verify_coverage(vid, args.url, sample_frames)
        
        print("\n" + "="*70)
        if coverage["covered"] >= len(sample_frames) * 0.9:
            print("SUCCESS! Batch analysis complete with good coverage")
            sys.exit(0)
        else:
            print("WARNING: Coverage incomplete, consider re-running with different parameters")
            sys.exit(2)
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Analysis stopped by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
