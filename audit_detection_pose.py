"""Comprehensive audit of player detection and pose estimation quality."""

import json
import time
import sys
from urllib.request import urlopen, Request

BASE_URL = "http://127.0.0.1:8765"


def api_get(endpoint: str) -> dict:
    """Make GET request to API."""
    try:
        req = Request(f"{BASE_URL}{endpoint}")
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return {}


def test_video_metadata():
    """Test 1: Check video metadata."""
    print("\n" + "="*70)
    print("TEST 1: Video Metadata & Indexing")
    print("="*70)
    
    videos = api_get("/api/videos")
    if not videos.get("videos"):
        print("[FAIL] No videos found")
        return None
    
    vid_info = videos["videos"][0]
    print(f"Video ID: {vid_info['vid']}")
    print(f"Resolution: {vid_info['width']}x{vid_info['height']}")
    print(f"FPS: {vid_info['fps']:.2f}")
    print(f"Total frames: {vid_info['nframes']}")
    print(f"Has poses: {vid_info.get('has_poses', False)}")
    
    return vid_info


def test_detection_quality(video_id: str, num_frames: int = 10):
    """Test 2: Analyze detection quality."""
    print("\n" + "="*70)
    print("TEST 2: Detection Quality Analysis")
    print("="*70)
    
    data = api_get(f"/api/videos/{video_id}/frames?from=1&to={num_frames}&limit={num_frames}")
    if not data or not data.get("frames"):
        print("[FAIL] No frame data")
        return
    
    frames = data["frames"]
    total_detections = 0
    confidences = []
    bbox_areas = []
    
    for frame_data in frames:
        objects = frame_data.get("objects", [])
        total_detections += len(objects)
        
        for obj in objects:
            confidences.append(obj.get("conf", 0.0))
            bbox = obj.get("bbox_px")
            if bbox:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                bbox_areas.append(area)
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    avg_area = sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0
    
    print(f"Frames analyzed: {len(frames)}")
    print(f"Total detections: {total_detections}")
    print(f"Detections per frame: {total_detections/len(frames):.1f}")
    print(f"Average confidence: {avg_conf:.3f}")
    print(f"Confidence range: [{min(confidences):.3f}, {max(confidences):.3f}]")
    print(f"Average bbox area: {avg_area:.0f} px^2")
    
    low_conf = [c for c in confidences if c < 0.5]
    if low_conf:
        print(f"WARNING: {len(low_conf)} detections below 0.5 confidence")


def test_pose_accuracy(video_id: str, num_frames: int = 10):
    """Test 3: Analyze pose estimation accuracy."""
    print("\n" + "="*70)
    print("TEST 3: Pose Estimation Accuracy")
    print("="*70)
    
    data = api_get(f"/api/videos/{video_id}/frames?from=1&to={num_frames}&limit={num_frames}")
    if not data or not data.get("poses"):
        print("[FAIL] No pose data")
        return
    
    poses_by_frame = data["poses"]
    total_poses = 0
    keypoint_stats = {i: {"valid": 0, "conf_sum": 0.0, "count": 0} for i in range(17)}
    
    COCO_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    for frame_str, poses in poses_by_frame.items():
        for pose in poses:
            kpts = pose.get("kpts", [])
            if not kpts or len(kpts) != 17:
                print(f"WARNING: Invalid keypoint count: {len(kpts)}")
                continue
            
            total_poses += 1
            
            for i, kpt in enumerate(kpts):
                u, v, conf = kpt
                is_valid = conf > 0.3 and 0 <= u <= 1920 and 0 <= v <= 1080
                
                keypoint_stats[i]["count"] += 1
                if is_valid:
                    keypoint_stats[i]["valid"] += 1
                keypoint_stats[i]["conf_sum"] += conf
    
    print(f"Total poses analyzed: {total_poses}")
    print(f"\nPer-keypoint statistics:")
    print(f"{'Keypoint':<20} {'Valid %':<10} {'Avg Conf':<10}")
    print("-" * 40)
    
    for i in range(17):
        stats = keypoint_stats[i]
        valid_pct = (stats["valid"] / stats["count"] * 100) if stats["count"] > 0 else 0
        avg_conf = (stats["conf_sum"] / stats["count"]) if stats["count"] > 0 else 0
        
        status = "OK" if valid_pct > 80 else "WARN" if valid_pct > 50 else "BAD"
        print(f"{status} {COCO_NAMES[i]:<18} {valid_pct:>6.1f}%   {avg_conf:.3f}")


def test_performance(video_id: str):
    """Test 4: Performance metrics."""
    print("\n" + "="*70)
    print("TEST 4: Performance Metrics")
    print("="*70)
    
    times = []
    for n in [50, 100, 150]:
        start = time.time()
        api_get(f"/api/videos/{video_id}/frame.jpg?n={n}&layers=boxes,pose&realtime=1&width=1280&q=72")
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"Frame {n}: {elapsed:.0f}ms")
    
    avg_time = sum(times) / len(times)
    fps_estimate = 1000 / avg_time if avg_time > 0 else 0
    
    print(f"\nAverage inference time: {avg_time:.0f}ms/frame")
    print(f"Estimated FPS: {fps_estimate:.1f}")
    
    if fps_estimate < 30:
        print(f"WARNING: Below 30 FPS target ({fps_estimate:.1f} FPS)")
    else:
        print(f"Meets 30 FPS target")


def main():
    print("Basketball Video Analysis System - Audit")
    print("="*70)
    
    vid_info = test_video_metadata()
    if not vid_info:
        print("[ABORT] Cannot proceed without video data")
        sys.exit(1)
    
    video_id = vid_info["vid"]
    test_detection_quality(video_id, num_frames=10)
    test_pose_accuracy(video_id, num_frames=10)
    test_performance(video_id)
    
    print("\n" + "="*70)
    print("AUDIT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
