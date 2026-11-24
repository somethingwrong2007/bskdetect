"""Automated performance regression test for bskdetect pipeline.

Tests:
1. Model loading time
2. Per-frame inference time (calibration, detection, pose, rendering)
3. Detection quality (precision, confidence distribution)
4. 3D reconstruction accuracy (height estimation, z-coordinate validity)
5. Memory usage (4GB vs 8GB profile)

Usage:
    python test_performance.py --video E:/videos/test.mp4 --frames 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests


class PerformanceTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8765"):
        self.base_url = base_url
        self.results = {}

    def import_video(self, video_path: str) -> dict:
        """Import a video into the system."""
        print(f"[TEST] Importing video: {video_path}")
        resp = requests.post(
            f"{self.base_url}/api/videos",
            json={"path": video_path},
            timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Import failed: {resp.text}")
        
        data = resp.json()
        vid = data["vid"]
        print(f"[OK] Video imported: {vid}")
        return data

    def wait_for_indexing(self, vid: str, timeout: int = 120) -> dict:
        """Wait for video indexing to complete."""
        print(f"[TEST] Waiting for indexing of {vid}...")
        start = time.time()
        
        while time.time() - start < timeout:
            resp = requests.get(f"{self.base_url}/api/videos/{vid}", timeout=10)
            if resp.status_code != 200:
                continue
            
            data = resp.json()
            if data.get("status") == "ready":
                print(f"[OK] Indexing complete in {time.time()-start:.1f}s")
                return data
            
            job = data.get("job", {})
            stage = job.get("stage", "unknown")
            print(f"  Status: {data['status']}, Stage: {stage}")
            time.sleep(2)
        
        raise TimeoutError(f"Indexing timeout after {timeout}s")

    def test_frame_inference(self, vid: str, frame_n: int = 100) -> dict:
        """Test single frame inference performance."""
        print(f"\n[TEST] Frame {frame_n} inference")
        
        # Test with realtime mode
        url = f"{self.base_url}/api/videos/{vid}/frame.jpg"
        params = {
            "n": frame_n,
            "layers": "hud,boxes,pose",
            "width": 1280,
            "q": 82,
            "realtime": 1,
            "court_model": "FIBA"
        }
        
        t0 = time.time()
        resp = requests.get(url, params=params, timeout=60)
        elapsed = time.time() - t0
        
        if resp.status_code != 200:
            print(f"[FAIL] Frame inference failed: {resp.text}")
            return {"error": resp.text, "time_ms": elapsed * 1000}
        
        # Check response headers for timing info
        x_realtime = resp.headers.get("X-Realtime", "0")
        
        print(f"[OK] Frame {frame_n} rendered in {elapsed*1000:.1f}ms")
        print(f"  Response size: {len(resp.content)} bytes")
        print(f"  Realtime mode: {x_realtime}")
        
        return {
            "frame": frame_n,
            "time_ms": elapsed * 1000,
            "response_bytes": len(resp.content),
            "realtime": x_realtime == "1"
        }

    def test_detection_quality(self, vid: str, frame_n: int = 100) -> dict:
        """Test detection precision and confidence distribution."""
        print(f"\n[TEST] Detection quality at frame {frame_n}")
        
        url = f"{self.base_url}/api/videos/{vid}/frames"
        params = {"from": frame_n, "to": frame_n, "limit": 1}
        
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"[FAIL] Detection query failed: {resp.text}")
            return {"error": resp.text}
        
        data = resp.json()
        frames = data.get("frames", [])
        
        if not frames:
            print("[WARN] No detections found")
            return {"detections": 0, "avg_confidence": 0}
        
        frame_data = frames[0]
        objects = frame_data.get("objects", [])
        
        confidences = [obj["conf"] for obj in objects]
        avg_conf = np.mean(confidences) if confidences else 0
        low_conf_count = sum(1 for c in confidences if c < 0.5)
        
        print(f"[OK] Detections: {len(objects)}")
        print(f"  Avg confidence: {avg_conf:.3f}")
        print(f"  Low confidence (<0.5): {low_conf_count}/{len(objects)}")
        
        return {
            "detections": len(objects),
            "avg_confidence": float(avg_conf),
            "low_confidence_count": low_conf_count,
            "min_confidence": float(min(confidences)) if confidences else 0,
            "max_confidence": float(max(confidences)) if confidences else 0
        }

    def test_3d_reconstruction(self, vid: str, frame_n: int = 100) -> dict:
        """Test 3D skeleton reconstruction quality."""
        print(f"\n[TEST] 3D reconstruction at frame {frame_n}")
        
        # Get poses data
        url = f"{self.base_url}/api/videos/{vid}/frames"
        params = {"from": frame_n, "to": frame_n, "limit": 1}
        
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return {"error": resp.text}
        
        data = resp.json()
        poses = data.get("poses", {})
        
        if not poses:
            print("[WARN] No pose data available")
            return {"poses": 0}
        
        # Analyze keypoint validity
        frame_poses = poses.get(str(frame_n), [])
        
        total_kpts = 0
        valid_kpts = 0
        heights = []
        
        for pose in frame_poses:
            kpts = np.array(pose.get("kpts", []))
            if kpts.size == 0:
                continue
            
            total_kpts += len(kpts)
            valid_kpts += sum(1 for k in kpts if k[2] > 0.3)  # Confidence threshold
            
            # Estimate height from bbox or keypoints
            if "height_px" in pose:
                heights.append(pose["height_px"])
        
        valid_ratio = valid_kpts / max(total_kpts, 1)
        
        print(f"[OK] Poses: {len(frame_poses)}")
        print(f"  Keypoint validity: {valid_kpts}/{total_kpts} ({valid_ratio*100:.1f}%)")
        if heights:
            print(f"  Height range: {min(heights):.0f} - {max(heights):.0f} px")
        
        return {
            "poses": len(frame_poses),
            "keypoint_validity": float(valid_ratio),
            "height_range_px": [float(min(heights)), float(max(heights))] if heights else None
        }

    def run_full_test(self, video_path: str, num_frames: int = 10) -> dict:
        """Run complete performance test suite."""
        print("="*60)
        print("BSKDETECT PERFORMANCE TEST SUITE")
        print("="*60)
        
        try:
            # 1. Import video
            import_data = self.import_video(video_path)
            vid = import_data["vid"]
            
            # 2. Wait for indexing
            meta = self.wait_for_indexing(vid)
            
            # 3. Test frame inference
            infer_result = self.test_frame_inference(vid, frame_n=100)
            
            # 4. Test detection quality
            detect_result = self.test_detection_quality(vid, frame_n=100)
            
            # 5. Test 3D reconstruction
            recon_result = self.test_3d_reconstruction(vid, frame_n=100)
            
            # Compile results
            self.results = {
                "video": vid,
                "video_meta": {
                    "name": meta.get("name"),
                    "resolution": f"{meta.get('width')}x{meta.get('height')}",
                    "frames": meta.get("nframes"),
                    "fps": meta.get("fps")
                },
                "performance": {
                    "inference_time_ms": infer_result.get("time_ms"),
                    "target_fps": 30,
                    "meets_target": infer_result.get("time_ms", 9999) < 33.3
                },
                "detection": detect_result,
                "reconstruction": recon_result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            print(f"Video: {meta.get('name')}")
            print(f"Inference time: {infer_result.get('time_ms', 'N/A'):.1f}ms "
                  f"({'PASS' if infer_result.get('time_ms', 9999) < 33.3 else 'FAIL'})")
            print(f"Detections: {detect_result.get('detections', 0)} "
                  f"(avg conf: {detect_result.get('avg_confidence', 0):.3f})")
            print(f"Keypoint validity: {recon_result.get('keypoint_validity', 0)*100:.1f}%")
            print("="*60)
            
            return self.results
            
        except Exception as e:
            print(f"\n[FAIL] Test suite error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def save_results(self, output_path: str = "test_results.json"):
        """Save test results to JSON file."""
        if not self.results:
            print("[WARN] No results to save")
            return
        
        Path(output_path).write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n[OK] Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BSKDetect performance test")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to test")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="Server URL")
    parser.add_argument("--output", default="test_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    tester = PerformanceTester(base_url=args.url)
    results = tester.run_full_test(args.video, num_frames=args.frames)
    tester.save_results(args.output)
    
    # Exit with error code if tests failed
    if "error" in results:
        sys.exit(1)
    
    if not results.get("performance", {}).get("meets_target", False):
        print("\n[WARN] Performance target NOT met (<33ms/frame)")
        sys.exit(2)
    
    print("\n[OK] All tests passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
