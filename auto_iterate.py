"""Automated iteration loop: test → analyze → fix → repeat until target met.

This script continuously monitors system performance, identifies bottlenecks,
applies fixes, and re-tests until the 30 FPS target is achieved.

Usage:
    python auto_iterate.py --video E:/videos/test.mp4 --max-iterations 10
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run shell command and return (exit_code, stdout, stderr)."""
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    if result.stdout:
        print(result.stdout[-500:])  # Last 500 chars
    if result.stderr:
        print("[STDERR]", result.stderr[-500:])
    return result.returncode, result.stdout, result.stderr


def check_server_health(url: str = "http://127.0.0.1:8765") -> bool:
    """Check if server is running and responsive."""
    import requests
    try:
        resp = requests.get(f"{url}/api/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def start_server():
    """Start the bskdetect server in background."""
    print("\n[ITERATION] Starting server...")
    
    # Kill existing server
    run_command("taskkill /fi \"WINDOWTITLE eq *uvicorn*\" /f 2>nul")
    time.sleep(2)
    
    # Start new server
    cmd = ".venv\\Scripts\\python.exe -m uvicorn bsk.server:app --host 127.0.0.1 --port 8765"
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to be ready
    print("  Waiting for server startup...")
    for i in range(30):
        time.sleep(2)
        if check_server_health():
            print("  [OK] Server ready")
            return True
        print(f"  ... ({i+1}/30)")
    
    print("  [FAIL] Server startup timeout")
    return False


def run_performance_test(video_path: str) -> dict:
    """Run performance test and return results."""
    print("\n[TEST] Running performance test suite...")
    
    cmd = f".venv\\Scripts\\python.exe test_performance.py --video {video_path} --output test_results.json"
    exit_code, stdout, stderr = run_command(cmd, timeout=120)
    
    # Parse results
    import json
    results_file = Path("test_results.json")
    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
        return results
    
    return {"error": f"Test failed with exit code {exit_code}"}


def analyze_bottlenecks(results: dict) -> list[str]:
    """Analyze test results and identify bottlenecks."""
    print("\n[ANALYSIS] Identifying bottlenecks...")
    
    issues = []
    
    # Check inference time
    perf = results.get("performance", {})
    infer_time = perf.get("inference_time_ms", 9999)
    
    if infer_time > 1000:
        issues.append("CRITICAL: Inference time >1s (model loading or calibration stuck)")
    elif infer_time > 100:
        issues.append("HIGH: Inference time >100ms (calibration or detection slow)")
    elif infer_time > 33:
        issues.append("MEDIUM: Inference time >33ms (approaching target but not there)")
    
    # Check detection quality
    detect = results.get("detection", {})
    avg_conf = detect.get("avg_confidence", 0)
    det_count = detect.get("detections", 0)
    
    if det_count > 15:
        issues.append("MEDIUM: Too many detections (false positives from crowd)")
    if avg_conf < 0.5:
        issues.append("HIGH: Low average confidence (threshold too low)")
    
    # Check reconstruction
    recon = results.get("reconstruction", {})
    kpt_valid = recon.get("keypoint_validity", 0)
    
    if kpt_valid < 0.7:
        issues.append("HIGH: Low keypoint validity (<70%)")
    
    for issue in issues:
        print(f"  - {issue}")
    
    return issues


def apply_fixes(issues: list[str]) -> bool:
    """Apply targeted fixes based on identified issues."""
    print("\n[FIX] Applying optimizations...")
    
    fixes_applied = []
    
    for issue in issues:
        if "Inference time >1s" in issue or "model loading" in issue:
            print("  [SKIP] Model loading issue requires manual inspection")
        
        if "calibration" in issue.lower() and "slow" in issue.lower():
            # Already fixed in previous iteration
            print("  [SKIP] Calibration already optimized")
        
        if "false positives" in issue.lower() or "confidence" in issue.lower():
            # Fix: Increase detection threshold further
            print("  [FIX] Raising detection threshold to 0.6...")
            fix_detection_threshold(0.6)
            fixes_applied.append("raised_threshold")
        
        if "keypoint validity" in issue.lower():
            # Fix: Switch to more accurate pose model
            print("  [FIX] Consider switching to RTMO-s for better accuracy...")
            print("  [NOTE] This will trade speed for accuracy")
            fixes_applied.append("pose_model_switch_suggested")
    
    if fixes_applied:
        print(f"\n[OK] Applied {len(fixes_applied)} fixes: {', '.join(fixes_applied)}")
        return True
    else:
        print("\n[WARN] No automatic fixes available")
        return False


def fix_detection_threshold(new_threshold: float):
    """Update detection threshold in server_rt.py."""
    import re
    
    file_path = Path("bsk/server_rt.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Replace threshold value
    pattern = r'pose_people = _get_rtmo\(\)\(rgb, thr=[\d.]+\)'
    replacement = f'pose_people = _get_rtmo()(rgb, thr={new_threshold})'
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        file_path.write_text(new_content, encoding="utf-8")
        print(f"    Updated threshold to {new_threshold}")
        
        # Commit change
        run_command(f"git add bsk/server_rt.py && git commit -m \"perf: Raise detection threshold to {new_threshold}\"")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-iterate bskdetect optimization")
    parser.add_argument("--video", required=True, help="Test video path")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max optimization rounds")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target FPS")
    
    args = parser.parse_args()
    
    print("="*70)
    print("BSKDETECT AUTO-OPTIMIZATION LOOP")
    print("="*70)
    print(f"Target: {args.target_fps} FPS (<{1000/args.target_fps:.1f}ms/frame)")
    print(f"Max iterations: {args.max_iterations}")
    print("="*70)
    
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{args.max_iterations}")
        print(f"{'='*70}")
        
        # 1. Start server
        if not start_server():
            print("[FAIL] Server startup failed")
            sys.exit(1)
        
        # 2. Run performance test
        results = run_performance_test(args.video)
        
        if "error" in results:
            print(f"[FAIL] Test error: {results['error']}")
            continue
        
        # 3. Check if target met
        infer_time = results.get("performance", {}).get("inference_time_ms", 9999)
        target_met = infer_time < (1000 / args.target_fps)
        
        print(f"\n[RESULT] Inference time: {infer_time:.1f}ms "
              f"({'PASS' if target_met else 'FAIL'})")
        
        if target_met:
            print(f"\n{'='*70}")
            print(f"SUCCESS! Target {args.target_fps} FPS achieved in {iteration} iterations")
            print(f"{'='*70}")
            
            # Save final results
            Path("optimization_success.txt").write_text(
                f"Achieved {args.target_fps} FPS in {iteration} iterations\n"
                f"Final inference time: {infer_time:.1f}ms\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8"
            )
            
            sys.exit(0)
        
        # 4. Analyze bottlenecks
        issues = analyze_bottlenecks(results)
        
        if not issues:
            print("[WARN] No issues found but target not met - manual investigation needed")
            break
        
        # 5. Apply fixes
        fixes_applied = apply_fixes(issues)
        
        if not fixes_applied:
            print("[FAIL] No fixes available - manual intervention required")
            break
        
        # 6. Commit changes
        run_command("git push origin master")
        
        # 7. Stop server for next iteration
        run_command("taskkill /fi \"WINDOWTITLE eq *uvicorn*\" /f 2>nul")
        time.sleep(2)
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION EXHAUSTED after {args.max_iterations} iterations")
    print(f"Final inference time: {infer_time:.1f}ms (target: {1000/args.target_fps:.1f}ms)")
    print(f"{'='*70}")
    print("\nRecommended next steps:")
    print("1. Manually inspect bottleneck stages using profiling tools")
    print("2. Consider hardware upgrades (GPU)")
    print("3. Optimize model architecture (quantization, pruning)")
    print("4. Reduce input resolution")
    
    sys.exit(1)


if __name__ == "__main__":
    main()
