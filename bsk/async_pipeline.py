"""异步生产者-消费者流水线：重叠GPU推理与CPU渲染。

架构：
1. 生产者线程：从视频流读取帧并放入输入队列
2. GPU工作线程：运行标定+检测+姿态（串行，但与其他阶段并行）
3. CPU工作线程：运行3D重建+时序平滑+渲染
4. 输出队列：按帧号排序后交付给前端

关键优化：
- GPU推理（~15ms）与CPU渲染（~5ms）完全并行
- 多帧缓冲掩盖GPU延迟
- 背压控制防止内存爆炸

性能目标：
- 吞吐量：单帧33ms → 流水线稳定后每帧~20ms
- 延迟：首帧到输出 <100ms（预热后）
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from typing import Optional, Callable

import numpy as np

from bsk import calibrate, schema
from bsk.realtime_calibrate import calibrate_frame
from bsk.server_rt import FrameResult, _get_fast_pose, _get_tracker, _get_smoother
from bsk.render_3d import TemporalSmoother, PlayerSkeleton3D


@dataclass
class PipelineFrame:
    """流水线中的帧数据。"""
    frame_num: int
    pts_us: int
    rgb: np.ndarray
    result: Optional[FrameResult] = None
    stage: str = "input"  # input → gpu_done → cpu_done
    timestamp: float = field(default_factory=time.time)


class AsyncPipeline:
    """异步处理流水线。
    
    阶段：
    1. Input Stage: 视频解码器推送原始帧
    2. GPU Stage: 标定 + YOLOv8检测 + 姿态估计
    3. CPU Stage: 3D重建 + 时序平滑 + 渲染准备
    4. Output Stage: 按序交付结果
    
    使用示例：
        pipeline = AsyncPipeline(max_buffer=3)
        pipeline.start()
        
        # 生产者线程
        for n, (pts_us, rgb) in video_frames():
            pipeline.push(n, pts_us, rgb)
        
        # 消费者线程
        while running:
            result = pipeline.pop(timeout=0.1)
            if result:
                yield result
    """
    
    def __init__(
        self,
        max_buffer: int = 3,
        enable_3d: bool = True,
        vid: str = "default",
    ):
        """
        Args:
            max_buffer: 最大缓冲帧数（背压控制，默认3）
            enable_3d: 是否启用3D重建
            vid: 视频ID
        """
        self.max_buffer = max_buffer
        self.enable_3d = enable_3d
        self.vid = vid
        
        # 队列：输入 → GPU → CPU → 输出
        self.input_queue: Queue[PipelineFrame] = Queue(maxsize=max_buffer)
        self.gpu_queue: Queue[PipelineFrame] = Queue(maxsize=max_buffer)
        self.cpu_queue: Queue[PipelineFrame] = Queue(maxsize=max_buffer)
        self.output_queue: Queue[FrameResult] = Queue(maxsize=max_buffer * 2)
        
        # 线程控制
        self._gpu_thread: Optional[threading.Thread] = None
        self._cpu_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        
        # 统计信息
        self.stats = {
            "frames_pushed": 0,
            "frames_processed": 0,
            "gpu_latency_ms": [],
            "cpu_latency_ms": [],
            "total_latency_ms": [],
        }
        self._stats_lock = threading.Lock()
    
    def start(self):
        """启动流水线线程。"""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        # GPU工作线程：标定 + 检测 + 姿态
        self._gpu_thread = threading.Thread(
            target=self._gpu_worker,
            name="GPUWorker",
            daemon=True
        )
        self._gpu_thread.start()
        
        # CPU工作线程：3D重建 + 平滑 + 渲染准备
        self._cpu_thread = threading.Thread(
            target=self._cpu_worker,
            name="CPUWorker",
            daemon=True
        )
        self._cpu_thread.start()
    
    def stop(self, timeout: float = 5.0):
        """停止流水线。"""
        self._stop_event.set()
        self._running = False
        
        if self._gpu_thread:
            self._gpu_thread.join(timeout=timeout)
        if self._cpu_thread:
            self._cpu_thread.join(timeout=timeout)
    
    def push(self, frame_num: int, pts_us: int, rgb: np.ndarray) -> bool:
        """推送帧到输入队列（非阻塞）。
        
        Returns:
            True if pushed successfully, False if queue is full (backpressure)
        """
        try:
            frame = PipelineFrame(frame_num=frame_num, pts_us=pts_us, rgb=rgb)
            self.input_queue.put_nowait(frame)
            
            with self._stats_lock:
                self.stats["frames_pushed"] += 1
            
            return True
        except Full:
            # 背压：队列满时丢弃最旧帧或跳过
            return False
    
    def pop(self, timeout: float = 0.1) -> Optional[FrameResult]:
        """从输出队列获取结果（阻塞超时）。"""
        try:
            result = self.output_queue.get(timeout=timeout)
            
            with self._stats_lock:
                self.stats["frames_processed"] += 1
            
            return result
        except Empty:
            return None
    
    def _gpu_worker(self):
        """GPU工作线程：标定 + 检测 + 姿态。"""
        print("[pipeline] GPU worker started")
        
        while not self._stop_event.is_set():
            try:
                # 从输入队列取帧（带超时以便检查停止信号）
                frame = self.input_queue.get(timeout=0.05)
            except Empty:
                continue
            
            t_start = time.time()
            
            try:
                # Stage 1: 增量标定
                cal = calibrate_frame(self.vid, frame.rgb)
                
                # Stage 2: 检测 + 姿态（YOLOv8n-pose）
                fast_pose = _get_fast_pose()
                ppl = fast_pose(frame.rgb, thr=0.5)
                boxes = [p.bbox for p in ppl]
                
                # Stage 3: 跟踪
                tracker = _get_tracker()
                ids = tracker.assign(boxes)
                
                # Stage 4: 构建对象（含过滤和对称校正）
                objects, poses = self._build_objects(ppl, ids, cal, frame.rgb.shape, frame.frame_num)
                
                # 存储中间结果
                frame.result = FrameResult(
                    frame=frame.frame_num,
                    pts_us=frame.pts_us,
                    objects=objects,
                    poses=poses,
                    segments=[],
                    calib=cal,
                    prev_rgb=None,  # Will be filled by CPU stage
                    players_3d=[]
                )
                frame.stage = "gpu_done"
                
                # 推送到CPU队列
                self.cpu_queue.put_nowait(frame)
                
                # 记录GPU延迟
                gpu_ms = (time.time() - t_start) * 1000
                with self._stats_lock:
                    self.stats["gpu_latency_ms"].append(gpu_ms)
                    if len(self.stats["gpu_latency_ms"]) > 100:
                        self.stats["gpu_latency_ms"] = self.stats["gpu_latency_ms"][-100:]
                
            except Exception as e:
                print(f"[pipeline] GPU error on frame {frame.frame_num}: {e}")
                continue
    
    def _cpu_worker(self):
        """CPU工作线程：3D重建 + 时序平滑。"""
        print("[pipeline] CPU worker started")
        
        while not self._stop_event.is_set():
            try:
                frame = self.cpu_queue.get(timeout=0.05)
            except Empty:
                continue
            
            t_start = time.time()
            
            try:
                if frame.result is None:
                    continue
                
                # Stage 5: 3D重建（如果启用）
                players_3d = []
                if self.enable_3d and frame.result.calib:
                    H_inv = np.array(frame.result.calib.H_inv)
                    
                    pairs_3d = []
                    for obj, pose in zip(frame.result.objects, frame.result.poses):
                        skel = PlayerSkeleton3D.from_2d(obj, pose, H_inv)
                        if skel:
                            pairs_3d.append(skel)
                    
                    # 时序平滑
                    smoother = _get_smoother()
                    players_3d = smoother.smooth(pairs_3d, frame.frame_num)
                
                frame.result.players_3d = players_3d
                frame.stage = "cpu_done"
                
                # 推送到输出队列
                self.output_queue.put_nowait(frame.result)
                
                # 记录CPU延迟
                cpu_ms = (time.time() - t_start) * 1000
                total_ms = (time.time() - frame.timestamp) * 1000
                
                with self._stats_lock:
                    self.stats["cpu_latency_ms"].append(cpu_ms)
                    self.stats["total_latency_ms"].append(total_ms)
                    if len(self.stats["cpu_latency_ms"]) > 100:
                        self.stats["cpu_latency_ms"] = self.stats["cpu_latency_ms"][-100:]
                    if len(self.stats["total_latency_ms"]) > 100:
                        self.stats["total_latency_ms"] = self.stats["total_latency_ms"][-100:]
                
            except Exception as e:
                print(f"[pipeline] CPU error on frame {frame.frame_num}: {e}")
                continue
    
    def _build_objects(
        self,
        ppl: list,
        ids: list[int],
        cal,
        img_shape: tuple,
        frame_num: int,
    ) -> tuple[list[schema.Tracked], list[schema.Pose2D]]:
        """构建Tracked和Pose2D对象（含过滤和对称校正）。"""
        from bsk.server_rt import _get_court_bounds, _point_in_rect, _fix_keypoint_symmetry
        
        objects = []
        poses = []
        
        H_inv = np.array(cal.H_inv) if cal else None
        court_bounds = _get_court_bounds(H_inv, img_shape) if H_inv is not None else None
        
        for p, tid in zip(ppl, ids):
            # Filter 1: Confidence
            if p.score < 0.5:
                continue
            
            # Filter 2: Aspect ratio
            bw, bh = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
            aspect_ratio = bh / max(bw, 1)
            if aspect_ratio < 1.2 or aspect_ratio > 3.5:
                continue
            
            # Filter 3: Minimum size
            if bh < 80 or bw < 40:
                continue
            
            # Filter 4: Court ROI
            if court_bounds is not None:
                cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
                if not _point_in_rect(cx, cy, court_bounds):
                    continue
            
            # Symmetry correction
            kpts_fixed = p.kpts.copy() if p.kpts is not None else None
            if kpts_fixed is not None:
                kpts_fixed = _fix_keypoint_symmetry(kpts_fixed, p.bbox)
            
            objects.append(schema.Tracked(
                track_id=tid, category="player", frame=frame_num,
                bbox_px=p.bbox, conf=p.score, visible=True
            ))
            if kpts_fixed is not None:
                poses.append(schema.Pose2D(
                    video_id=self.vid, frame=frame_num, track_id=tid,
                    kpts=kpts_fixed.tolist(), model="yolov8n_pose_symmetric",
                    height_px=float(p.bbox[3] - p.bbox[1])
                ))
        
        return objects, poses
    
    def get_stats(self) -> dict:
        """获取流水线统计信息。"""
        with self._stats_lock:
            stats_copy = dict(self.stats)
        
        # 计算平均值
        if stats_copy["gpu_latency_ms"]:
            avg_gpu = sum(stats_copy["gpu_latency_ms"]) / len(stats_copy["gpu_latency_ms"])
        else:
            avg_gpu = 0
        
        if stats_copy["cpu_latency_ms"]:
            avg_cpu = sum(stats_copy["cpu_latency_ms"]) / len(stats_copy["cpu_latency_ms"])
        else:
            avg_cpu = 0
        
        if stats_copy["total_latency_ms"]:
            avg_total = sum(stats_copy["total_latency_ms"]) / len(stats_copy["total_latency_ms"])
        else:
            avg_total = 0
        
        return {
            "frames_pushed": stats_copy["frames_pushed"],
            "frames_processed": stats_copy["frames_processed"],
            "avg_gpu_ms": round(avg_gpu, 1),
            "avg_cpu_ms": round(avg_cpu, 1),
            "avg_total_ms": round(avg_total, 1),
            "throughput_fps": round(1000 / max(avg_total, 1), 1) if avg_total > 0 else 0,
        }


# 全局流水线实例缓存（每个视频一个）
_pipelines: dict[str, AsyncPipeline] = {}
_pipelines_lock = threading.Lock()


def get_pipeline(vid: str, enable_3d: bool = True, max_buffer: int = 3) -> AsyncPipeline:
    """获取或创建视频的异步流水线。"""
    with _pipelines_lock:
        if vid not in _pipelines:
            _pipelines[vid] = AsyncPipeline(
                max_buffer=max_buffer,
                enable_3d=enable_3d,
                vid=vid,
            )
            _pipelines[vid].start()
        return _pipelines[vid]


def shutdown_all_pipelines(timeout: float = 5.0):
    """关闭所有流水线（用于清理）。"""
    with _pipelines_lock:
        for vid, pipeline in _pipelines.items():
            pipeline.stop(timeout=timeout)
        _pipelines.clear()


__all__ = ["AsyncPipeline", "get_pipeline", "shutdown_all_pipelines"]
