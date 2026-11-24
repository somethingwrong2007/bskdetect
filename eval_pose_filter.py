"""评估 2D 骨架时序状态层：抖动抑制 / 跟随延迟 / 遮挡预测与恢复 / 单帧开销。

用法：
    .venv/Scripts/python.exe eval_pose_filter.py [--frm 2000] [--frames 90] [--width 960]

流程：
1. 顺序解码一段素材（默认 960 宽，准确档参数），RTMO 逐帧出框 + 17 点，Tracker 分配
   track_id；过滤规则与 server_rt 一致，保证评估输入与线上一致。
2. "优化前" = 旧管线：原始坐标直出 + 每帧做左右镜像补全。
3. "优化后" = bsk.perceive.pose_filter.PoseFilter 在线跑一遍（One Euro 与恒速卡尔曼对比）。
4. 指标口径：抖动只统计"前后三帧都有可靠观测（raw conf>=0.3）"的关节，两种序列用同一个
   关节集合，避免把点藏起来刷指标；镜像点/预测点这类推测坐标不计入抖动分子。

指标：
- 抖动能量：||p[t+1] - 2p[t] + p[t-1]|| / 身高，逐关节逐帧平均（越小越稳）
- 大跳变：单帧位移 > 0.10 身高 的关节帧数（镜像来回切、模型跑飞都会计入）
- 跟随延迟代理：滤波输出与原始点的平均位移偏差 / 身高（越大越"粘"）
- 真实延迟：把滤波序列与原始序列做互相关，取最优时移（0..4 帧），中位数即滞后帧数
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from bsk import ingest, schema
from bsk.analyze import Tracker
from bsk.perceive.persons import Rtmo
from bsk.perceive.pose_filter import PoseFilter, PoseFilterConfig
from bsk.store import Library

VID = "harden-55"
MIRROR_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


# ── 数据采集 ────────────────────────────────────────────────
def collect(frm: int, nframes: int, width: int, cache: bool = True) -> tuple[list[int], list[dict], int, int]:
    """顺序解码 + RTMO + 跟踪，返回 (帧号列表, 每帧 {tid: (bbox, kpts)}, 宽, 高)。

    RTMO 一遍要几十秒，调参要反复跑同一段，所以默认把观测缓存到 .tmp_pf_cache/（不进库）。
    """
    import pickle
    from pathlib import Path
    cpath = Path(".tmp_pf_cache") / f"{VID}_{frm}_{nframes}_{width}.pkl"
    if cache and cpath.exists():
        with open(cpath, "rb") as f:
            frame_ids, obs, w, h = pickle.load(f)
        print(f"[collect] 命中缓存 {cpath}（{len(frame_ids)} 帧 {w}x{h}）")
        return frame_ids, obs, w, h
    lib = Library()
    v = lib.get(VID)
    table = lib.table(VID)
    anchor = table.key_before(frm)
    size = (width, int(round(width * v.height / max(v.width, 1))))
    rtmo = Rtmo()
    tracker = Tracker()

    frame_ids: list[int] = []
    obs: list[dict] = []
    for idx, _pts, rgb in ingest.frames(v.path, start=anchor, size=size, stride=1,
                                        seek_us=table.pts_us[anchor - 1]):
        if idx < frm:
            continue
        if idx >= frm + nframes:
            break
        ppl = rtmo(rgb, thr=0.3)
        keep = []
        for p in ppl:
            bw, bh = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
            if bh / max(bw, 1) < 0.6 or bh / max(bw, 1) > 5.0:
                continue
            if bh < max(36, rgb.shape[0] * 0.03) or bw < max(18, rgb.shape[1] * 0.015):
                continue
            if p.kpts is None or np.sum(p.kpts[:, 2] > 0.5) < 2:
                continue
            keep.append(p)
        ids = tracker.assign([p.bbox for p in keep])
        frame_ids.append(idx)
        obs.append({tid: (p.bbox, np.asarray(p.kpts, dtype=np.float64))
                    for p, tid in zip(keep, ids)})
        print(f"\r[collect] {len(frame_ids)}/{nframes}", end="", flush=True)
    print()
    if cache:
        cpath.parent.mkdir(exist_ok=True)
        with open(cpath, "wb") as f:
            pickle.dump((frame_ids, obs, size[0], size[1]), f, protocol=4)
    return frame_ids, obs, size[0], size[1]


# ── 旧管线：每帧镜像补全 ────────────────────────────────────
def legacy_fix(kpts: np.ndarray, bbox) -> np.ndarray:
    """改造前的 _fix_keypoint_symmetry：一侧 <0.3、另一侧 >0.7 就立刻镜像写回。"""
    k = kpts.copy()
    cx = (bbox[0] + bbox[2]) / 2
    for l, r in MIRROR_PAIRS:
        lc, rc = k[l, 2], k[r, 2]
        if lc < 0.3 and rc > 0.7:
            k[l, 0], k[l, 1], k[l, 2] = 2 * cx - k[r, 0], k[r, 1], rc * 0.8
        elif rc < 0.3 and lc > 0.7:
            k[r, 0], k[r, 1], k[r, 2] = 2 * cx - k[l, 0], k[l, 1], lc * 0.8
    return k


def to_poses(frame: int, frame_obs: dict, legacy: bool) -> tuple[list[schema.Pose2D], dict]:
    poses, boxes = [], {}
    for tid, (bbox, kpts) in frame_obs.items():
        k = legacy_fix(kpts, bbox) if legacy else kpts
        poses.append(schema.Pose2D(video_id=VID, frame=frame, track_id=tid,
                                   kpts=k.tolist(), height_px=float(bbox[3] - bbox[1])))
        boxes[tid] = bbox
    return poses, boxes


# ── 指标 ────────────────────────────────────────────────────
def to_series(frame_ids: list[int], records: list[dict]) -> dict:
    """records[i] = {tid: (kpts(17,3), bbox)} → {tid: (kpts(T,17,3), bbox(T,4))}。"""
    out: dict[int, dict[str, np.ndarray]] = {}
    T = len(frame_ids)
    for i, rec in enumerate(records):
        for tid, (kpts, bbox) in rec.items():
            slot = out.setdefault(tid, {"kpts": np.full((T, 17, 3), np.nan),
                                        "bbox": np.full((T, 4), np.nan)})
            slot["kpts"][i] = kpts
            slot["bbox"][i] = bbox
    return out


def jitter(base: dict, filt: dict, thresh: float = 0.10, static_step: float = 0.03) -> dict:
    """同一关节集合下的抖动能量与大跳变计数（按身高归一化）。

    额外给"静止子集"（原始单帧位移 < 3% 身高）的口径：抖动最容易被肉眼看见的正是
    这种近乎静止的时刻，整体均值会被快速挥臂的真实加速度稀释。
    """
    e_b = e_f = s_b = s_f = 0.0
    n = n_s = 0
    big_b = big_f = 0
    for tid in base:
        if tid not in filt:
            continue
        rb, rb_box = base[tid]["kpts"], base[tid]["bbox"]
        rf = filt[tid]["kpts"]
        T = rb.shape[0]
        h = np.where(np.isnan(rb_box[:, 3] - rb_box[:, 1]), np.nan, rb_box[:, 3] - rb_box[:, 1])
        for t in range(1, T - 1):
            if np.isnan(h[t]) or h[t] < 8:
                continue
            ok = (rb[t - 1: t + 2, :, 2] >= 0.3).all(0) & np.isfinite(rb[t - 1: t + 2, :, :2]).all((0, 2))
            ok &= np.isfinite(rf[t - 1: t + 2, :, :2]).all((0, 2))
            ok &= (rf[t - 1: t + 2, :, 2] > 0).all(0)
            for j in np.nonzero(ok)[0]:
                acc_b = np.linalg.norm(rb[t + 1, j, :2] - 2 * rb[t, j, :2] + rb[t - 1, j, :2]) / h[t]
                acc_f = np.linalg.norm(rf[t + 1, j, :2] - 2 * rf[t, j, :2] + rf[t - 1, j, :2]) / h[t]
                e_b += acc_b
                e_f += acc_f
                n += 1
                step = np.linalg.norm(rb[t, j, :2] - rb[t - 1, j, :2]) / h[t]
                if step < static_step:
                    s_b += acc_b
                    s_f += acc_f
                    n_s += 1
        # 大跳变：单帧位移（不要求三帧连续，覆盖面更广）
        for t in range(1, T):
            if np.isnan(h[t]) or h[t] < 8:
                continue
            ok = (rb[t, :, 2] >= 0.3) & (rb[t - 1, :, 2] >= 0.3) & np.isfinite(rb[t - 1, :, :2]).all(1)
            db = np.linalg.norm(rb[t, :, :2] - rb[t - 1, :, :2], axis=1) / h[t]
            df = np.linalg.norm(rf[t, :, :2] - rf[t - 1, :, :2], axis=1) / h[t]
            big_b += int(np.sum((db > thresh) & ok))
            big_f += int(np.sum((df > thresh) & ok))
    return dict(eb=e_b / max(n, 1), ef=e_f / max(n, 1), bb=big_b, bf=big_f,
                sb=s_b / max(n_s, 1), sf=s_f / max(n_s, 1), n=n, n_static=n_s)


JOINT_NAMES = ("鼻", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
               "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝")


def joint_coverage(base: dict, filt: dict, solid: float = 0.15, strong: float = 0.30) -> dict:
    """逐关节"可见覆盖率"：分母是全部人体帧 × 关节（不挑有没有测到）。

    为什么分母不挑"原始可靠"：挑出来的话覆盖率天然 100%，看不出头部/手腕经常整段
    没点的问题。这里口径是"画面上这个关节到底有没有一个可见点"（渲染 conf>=0.15
    才会连边），补位/镜像/先验补全只要可见就计入分子——这正是下游姿态角用得上的
    东西。补位质量单独看 fill_err：只在同期原始观测可靠（raw conf>=0.5）的帧上，
    把补位坐标和实测比（参照取 raw>=0.15：模型自己也不确定的弱观测，只能当粗略参照），
    偏得远就是补错了（"可疑镜像"）。
    """
    n = np.zeros(17)
    cov_s = np.zeros(17)
    cov_t = np.zeros(17)
    cov_b = np.zeros(17)          # 旧管线（原始+逐帧镜像）参照
    fill_n = np.zeros(17)
    fill_err = np.zeros(17)
    bad_mirror = 0
    mirror_n = 0
    for tid, rb in base.items():
        if tid not in filt:
            continue
        rf = filt[tid]
        k_b, k_f = rb["kpts"], rf["kpts"]
        h = rb["bbox"][:, 3] - rb["bbox"][:, 1]
        for t in range(k_b.shape[0]):
            if not np.isfinite(h[t]) or h[t] < 8:
                continue
            if not np.isfinite(k_b[t, :, :2]).any():
                continue
            n[:] += 1
            cf = k_f[t, :, 2]
            cov_s[cf >= solid] += 1
            cov_t[cf >= strong] += 1
            cov_b[k_b[t, :, 2] >= solid] += 1
            # 补位点（输出 conf 恰好等于补位常数）与实测对比
            fill = is_fill(cf) & (k_b[t, :, 2] >= 0.15) & np.isfinite(k_b[t, :, :2]).all(1)
            if fill.any():
                e = np.linalg.norm(k_f[t, fill, :2] - k_b[t, fill, :2], axis=1) / h[t]
                fill_n[fill] += 1
                fill_err[fill] += e
                bad_mirror += int(np.sum(e > 0.10))
                mirror_n += int(fill.sum())
    return dict(n=n, cov_s=cov_s, cov_t=cov_t, cov_b=cov_b, fill_n=fill_n,
                fill_err=fill_err, bad_mirror=bad_mirror, mirror_n=mirror_n)


def is_fill(conf: np.ndarray) -> bool:
    """输出 conf 恰好等于补位常数（CONF_PRIOR/MIRROR/TEMPLATE）= 这是补出来的点。

    实测 conf 是模型连续输出，恰好落在 0.20/0.30/0.35 上的概率可忽略，所以可以精确判定。
    """
    from bsk.perceive.pose_filter import CONF_MIRROR, CONF_PRIOR, CONF_TEMPLATE
    return (conf == CONF_PRIOR) | (conf == CONF_MIRROR) | (conf == CONF_TEMPLATE)


def print_coverage(cov: dict, solid: float = 0.15) -> None:
    n = np.maximum(cov["n"], 1)
    print(f"  逐关节可见覆盖率（分母=全部人体帧，solid≥{solid}）：")
    for j, name in enumerate(JOINT_NAMES):
        if cov["n"][j] < 1:
            continue
        err = cov["fill_err"][j] / max(cov["fill_n"][j], 1)
        print(f"    {name:<3} 原始 {100*cov['cov_b'][j]/n[j]:5.1f}% → 输出 "
              f"{100*cov['cov_s'][j]/n[j]:5.1f}% (strong {100*cov['cov_t'][j]/n[j]:5.1f}%)"
              f"  补位 {cov['fill_n'][j]:4.0f} 次 误差 {100*err:.1f}% 身高")
    grp = {"头部5点": (0, 1, 2, 3, 4), "双腕": (9, 10)}
    for g, js in grp.items():
        js = list(js)
        tot = n[js].sum()
        print(f"  [{g}] 原始 {100*cov['cov_b'][js].sum()/max(tot,1):.1f}% → "
              f"输出 {100*cov['cov_s'][js].sum()/max(tot,1):.1f}%"
              f"（strong {100*cov['cov_t'][js].sum()/max(tot,1):.1f}%）")
    if cov["mirror_n"]:
        print(f"  补位点质量：{cov['mirror_n']} 个可比对补位中 {cov['bad_mirror']} 个偏移 >10% 身高"
              f"（{100*cov['bad_mirror']/cov['mirror_n']:.1f}%）")


def lag(base: dict, filt: dict) -> tuple[float, float, float]:
    """跟随延迟代理：平均位移偏差、最优时移中位数、时移后残差。"""
    devs, shifts, resid = [], [], []
    for tid in base:
        if tid not in filt:
            continue
        rb, rb_box, rf = base[tid]["kpts"], base[tid]["bbox"], filt[tid]["kpts"]
        h = rb_box[:, 3] - rb_box[:, 1]
        for t in range(rb.shape[0]):
            if np.isnan(h[t]) or h[t] < 8:
                continue
            ok = (rb[t, :, 2] >= 0.3) & np.isfinite(rf[t, :, :2]).all(1)
            if not ok.any():
                continue
            devs.extend((np.linalg.norm(rf[t, ok, :2] - rb[t, ok, :2], axis=1) / h[t]).tolist())
        # 互相关时移：滤波输出 vs 原始输入延迟多少帧最像
        for j in range(17):
            best_k, best_v = 0, 1e9
            for k in range(0, 5):
                vals = []
                for t in range(k, rb.shape[0]):
                    if np.isnan(h[t]) or h[t] < 8:
                        continue
                    if rb[t, j, 2] >= 0.3 and rb[t - k, j, 2] >= 0.3 and rf[t, j, 2] > 0:
                        vals.append(np.linalg.norm(rf[t, j, :2] - rb[t - k, j, :2]) / h[t])
                if len(vals) > 10:
                    v = float(np.mean(vals))
                    if v < best_v:
                        best_k, best_v = k, v
            if best_v < 1e8:
                shifts.append(best_k)
                resid.append(best_v)
    return float(np.mean(devs)) if devs else 0.0, \
        float(np.median(shifts)) if shifts else 0.0, \
        float(np.mean(resid)) if resid else 0.0


# ── 遮挡实验 ────────────────────────────────────────────────
def pick_track(series: dict) -> int:
    best, best_tid = -1, None
    for tid, slot in series.items():
        n = int(np.sum(np.isfinite(slot["kpts"][:, 0, 0])))
        conf = np.nanmean(np.where(np.isfinite(slot["kpts"][:, :, 2]), slot["kpts"][:, :, 2], np.nan))
        if n >= 25 and conf > 0.45 and n > best:
            best, best_tid = n, tid
    if best_tid is None:
        raise SystemExit("没有足够长的轨迹可用于遮挡实验")
    return best_tid


def run_filter(frame_ids, obs, cfg, mutate=None):
    """跑一遍在线滤波；mutate(frame, frame_obs) 可在喂入前篡改观测（模拟遮挡）。"""
    pf = PoseFilter(cfg)
    out = []
    t_total = 0.0
    for frame, fobs in zip(frame_ids, obs):
        fobs = {k: (v[0], v[1].copy()) for k, v in fobs.items()}
        if mutate is not None:
            mutate(frame, fobs)
        poses, boxes = to_poses(frame, fobs, legacy=False)
        t0 = time.perf_counter()
        poses_out, occ_boxes = pf.apply(frame, poses, boxes)
        t_total += time.perf_counter() - t0
        rec = {p.track_id: (np.asarray(p.kpts), boxes.get(p.track_id, (0, 0, 0, 1)))
               for p in poses_out}
        for tid, box in occ_boxes.items():
            if tid in rec:
                rec[tid] = (rec[tid][0], box)
        out.append(rec)
    return out, t_total / max(len(frame_ids), 1) * 1000


def occlusion_experiments(frame_ids, obs, series, cfg):
    """遮挡实验：把某个关节/整人的观测清零若干帧，看补位质量与恢复是否跳变。

    "恢复跳变"用归一化单帧位移衡量：恢复首帧的位移 / 该轨迹正常帧的位移中位数，
    比值 ~1 说明恢复和正常运动一样平顺，远大于 1 就是"弹回"。
    """
    tid = pick_track(series)
    present = [i for i, rec in enumerate(obs) if tid in rec]
    start = present[len(present) // 2]
    joints = (9, 15)  # 左腕 + 左踝
    h_of = lambda k: max(float(obs[k][tid][0][3] - obs[k][tid][0][1]), 8.0)  # noqa: E731

    # 正常帧的单帧位移（作为"平顺"的参照）；用原始观测算，避免用被滤波后的序列自证
    norm_step = []
    for k in range(1, len(frame_ids)):
        if tid in obs[k] and tid in obs[k - 1]:
            d = np.linalg.norm(obs[k][tid][1][:, :2] - obs[k - 1][tid][1][:, :2], axis=1) / h_of(k)
            norm_step.extend(d.tolist())
    step_ref = float(np.median(norm_step))
    print(f"\n[遮挡实验] track #{tid}，起点帧 {frame_ids[start]}，"
          f"正常单帧位移中位数 {100*step_ref:.1f}% 身高")

    for dur in (5, 10, 15):
        print(f"- 遮挡 {dur} 帧（{dur/30:.2f}s）")
        # 1) 关节级：左腕 + 左踝置信度清零
        def drop_joints(frame, fobs, dur=dur):
            i = frame_ids.index(frame)
            if start <= i < start + dur and tid in fobs:
                bbox, k = fobs[tid]
                k = k.copy()
                k[list(joints), 2] = 0.0
                fobs[tid] = (bbox, k)

        filt, _ = run_filter(frame_ids, obs, cfg, drop_joints)
        cover, errs, step, step6 = [], [], [], []
        for k in range(start, min(start + dur + 2, len(frame_ids))):
            if tid not in obs[k] or tid not in filt[k]:
                continue
            rk, fk = obs[k][tid][1], filt[k][tid][0]
            if start <= k < start + dur:
                for j in joints:
                    cover.append(float(fk[j, 2] > 0))
                    errs.append(float(np.linalg.norm(fk[j, :2] - rk[j, :2]) / h_of(k)))
            if k == start + dur and tid in filt[k - 1]:
                prev = filt[k - 1][tid][0]
                for j in joints:
                    step.append(float(np.linalg.norm(fk[j, :2] - prev[j, :2]) / h_of(k)))
            if start + dur <= k <= start + dur + 6 and k - 1 in range(len(filt))                     and tid in filt[k] and tid in filt[k - 1]:
                cur, prev = filt[k][tid][0], filt[k - 1][tid][0]
                for j in joints:
                    step6.append(float(np.linalg.norm(cur[j, :2] - prev[j, :2]) / h_of(k)))
        print(f"  关节级：补位覆盖率 {100*np.mean(cover):.0f}%（优化前 0%，点直接消失），"
              f"预测误差 {100*np.mean(errs):.1f}% 身高，"
              f"恢复首帧位移 {100*np.mean(step):.1f}% = {np.mean(step)/max(step_ref,1e-6):.1f}x 正常，"
              f"恢复后 0.2s 内最大单帧位移 {100*max(step6 or [0]):.1f}% "
              f"= {max(step6 or [0])/max(step_ref,1e-6):.1f}x")

        # 2) 人体级：整人观测抽掉
        def drop_person(frame, fobs, dur=dur):
            i = frame_ids.index(frame)
            if start <= i < start + dur:
                fobs.pop(tid, None)

        filt2, _ = run_filter(frame_ids, obs, cfg, drop_person)
        cover2, iou2, err2, step2 = [], [], [], []
        for k in range(start, min(start + dur + 2, len(frame_ids))):
            if tid not in obs[k]:
                continue
            bbox, rk = obs[k][tid]
            if start <= k < start + dur and tid in filt2[k]:
                cover2.append(1.0)
                fk, fb = filt2[k][tid]
                iw = max(0.0, min(fb[2], bbox[2]) - max(fb[0], bbox[0]))
                ih = max(0.0, min(fb[3], bbox[3]) - max(fb[1], bbox[1]))
                inter = iw * ih
                union = (fb[2] - fb[0]) * (fb[3] - fb[1]) + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter
                iou2.append(inter / union if union > 0 else 0.0)
                err2.append(float(np.mean(np.linalg.norm(fk[:, :2] - rk[:, :2], axis=1)) / h_of(k)))
            if start <= k < start + dur and tid not in filt2[k]:
                cover2.append(0.0)
            if k == start + dur and tid in filt2[k] and tid in filt2[k - 1]:
                d = np.linalg.norm(filt2[k][tid][0][:, :2] - filt2[k - 1][tid][0][:, :2], axis=1) / h_of(k)
                step2.append(float(np.mean(d)))
        rec_txt = (f"恢复首帧位移 {100*np.mean(step2):.1f}% = "
                   f"{np.mean(step2)/max(step_ref,1e-6):.1f}x 正常") if step2 else                   "恢复首帧无样本（遮挡超过 max_track_age，track 已按设计丢弃）"
        print(f"  人体级：track 维持率 {100*np.mean(cover2):.0f}%（优化前 0%，骨架整段消失），"
              f"预测框 IoU {np.mean(iou2):.2f}，骨架平均误差 {100*np.mean(err2):.1f}% 身高，"
              f"{rec_txt}")


def natural_occlusion(frame_ids, obs, cfg) -> None:
    """真实遮挡统计：素材里"人被挡住"（整人检不出）恰好 1-10 帧的空洞，看能否桥接。

    比注入实验更可信的地方是：遮挡原因、运动状态都是真实的。只统计已确认（>=10 帧）
    的轨迹，缺口刚好在 max_track_age 内的算"可桥接"，超出的按设计丢弃。
    """
    gaps = []          # (tid, 缺失起点下标, 长度)
    for tid in {t for rec in obs for t in rec}:
        present = [i for i, rec in enumerate(obs) if tid in rec]
        if len(present) < 10:
            continue
        for a, b in zip(present, present[1:]):
            if 1 < b - a <= 20:
                gaps.append((tid, a + 1, b - a - 1))
    if not gaps:
        print("[真实遮挡] 该片段没有整人级别的检测空洞")
        return
    bridged = sum(1 for _t, _s, n in gaps if n <= cfg.max_track_age)
    print(f"[真实遮挡] 检出空洞 {len(gaps)} 处（中位 {statistics.median([g[2] for g in gaps]):.0f} 帧），"
          f"其中 {bridged} 处（{100*bridged/len(gaps):.0f}%）在 max_track_age={cfg.max_track_age} 内可桥接")
    # 桥接质量：缺口恢复首帧的位移 vs 该轨迹正常步长
    recs, _ = run_filter(frame_ids, obs, cfg)
    series = to_series(frame_ids, recs)
    jumps, refs = [], []
    for tid, start, n in gaps:
        if n > cfg.max_track_age or tid not in series:
            continue
        k = series[tid]["kpts"]
        box = series[tid]["bbox"]
        i0, i1 = start + n - 1, start + n
        if i1 >= k.shape[0]:
            continue
        h = box[i1, 3] - box[i1, 1]
        if not np.isfinite(h) or h < 8 or not (k[i1, :, 2] > 0).any() or not (k[i0, :, 2] > 0).any():
            continue
        d = np.linalg.norm(k[i1, :, :2] - k[i0, :, :2], axis=1) / h
        jumps.append(float(np.mean(d)))
        ok = (k[i1, :, 2] >= 0.3) & (k[i0, :, 2] >= 0.3)
        if ok.any():
            refs.append(float(np.mean(d[ok])))
    if jumps:
        ref_txt = f"，同轨迹正常步长 {100*np.mean(refs):.1f}%" if refs else ""
        print(f"    桥接后恢复首帧位移 均值 {100*np.mean(jumps):.1f}% 身高"
              f"（p90 {100*np.percentile(jumps, 90):.1f}%）{ref_txt}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--frm", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--occlusion", action="store_true", help="跑遮挡与恢复实验")
    args = ap.parse_args()

    frame_ids, obs, w, h = collect(args.frm, args.frames, args.width)
    print(f"[collect] {len(frame_ids)} 帧 {w}x{h}，"
          f"人数中位数 {statistics.median([len(o) for o in obs]):.0f}")

    # 优化前：原始坐标 + 每帧镜像补全
    base_recs = []
    for f, o in zip(frame_ids, obs):
        rec = {}
        for tid, (bbox, k) in o.items():
            rec[tid] = (legacy_fix(k, bbox), bbox)
        base_recs.append(rec)
    base = to_series(frame_ids, base_recs)

    results = {}
    for name, cfg in (
        ("OneEuro", PoseFilterConfig()),
        ("Kalman", PoseFilterConfig(mode="kalman", kalman_k=0.55, kalman_b=0.12)),
    ):
        recs, ms = run_filter(frame_ids, obs, cfg)
        series = to_series(frame_ids, recs)
        j = jitter(base, series)
        dev, shift, res = lag(base, series)
        results[name] = dict(series=series, cfg=cfg, ms=ms, j=j, dev=dev, shift=shift, res=res)
        print(f"\n[{name}] 开销 {ms:.2f}ms/帧，样本 {j['n']} 关节帧（静止子集 {j['n_static']}）")
        print(f"  抖动能量 优化前 {j['eb']*100:.2f}% → 优化后 {j['ef']*100:.2f}% 身高/帧^2"
              f"（下降 {100*(1-j['ef']/max(j['eb'],1e-9)):.0f}%）")
        print(f"  其中静止子集 {j['sb']*100:.2f}% → {j['sf']*100:.2f}%"
              f"（下降 {100*(1-j['sf']/max(j['sb'],1e-9)):.0f}%）")
        print(f"  大跳变(>10%身高/帧) {j['bb']} → {j['bf']} 处（下降 {100*(1-j['bf']/max(j['bb'],1)):.0f}%）")
        print(f"  跟随延迟代理 {dev*100:.2f}% 身高；互相关最优时移 {shift:.0f} 帧，"
              f"时移后残差 {res*100:.2f}%")
        print_coverage(joint_coverage(base, series))

    if args.occlusion:
        cfg = results["OneEuro"]["cfg"]
        occlusion_experiments(frame_ids, obs, base, cfg)
        natural_occlusion(frame_ids, obs, cfg)


if __name__ == "__main__":
    main()