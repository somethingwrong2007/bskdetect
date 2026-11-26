/**
 * 三维复刻查看器：球场（米制）+ 球员 17 点骨架，世界坐标 x/y/z 与 bsk.court 一致。
 *
 * 依赖：gui/vendor/three.min.js（Three.js r149，MIT，见 gui/vendor/THREE-LICENSE.txt）。
 * 本地 vendor 的重要原因：运行时不依赖 CDN，内网/离线也能打开。
 *
 * 坐标约定（务必与后端一致）：
 *   世界 (x, y, z) 米：原点球场中心，x 沿长边（A 队进攻方向为正），y 沿短边，z 向上。
 *   Three.js 是 y 向上、右手系，这里统一映射 (x, y, z) -> (x, z, -y)。
 *   所有几何都先按世界坐标生成，再经 w2t() 落到 Three.js，避免"某个轴忘了翻"。
 *
 * 尺寸来源：优先用 /rt_3d 载荷里的 court（后端 bsk.court.Court.dims() 生成），
 * 下面的 FALLBACK_DIMS 只是服务端不给时的兜底值，数值与 bsk/court.py 对齐。
 *
 * 数据流：
 *   2D 播放 -> <img> 请求 /rt_frame.jpg?enable_3d=1 -> 服务端分析（含 3D 提升）
 *   本模块 -> GET /rt_3d?n=...（只读缓存，不重跑检测）-> 骨架 -> 插值渲染
 * 不预取未来帧：只在 app.js 显示某一帧时请求该帧，缓冲里只有已到达的历史帧。
 */
(function () {
  'use strict';

  const T = window.THREE;
  const DEG = Math.PI / 180;
  const EDGES = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
  ];
  const PALETTE = [0x4fc3f7, 0xffb74d, 0x81c784, 0xe57373, 0xba68c8,
    0x4db6ac, 0xfff176, 0x90a4ae, 0xf06292, 0xaed581];
  const LINE_Z = 0.006;      // 地面线抬高一点，避免 z-fighting

  // 与 bsk/court.py 对齐的兜底尺寸（服务端 court 载荷优先）
  const FALLBACK_DIMS = {
    FIBA: {
      model: 'FIBA', length: 28.0, width: 15.0, three_pt_radius: 6.75,
      three_pt_corner_dist: 0.90, free_throw_dist: 5.80, free_throw_width: 3.60,
      restricted_radius: 1.30, paint_width: 4.90, center_circle_radius: 1.80,
      rim_height: 3.05, rim_offset_from_baseline: 1.575,
      backboard_from_baseline: 1.20, backboard_width: 1.80,
      backboard_bottom: 2.90, backboard_height: 1.05, rim_radius: 0.225,
      line_width: 0.05, center_line_extension: 0.15, free_throw_dashed_inside: true,
    },
    NBA: {
      model: 'NBA', length: 28.65, width: 15.24, three_pt_radius: 7.239,
      three_pt_corner_dist: 0.9144, free_throw_dist: 5.7912, free_throw_width: 3.6576,
      restricted_radius: 1.2192, paint_width: 4.8768, center_circle_radius: 1.8288,
      rim_height: 3.05, rim_offset_from_baseline: 1.6002,
      backboard_from_baseline: 1.2192, backboard_width: 1.80,
      backboard_bottom: 2.90, backboard_height: 1.05, rim_radius: 0.225,
      line_width: 0.0508, center_line_extension: 0.0, free_throw_dashed_inside: false,
    },
    '3x3': {
      model: '3x3', length: 15.0, width: 11.0, three_pt_radius: 6.75,
      three_pt_corner_dist: 0.90, free_throw_dist: 5.80, free_throw_width: 3.60,
      restricted_radius: 1.30, paint_width: 4.90, center_circle_radius: 1.80,
      rim_height: 3.05, rim_offset_from_baseline: 1.575,
      backboard_from_baseline: 1.20, backboard_width: 1.80,
      backboard_bottom: 2.90, backboard_height: 1.05, rim_radius: 0.225,
      line_width: 0.05, center_line_extension: 0.0, free_throw_dashed_inside: true,
    },
  };

  const S = {
    available: false,
    error: null,
    inited: false,
    scene: null, camera: null, renderer: null,
    courtGroup: null, playerGroup: null, canvas: null, statusEl: null,
    dims: FALLBACK_DIMS.FIBA,
    model: 'FIBA',
    yaw: Math.PI / 2, pitch: 24 * DEG, dist: 27,
    yawT: Math.PI / 2, pitchT: 24 * DEG, distT: 27,
    vid: null, quality: 'accurate', width: 960,
    enabled: false,
    demo: false, demoStart: 0,
    frame: 0,
    curr: null, prev: null, blendT0: 0, blendDur: 60,
    players: {},              // track_id -> { group, joints, bones, ring, label }
    buf: new Map(),           // frame -> { players, status, calib_ok, detail }
    inflight: new Set(),
    lastStatus: null,
    lastArrival: 0,
    fps: 30,
    dragging: false, lastX: 0, lastY: 0,
    frames: 0, lastFpsT: 0, renderFps: 0,
  };

  // ── 世界坐标 → Three.js ───────────────────────────────────
  function w2t(x, y, z) { return new T.Vector3(x, z, -y); }

  function statusEl() {
    if (S.statusEl && document.body.contains(S.statusEl)) return S.statusEl;
    const host = document.querySelector('.viewer-container') || document.body;
    const el = document.getElementById('view3d-status');
    if (el) { S.statusEl = el; return el; }
    return null;
  }

  function setStatus(msg, kind) {
    const el = statusEl();
    if (!el) return;
    if (!msg) { el.style.display = 'none'; el.textContent = ''; return; }
    el.textContent = msg;
    el.style.display = 'block';
    el.dataset.kind = kind || 'info';
  }

  // ── 初始化 ────────────────────────────────────────────────
  function init() {
    if (S.inited) return true;
    S.canvas = document.getElementById('view3d');
    if (!S.canvas) return false;
    if (!T) {
      S.error = 'Three.js 未加载（gui/vendor/three.min.js 缺失或 404）';
      setStatus('3D 不可用：' + S.error, 'error');
      console.error('[3D] ' + S.error);
      return false;
    }
    const scene = new T.Scene();
    scene.background = new T.Color(0x10141c);
    scene.fog = new T.Fog(0x10141c, 55, 120);
    S.scene = scene;

    const camera = new T.PerspectiveCamera(50, 16 / 9, 0.1, 400);
    S.camera = camera;

    const renderer = new T.WebGLRenderer({ canvas: S.canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = false;
    S.renderer = renderer;

    scene.add(new T.AmbientLight(0xffffff, 0.85));
    const dir = new T.DirectionalLight(0xffffff, 0.55);
    dir.position.set(6, 18, 8);
    scene.add(dir);
    const fill = new T.DirectionalLight(0x88aaff, 0.25);
    fill.position.set(-10, 8, -12);
    scene.add(fill);

    S.courtGroup = new T.Group();
    S.playerGroup = new T.Group();
    scene.add(S.courtGroup, S.playerGroup);
    buildCourt(S.model);

    bindControls();
    S.inited = true;
    S.available = true;
    updateCamera(true);
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
    return true;
  }

  function resize() {
    if (!S.renderer || !S.canvas) return;
    const w = S.canvas.clientWidth || S.canvas.parentElement?.clientWidth || 960;
    const h = S.canvas.clientHeight || S.canvas.parentElement?.clientHeight || 540;
    if (!w || !h) return;
    S.renderer.setSize(w, h, false);
    S.camera.aspect = w / h;
    S.camera.updateProjectionMatrix();
  }

  // ── 球场几何 ──────────────────────────────────────────────
  function buildCourt(model) {
    S.model = model || S.model;
    const dims = (S.dims && S.dims.model === S.model) ? S.dims : (FALLBACK_DIMS[S.model] || FALLBACK_DIMS.FIBA);
    const g = S.courtGroup;
    while (g.children.length) {
      const c = g.children.pop();
      c.traverse?.((o) => { o.geometry?.dispose?.(); });
    }
    const hl = dims.length / 2, hw = dims.width / 2;
    const segs = [];
    const add = (x1, y1, x2, y2, z) => {
      const p1 = w2t(x1, y1, z || LINE_Z), p2 = w2t(x2, y2, z || LINE_Z);
      segs.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    };
    const arc = (cx, cy, r, a0, a1, n, z) => {
      let prev = null;
      for (let i = 0; i <= n; i++) {
        const t = a0 + (a1 - a0) * i / n;
        const p = [cx + r * Math.cos(t), cy + r * Math.sin(t)];
        if (prev) add(prev[0], prev[1], p[0], p[1], z);
        prev = p;
      }
    };
    const dashedArc = (cx, cy, r, a0, a1, dash = 0.15, gap = 0.10) => {
      const dir = a1 >= a0 ? 1 : -1;
      const len = Math.abs(a1 - a0) * r;
      let pos = 0;
      while (pos < len) {
        const end = Math.min(pos + dash, len);
        const s = a0 + dir * pos / r;
        const e = a0 + dir * end / r;
        arc(cx, cy, r, s, e, 4);
        pos += dash + gap;
      }
    };

    const lw = dims.line_width ?? 0.05;
    const hwLine = lw / 2;
    const ext = lw / 2 + (dims.center_line_extension ?? 0);
    const bx = hl + hwLine, by = hw + hwLine;

    // 外框（中心线在场地内沿外半个线宽）+ 中线
    add(-bx, -by, bx, -by); add(bx, -by, bx, by); add(bx, by, -bx, by); add(-bx, by, -bx, -by);
    add(0, -(hw + ext), 0, hw + ext);
    // 中圈
    arc(0, 0, dims.center_circle_radius - hwLine, 0, Math.PI * 2, 72);

    // 两端：三分线 / 罚球区 / 罚球圈 / 合理冲撞区 / 篮筐
    for (const side of ['A', 'B']) {
      const sign = side === 'A' ? 1 : -1;
      const bl = sign * hl;
      const rimX = bl - sign * dims.rim_offset_from_baseline;
      const ftX = bl - sign * (dims.free_throw_dist - hwLine);
      const R3 = dims.three_pt_radius - hwLine;
      const ylim = hw - (dims.three_pt_corner_dist + hwLine);
      const dx = Math.sqrt(Math.max(R3 ** 2 - ylim ** 2, 0));
      const arcX = rimX - sign * dx;
      // 三分直线段（端线 → 弧线切点）
      add(bl, ylim, arcX, ylim);
      add(bl, -ylim, arcX, -ylim);
      // 三分弧：朝球场中心鼓出
      const half = Math.asin(Math.min(1, ylim / R3));
      if (sign > 0) arc(rimX, 0, R3, Math.PI - half, Math.PI + half, 80);
      else arc(rimX, 0, R3, -half, half, 80);
      // 罚球区（限制区）：矩形轮廓
      const pw = dims.paint_width / 2 + hwLine;
      add(bl, -pw, ftX, -pw); add(ftX, -pw, ftX, pw); add(ftX, pw, bl, pw);
      // 罚球圈：FIBA 限制区内半边虚线，NBA 整圆实线
      const ftR = dims.free_throw_width / 2 - hwLine;
      if (dims.free_throw_dashed_inside) {
        if (sign > 0) {
          arc(ftX, 0, ftR, Math.PI / 2, Math.PI * 1.5, 48);
          dashedArc(ftX, 0, ftR, -Math.PI / 2, Math.PI / 2);
        } else {
          arc(ftX, 0, ftR, -Math.PI / 2, Math.PI / 2, 48);
          dashedArc(ftX, 0, ftR, Math.PI / 2, Math.PI * 1.5);
        }
      } else {
        arc(ftX, 0, ftR, 0, Math.PI * 2, 56);
      }
      // 合理冲撞区：半圆 + 两条到篮板前沿的短线
      const rr = dims.restricted_radius + hwLine;
      const backX = bl - sign * dims.backboard_from_baseline;
      if (sign > 0) arc(rimX, 0, rr, Math.PI / 2, Math.PI * 1.5, 36);
      else arc(rimX, 0, rr, -Math.PI / 2, Math.PI / 2, 36);
      add(backX, -rr, rimX, -rr);
      add(backX, rr, rimX, rr);

      // 篮筐（水平圆环）+ 篮板
      const rimGeo = new T.TorusGeometry(dims.rim_radius, 0.018, 6, 28);
      const rim = new T.Mesh(rimGeo, new T.MeshBasicMaterial({ color: 0xff7043 }));
      rim.rotation.x = Math.PI / 2;
      rim.position.copy(w2t(rimX, 0, dims.rim_height));
      g.add(rim);

      const bbX = bl - sign * dims.backboard_from_baseline;
      const bw = dims.backboard_width / 2;
      add(bbX, -bw, bbX, bw, dims.backboard_bottom);
      add(bbX, -bw, bbX, bw, dims.backboard_bottom + dims.backboard_height);
      add(bbX, -bw, bbX, -bw, dims.backboard_bottom);
      add(bbX, -bw, bbX, -bw, dims.backboard_bottom + dims.backboard_height);
      add(bbX, bw, bbX, bw, dims.backboard_bottom);
      add(bbX, bw, bbX, bw, dims.backboard_bottom + dims.backboard_height);
      const inner = 0.295;   // 篮板方框（内框半宽）
      add(bbX, -inner, bbX, inner, dims.rim_height + 0.15);
      add(bbX, -inner, bbX, -inner, dims.rim_height - 0.30);
      add(bbX, inner, bbX, inner, dims.rim_height - 0.30);
      add(bbX, -inner, bbX, inner, dims.rim_height - 0.30);
    }

    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(segs), 3));
    const lines = new T.LineSegments(geo, new T.LineBasicMaterial({ color: 0xf2f2f2, transparent: true, opacity: 0.92 }));
    g.add(lines);

    // 地板
    const floor = new T.Mesh(
      new T.PlaneGeometry(dims.length + 3.6, dims.width + 3.6),
      new T.MeshStandardMaterial({ color: 0x8a6642, roughness: 0.92, metalness: 0.02 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.01;
    g.add(floor);
    // 球场本体（外沿到界线外沿，含线宽）
    const courtTop = new T.Mesh(
      new T.PlaneGeometry(dims.length + (dims.line_width ?? 0.05),
                          dims.width + (dims.line_width ?? 0.05)),
      new T.MeshStandardMaterial({ color: 0xc79a62, roughness: 0.85, metalness: 0.02 }),
    );
    courtTop.rotation.x = -Math.PI / 2;
    courtTop.position.y = 0;
    g.add(courtTop);

    // 罚球区漆块（半透明）
    for (const side of ['A', 'B']) {
      const sign = side === 'A' ? 1 : -1;
      const bl = sign * hl;
      const ftX = bl - sign * (dims.free_throw_dist - (dims.line_width ?? 0.05) / 2);
      const paint = new T.Mesh(
        new T.PlaneGeometry(dims.free_throw_dist, dims.paint_width),
        new T.MeshBasicMaterial({ color: 0x2196a6, transparent: true, opacity: 0.34, depthWrite: false }),
      );
      paint.rotation.x = -Math.PI / 2;
      paint.position.copy(w2t((bl + ftX) / 2, 0, 0.004));
      g.add(paint);
    }

    updateCamera(true);
    resize();
  }

  // ── 相机 / 交互 ───────────────────────────────────────────
  function updateCamera(immediate) {
    if (!S.camera) return;
    if (immediate) { S.yaw = S.yawT; S.pitch = S.pitchT; S.dist = S.distT; }
    const r = S.dist;
    const y = Math.sin(S.pitch) * r;
    const horiz = Math.cos(S.pitch) * r;
    S.camera.position.set(Math.cos(S.yaw) * horiz, y, Math.sin(S.yaw) * horiz);
    S.camera.lookAt(0, 0.8, 0);
  }

  function bindControls() {
    const c = S.canvas;
    c.addEventListener('pointerdown', (e) => {
      S.dragging = true; S.lastX = e.clientX; S.lastY = e.clientY;
      c.setPointerCapture?.(e.pointerId);
    });
    c.addEventListener('pointermove', (e) => {
      if (!S.dragging) return;
      const dx = e.clientX - S.lastX, dy = e.clientY - S.lastY;
      S.lastX = e.clientX; S.lastY = e.clientY;
      S.yawT -= dx * 0.006;
      S.pitchT = Math.max(4 * DEG, Math.min(88 * DEG, S.pitchT + dy * 0.005));
    });
    const end = () => { S.dragging = false; };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      S.distT = Math.max(7, Math.min(90, S.distT * Math.exp(e.deltaY * 0.0012)));
    }, { passive: false });
    c.addEventListener('dblclick', () => { S.yawT = Math.PI / 2; S.pitchT = 24 * DEG; S.distT = 27; });
  }

  // ── 球员骨架 ──────────────────────────────────────────────
  function getPlayer(trackId) {
    let p = S.players[trackId];
    if (p) return p;
    const color = PALETTE[Math.abs(trackId) % PALETTE.length];
    const group = new T.Group();
    const joints = [];
    const jointGeo = new T.SphereGeometry(0.042, 8, 6);
    const matFull = new T.MeshBasicMaterial({ color });
    const matDim = new T.MeshBasicMaterial({ color, transparent: true, opacity: 0.42 });
    for (let i = 0; i < 17; i++) {
      const m = new T.Mesh(jointGeo, matFull);
      m.visible = false;
      group.add(m);
      joints.push(m);
    }
    const boneGeo = new T.BufferGeometry();
    boneGeo.setAttribute('position', new T.BufferAttribute(new Float32Array(EDGES.length * 6), 3));
    const bones = new T.LineSegments(boneGeo, new T.LineBasicMaterial({ color, linewidth: 2 }));
    bones.frustumCulled = false;
    group.add(bones);

    // 简单球员模型：先用低面数实体演示，后续可替换 GLTF 球员。
    const bodyMat = new T.MeshStandardMaterial({ color, roughness: 0.62, metalness: 0.04, transparent: true });
    const skinMat = new T.MeshStandardMaterial({ color: 0xd8a07a, roughness: 0.78, transparent: true });
    const torso = new T.Mesh(new T.CapsuleGeometry(0.15, 0.34, 4, 8), bodyMat);
    const head = new T.Mesh(new T.SphereGeometry(0.115, 10, 8), skinMat);
    const limbGeo = new T.CylinderGeometry(0.038, 0.038, 1, 7);
    const limbs = EDGES.map(() => new T.Mesh(limbGeo, bodyMat));
    group.add(torso, head, ...limbs);

    // 脚下投影环
    const ring = new T.Mesh(
      new T.RingGeometry(0.20, 0.26, 24),
      new T.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, side: T.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    group.add(ring);

    // track_id 标签
    const label = makeLabel('#' + trackId, color);
    label.position.set(0, 2.15, 0);
    group.add(label);

    S.playerGroup.add(group);
    p = { group, joints, bones, ring, label, matFull, matDim, torso, head, limbs, bodyMat, skinMat };
    S.players[trackId] = p;
    return p;
  }

  function makeLabel(text, color) {
    const cv = document.createElement('canvas');
    cv.width = 128; cv.height = 64;
    const ctx = cv.getContext('2d');
    ctx.font = 'bold 40px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 8, 128, 48);
    ctx.fillStyle = '#' + new T.Color(color).getHexString();
    ctx.fillText(text, 64, 33);
    const tex = new T.CanvasTexture(cv);
    tex.minFilter = T.LinearFilter;
    const sprite = new T.Sprite(new T.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    sprite.scale.set(1.1, 0.55, 1);
    return sprite;
  }

  function lerpPts(a, b, t) {
    if (!a || !b) return b || a;
    const out = new Array(17);
    for (let i = 0; i < 17; i++) {
      const va = a[i], vb = b[i];
      if (!va || !vb) { out[i] = vb || va; continue; }
      out[i] = [va[0] + (vb[0] - va[0]) * t, va[1] + (vb[1] - va[1]) * t, va[2] + (vb[2] - va[2]) * t];
    }
    return out;
  }

  function setSegment(mesh, a, b, visible) {
    if (!visible || !a || !b) { mesh.visible = false; return; }
    const p1 = w2t(a[0], a[1], a[2]), p2 = w2t(b[0], b[1], b[2]);
    const dir = new T.Vector3().subVectors(p2, p1);
    const len = dir.length();
    if (len < 1e-4) { mesh.visible = false; return; }
    mesh.visible = true;
    mesh.position.copy(p1).add(p2).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir.multiplyScalar(1 / len));
    mesh.scale.set(1, len, 1);
  }

  function updateMannequin(obj, kpts, vis, opacity) {
    const mid = (a, b) => {
      if (!kpts[a] || !kpts[b]) return null;
      return [(kpts[a][0] + kpts[b][0]) / 2,
              (kpts[a][1] + kpts[b][1]) / 2,
              (kpts[a][2] + kpts[b][2]) / 2];
    };
    const hip = mid(11, 12), shoulder = mid(5, 6);
    setSegment(obj.torso, hip, shoulder, !!(hip && shoulder));
    if (obj.torso.visible) obj.torso.scale.y /= 0.64;
    const head = kpts[0] || (shoulder ? [shoulder[0], shoulder[1], shoulder[2] + 0.22] : null);
    obj.head.visible = !!head;
    if (head) obj.head.position.copy(w2t(head[0], head[1], head[2] + 0.06));
    for (let i = 0; i < EDGES.length; i++) {
      const [a, b] = EDGES[i];
      setSegment(obj.limbs[i], kpts[a], kpts[b], !!(vis[a] && vis[b]));
    }
    obj.bodyMat.opacity = opacity;
    obj.skinMat.opacity = opacity;
    obj.bodyMat.depthWrite = opacity > 0.8;
    obj.skinMat.depthWrite = opacity > 0.8;
  }

  function applyPlayer(payload, kpts, t) {
    const id = payload.track_id;
    const obj = getPlayer(id);
    const vis = payload.visible || new Array(17).fill(1);
    const conf = payload.conf || [];
    const nVis = vis.reduce((a, b) => a + (b ? 1 : 0), 0);
    obj.group.visible = true;
    // 整人变暗只在"大部分关节都是推测/缺失"时启用；Pose2D.occluded 是"含任意推测点"，
    // 真实转播里眼/耳几乎每帧都缺，按它整体变暗会让所有人都显得不可信。
    const opacity = nVis < 10 ? 0.5 : 1.0;
    obj.bones.material.opacity = opacity * 0.42;
    obj.bones.material.transparent = true;
    updateMannequin(obj, kpts, vis, opacity);

    for (let i = 0; i < 17; i++) {
      const m = obj.joints[i];
      const k = kpts[i];
      if (!k || !vis[i]) { m.visible = false; continue; }
      m.visible = true;
      m.position.copy(w2t(k[0], k[1], k[2]));
      // 低置信关节缩小并变暗：前端不允许把推测点画得和真实点一样
      const lo = (conf[i] || 0) < 0.45;
      m.material = lo ? obj.matDim : obj.matFull;
      m.scale.setScalar(lo ? 0.62 : 1.0);
    }
    const arr = obj.bones.geometry.attributes.position.array;
    let n = 0;
    for (const [i, j] of EDGES) {
      const a = kpts[i], b = kpts[j];
      if (a && b && vis[i] && vis[j]) {
        const p1 = w2t(a[0], a[1], a[2]), p2 = w2t(b[0], b[1], b[2]);
        arr[n++] = p1.x; arr[n++] = p1.y; arr[n++] = p1.z;
        arr[n++] = p2.x; arr[n++] = p2.y; arr[n++] = p2.z;
      } else {
        for (let q = 0; q < 6; q++) arr[n++] = 0;
      }
    }
    obj.bones.geometry.attributes.position.needsUpdate = true;

    // 脚下投影：用踝点（不可见时退回 ground）
    const g = payload.ground || [0, 0, 0];
    obj.ring.position.copy(w2t(g[0], g[1], 0.008));
    const head = kpts[0] || w2t(0, 0, 1.8);
    obj.label.position.copy(w2t(head[0], head[1], (head[2] || 1.8) + 0.45));
    obj.label.material.opacity = opacity;
  }

  function clearPlayers() {
    for (const id of Object.keys(S.players)) {
      const p = S.players[id];
      S.playerGroup.remove(p.group);
      p.group.traverse((o) => {
        o.geometry?.dispose?.();
        if (o.material?.map) o.material.map.dispose?.();
        o.material?.dispose?.();
      });
    }
    S.players = {};
  }

  /** 用一批 3D 数据立即重画（无插值）。track 不在列表里的玩家会被移除。 */
  function updatePlayers(players) {
    if (!S.inited && !init()) return;
    const seen = new Set();
    for (const p of (players || [])) {
      seen.add(p.track_id);
      applyPlayer(p, p.kpts, 1);
    }
    for (const id of Object.keys(S.players)) {
      if (!seen.has(Number(id))) {
        S.playerGroup.remove(S.players[id].group);
        delete S.players[id];
      }
    }
    S.lastPlayers = players || [];
  }

  function blendPlayers(prev, curr, t) {
    const currById = new Map((curr.players || []).map((p) => [p.track_id, p]));
    const prevById = new Map(((prev && prev.players) || []).map((p) => [p.track_id, p]));
    const ids = new Set([...currById.keys(), ...prevById.keys()]);
    for (const id of ids) {
      const b = currById.get(id), a = prevById.get(id);
      if (!b && !a) continue;
      const base = b || a;
      const kpts = (a && b) ? lerpPts(a.kpts, b.kpts, t) : (b ? b.kpts : a.kpts);
      applyPlayer(base, kpts, t);
    }
    for (const id of Object.keys(S.players)) {
      if (!ids.has(Number(id))) {
        S.playerGroup.remove(S.players[id].group);
        delete S.players[id];
      }
    }
  }

  // ── 独立 3D 演示（无后端也能验证前端）──────────────────────
  function demoPlayers(now) {
    const t = Math.max(0, (now - S.demoStart) / 1000);
    const hl = (S.dims.length || 28) / 2, hw = (S.dims.width || 15) / 2;
    return Array.from({ length: 10 }, (_, i) => {
      const phase = i * 0.63;
      const x = Math.sin(t * 0.24 + phase) * hl * 0.58;
      const y = Math.cos(t * 0.31 + phase * 1.37) * hw * 0.56;
      const stride = Math.sin(t * 3.1 + phase) * 0.24;
      const bob = Math.abs(Math.sin(t * 3.1 + phase)) * 0.035;
      const lean = Math.sin(t * 1.6 + phase) * 0.045;
      const P = (dx, dy, z) => [x + dx + lean, y + dy, z + bob];
      const kpts = [
        P(0, 0, 1.70), P(0, -0.045, 1.72), P(0, 0.045, 1.72),
        P(0, -0.09, 1.70), P(0, 0.09, 1.70),
        P(0, -0.21, 1.45), P(0, 0.21, 1.45),
        P(-0.10 - stride * 0.45, -0.34, 1.17),
        P(-0.10 + stride * 0.45, 0.34, 1.17),
        P(-0.16 - stride * 0.70, -0.36, 0.96),
        P(-0.16 + stride * 0.70, 0.36, 0.96),
        P(0, -0.12, 0.96), P(0, 0.12, 0.96),
        P(stride * 0.55, -0.13, 0.52), P(-stride * 0.55, 0.13, 0.52),
        P(stride, -0.14, 0.09), P(-stride, 0.14, 0.09),
      ];
      return {
        track_id: i + 1, kpts,
        visible: new Array(17).fill(1), conf: new Array(17).fill(0.92),
        ground: [x, y, 0], root: [x, y, 0.96], stature: 1.85,
      };
    });
  }

  function updateDemo(now) {
    if (!S.demo) return;
    updatePlayers(demoPlayers(now));
  }

  function startDemo() {
    if (!init()) return false;
    S.demo = true;
    S.demoStart = performance.now();
    S.vid = null;
    S.buf.clear();
    S.inflight.clear();
    S.curr = S.prev = null;
    S.enabled = true;
    clearPlayers();
    resize();
    updateDemo(S.demoStart);
    setStatus('3D 演示 · 10 名球员 · 拖拽旋转 / 滚轮缩放 / 双击复位', 'info');
    window.dispatchEvent(new CustomEvent('bsk3ddemo', { detail: { active: true } }));
    return true;
  }

  function stopDemo() {
    if (!S.demo) return;
    S.demo = false;
    clearPlayers();
    S.curr = S.prev = null;
    setStatus('');
    window.dispatchEvent(new CustomEvent('bsk3ddemo', { detail: { active: false } }));
  }

  // ── 数据请求 ──────────────────────────────────────────────
  async function fetchFrame(n, attempt) {
    if (!S.enabled || !S.vid) return;
    if (S.buf.has(n) || S.inflight.has(n)) { if (S.buf.has(n)) showFrame(n); return; }
    S.inflight.add(n);
    const url = `/api/videos/${S.vid}/rt_3d?n=${n}&quality=${S.quality}&court_model=${encodeURIComponent(S.model)}&width=${S.width}`;
    try {
      const r = await fetch(url, { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      if (d.court && d.court.model) { S.dims = d.court; if (d.court.model !== S.model) buildCourt(S.model); }
      if (d.status === 'no_frame' && (attempt || 0) < 3) {
        S.inflight.delete(n);
        setTimeout(() => fetchFrame(n, (attempt || 0) + 1), 70);
        return;
      }
      S.buf.set(d.frame, d);
      // 缓冲只留最近 90 帧（约 3 秒），避免长播占内存
      if (S.buf.size > 90) {
        const keys = [...S.buf.keys()].sort((a, b) => a - b);
        for (const k of keys.slice(0, S.buf.size - 90)) S.buf.delete(k);
      }
      if (d.frame === S.frame) onData(d);
      else if (S.buf.size === 1) onData(d);
    } catch (e) {
      setStatus('3D 数据请求失败：' + e.message, 'error');
    } finally {
      S.inflight.delete(n);
    }
  }

  function onData(d) {
    const now = performance.now();
    const fallback = d.status === '2d_fallback' && (d.players || []).length > 0;
    if ((!d.calib_ok && !fallback) || (d.status !== 'ok' && !fallback)) {
      const msg = d.status === 'profile_4g'
        ? '4GB 档不启用 3D（BSK_PROFILE=4g）：只输出 2D'
        : d.status === 'no_frame'
          ? '3D 缓冲中…'
          : `标定不可用（${d.status}）：3D 暂停`;
      setStatus(msg, d.status === 'profile_4g' ? 'warn' : 'error');
      clearPlayers();
      S.curr = null; S.prev = null;
      return;
    }
    setStatus(fallback ? '标定不可用：显示识别骨架的 2D 投影' : '', fallback ? 'warn' : 'info');
    if (S.curr && S.curr.frame === d.frame) return;
    if (S.curr && d.frame < S.curr.frame) {
      // 回退/跳转到更早的帧：历史数据直接画，不做跨帧插值（否则会拿"未来"帧插值）
      S.prev = null;
      S.curr = d;
      updatePlayers(d.players);
      return;
    }
    S.prev = S.curr;
    S.curr = d;
    S.blendT0 = now;
    const gap = S.lastArrival ? now - S.lastArrival : 33;
    S.blendDur = Math.max(25, Math.min(140, gap));
    S.lastArrival = now;
    if (!S.prev) updatePlayers(d.players);
  }

  /** app.js 每显示一帧就调用：请求该帧 3D（与 2D 图并行，不阻塞播放）。 */
  function showFrame(n) {
    if (!S.enabled) return;
    S.frame = n;
    if (S.buf.has(n)) { onData(S.buf.get(n)); return; }
    fetchFrame(n, 0);
  }

  // ── 渲染循环 ──────────────────────────────────────────────
  function loop(ts) {
    requestAnimationFrame(loop);
    if (!S.enabled || !S.renderer) return;
    const now = ts || performance.now();
    // 相机阻尼
    S.yaw += (S.yawT - S.yaw) * 0.22;
    S.pitch += (S.pitchT - S.pitch) * 0.22;
    S.dist += (S.distT - S.dist) * 0.22;
    updateCamera(false);

    // 独立演示直接生成当前帧；真实数据只在已到达帧之间插值，绝不预测未来帧
    if (S.demo) {
      updateDemo(now);
    } else if (S.curr && S.prev && S.curr.frame !== S.prev.frame) {
      const t = Math.min(1, Math.max(0, (now - S.blendT0) / S.blendDur));
      blendPlayers(S.prev, S.curr, t);
    }
    S.renderer.render(S.scene, S.camera);
    S.frames++;
    if (now - S.lastFpsT > 1000) {
      S.renderFps = S.frames * 1000 / (now - S.lastFpsT);
      S.frames = 0; S.lastFpsT = now;
    }
  }

  // ── 对外 API ──────────────────────────────────────────────
  function setEnabled(on) {
    S.enabled = !!on;
    if (S.enabled) {
      if (!init()) return false;
      resize();
      if (S.demo) updateDemo(performance.now());
      else {
        setStatus('3D 缓冲中…', 'info');
        showFrame(S.frame || 1);
      }
    } else {
      setStatus('');
    }
    return S.enabled;
  }

  function toggle() { return setEnabled(!S.enabled); }

  function setVideo(vid, opts) {
    if (S.demo && vid) stopDemo();
    if (S.vid !== vid) {
      S.buf.clear();
      S.curr = null; S.prev = null;
      clearPlayers();
    }
    S.vid = vid;
    if (opts && opts.quality) S.quality = opts.quality;
    if (opts && opts.width) S.width = opts.width;
    if (opts && opts.fps) S.fps = opts.fps;
  }

  function setQuality(q) { S.quality = q || 'accurate'; }

  function setCourtModel(model) {
    if (!FALLBACK_DIMS[model] && S.model !== model) return;
    S.model = model;
    if (model === 'FIBA' || model === 'NBA' || model === '3x3') {
      S.dims = { ...FALLBACK_DIMS[model], ...(S.dims.model === model ? S.dims : {}) };
    }
    if (S.inited) buildCourt(model);
    if (S.enabled) { S.buf.clear(); fetchFrame(S.frame, 0); }
  }

  if (T) {
    S.available = true;
  } else {
    S.error = 'Three.js 未加载';
  }

  window.BSK3D = {
    get available() { return !!T; },
    get error() { return S.error; },
    get enabled() { return S.enabled; },
    get demo() { return S.demo; },
    get renderFps() { return S.renderFps; },
    init, toggle, setEnabled, showFrame, updatePlayers,
    startDemo, stopDemo,
    setVideo, setQuality, setCourtModel, setStatus,
    get state() { return S; },
  };

  document.addEventListener('DOMContentLoaded', () => {
    if (!T) {
      setStatus('3D 不可用：Three.js 未加载（gui/vendor/three.min.js）', 'error');
      console.error('[3D] Three.js missing');
    }
  });
  if (document.readyState !== 'loading' && !T) {
    console.error('[3D] Three.js missing');
  }
})();