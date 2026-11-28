// === Modern Basketball Analytics UI ===
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// 实时档暂不支持：seg（分割模型未接入实时）、trail（球检测模型未训练）
const LAYERS = [
  ["hud", "HUD"],
  ["boxes", "检测框"],
  ["pose", "姿态"],
  ["jersey", "号码"],
  ["floor", "地板"],
  ["motion", "运动"],
];

const S = {
  vid: null,
  meta: null,
  nframes: 1,
  fps: 29.97,
  frame: 1,
  shots: [],
  layers: new Set(["hud", "boxes", "pose", "jersey"]),
  width: 1280,
  quality: 72,
  qualityMode: "accurate",
  playWidth: 1280,  // 播放时的实际宽度（可根据帧耗时自动降档）
  playing: false,
  strip: null,
  reqId: 0,
  imgReq: 0,
  loadFails: 0,
  playT0: 0,
  playFrames: 0,
  analyzed: new Set(),
  stats: { players: 0, balls: 0, playFps: 0, procFps: 0, jerseyPublished: 0 },
};

// API Helper
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 180)}`);
  return r.json();
}

// Toast Notifications
function toast(msg, kind = "info") {
  const container = $("#toast-container");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 300);
  }, 3000);
}

// Status Display
function status(msg) {
  const el = $(".frame-info span:first-child");
  if (el) el.textContent = msg;
}

function layerParam() {
  return [...S.layers].join(",");
}

// Build Layer Chips
function buildLayerChips() {
  const wrap = $("#layers");
  wrap.innerHTML = "";
  for (const [key, label] of LAYERS) {
    const chip = document.createElement("span");
    chip.className = "chip" + (S.layers.has(key) ? " active" : "");
    chip.textContent = label;
    chip.onclick = () => {
      if (S.layers.has(key)) {
        S.layers.delete(key);
        chip.classList.remove("active");
      } else {
        S.layers.add(key);
        chip.classList.add("active");
      }
      if (!S.playing) goto(S.frame);
    };
    wrap.appendChild(chip);
  }
}

// Video List
async function refreshVideos() {
  try {
    const data = await api("/api/videos");
    const wrap = $("#videos");
    wrap.innerHTML = "";

    if (!data.videos || data.videos.length === 0) {
      wrap.innerHTML = '<div class="video-item"><div class="video-name">暂无视频</div></div>';
      return;
    }

    for (const v of data.videos) {
      const item = document.createElement("div");
      item.className = "video-item" + (v.vid === S.vid ? " active" : "");
      item.innerHTML = `
        <div class="video-name">${v.name}</div>
        <div class="video-meta">${v.nframes} frames · ${v.fps?.toFixed(1) || '?'} fps</div>
      `;
      item.onclick = () => selectVideo(v.vid);
      wrap.appendChild(item);
    }
  } catch (e) {
    toast("加载视频列表失败: " + e.message, "error");
  }
}

// Select Video
async function selectVideo(vid) {
  S.vid = vid;
  S.playWidth = Math.min(S.width, 960);
  if (window.BSK3D) window.BSK3D.setVideo(vid, { quality: S.qualityMode, width: S.playWidth });
  try {
    const meta = await api(`/api/videos/${vid}`);
    S.meta = meta;
    S.nframes = meta.nframes || 1;
    S.fps = meta.fps || 29.97;
    S.frame = 1;

    // Update UI
    $("#scrub").max = S.nframes;
    $("#scrub").value = 1;
    $("#frame-no").value = 1;
    $("#frame-no").max = S.nframes;
    $("#width-v").textContent = S.width;

    // Load shots
    const shotData = await api(`/api/videos/${vid}/shots`);
    S.shots = shotData.shots || [];

    // Update stats
    updateStats();

    // Load first frame
    goto(1);

    // Refresh video list to show active
    refreshVideos();

    toast(`已选择: ${meta.name}`, "success");
  } catch (e) {
    toast("加载视频失败: " + e.message, "error");
  }
}

// Add Video
async function addVideo() {
  const path = $("#path").value.trim();
  if (!path) {
    toast("请输入视频路径", "error");
    return;
  }

  try {
    await api("/api/videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    toast("视频添加成功，正在索引...", "success");
    $("#path").value = "";
    await refreshVideos();
  } catch (e) {
    toast("添加失败: " + e.message, "error");
  }
}

// 离屏解码成功后才替换可见画面，失败时保留上一帧，避免请求竞态出现黑帧。
function replaceDecodedFrame(loader) {
  const current = $("#frame");
  if (!current) return;
  loader.id = current.id;
  loader.alt = current.alt;
  loader.draggable = current.draggable;
  loader.className = current.className;
  loader.style.cssText = current.style.cssText;
  current.replaceWith(loader);
}

function loadFrameImage(url, retries = 1) {
  return new Promise((resolve, reject) => {
    const req = ++S.imgReq;
    const loader = new Image();
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      loader.onload = null;
      loader.onerror = null;
    };
    const finish = (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      err ? reject(err) : resolve();
    };
    const timer = setTimeout(() => {
      if (req === S.imgReq) finish(new Error("帧请求超时"));
    }, 6000);
    loader.onload = () => {
      if (req !== S.imgReq) return;
      replaceDecodedFrame(loader);
      finish(null);
    };
    loader.onerror = () => {
      if (req !== S.imgReq) return;
      if (retries > 0) {
        cleanup();
        setTimeout(() => {
          if (req !== S.imgReq) {
            finish(new Error("帧请求已被替换"));
            return;
          }
          loadFrameImage(url, retries - 1).then(resolve, reject);
        }, 120);
        return;
      }
      finish(new Error("加载或解码失败（确认访问的是实时服务端口，如 :8000）"));
    };
    loader.src = url;
  });
}

// Go to Frame
function goto(n) {
  if (!S.vid) return;
  n = Math.max(1, Math.min(S.nframes, n));
  S.frame = n;

  const url = rtFrameUrl(n);
  loadFrameImage(url, 1).then(() => {
    refreshRtMeta(n, 0);
    if (window.BSK3D && window.BSK3D.enabled) {
      showSplit3D(true);
      window.BSK3D.showFrame(n);
    }
  }).catch((e) => status(`帧 ${n} 加载失败：${e.message}`));

  $("#frame-no").value = n;
  $("#scrub").value = n;
  updateFrameInfo();
}

// Update Frame Info Display
function updateFrameInfo() {
  const counter = $("#frame-counter");
  const timecode = $("#timecode");
  if (counter) {
    counter.textContent = `Frame: ${S.frame} / ${S.nframes}`;
  }
  if (timecode) {
    const totalSeconds = S.frame / (S.fps || 29.97);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    timecode.textContent = `${String(mins).padStart(2, '0')}:${secs.toFixed(3).padStart(6, '0')}`;
  }
}

// Realtime frame loader: 直接给 <img> 设 URL（浏览器原生解码，比 blob 少一次拷贝）
function rtFrameUrl(n) {
  // 3D 视图打开时让这一帧顺带做 3D 提升（服务端 8GB 档才生效），/rt_3d 直接读缓存
  const want3d = window.BSK3D && window.BSK3D.enabled ? "&enable_3d=1" : "";
  return `/api/videos/${S.vid}/rt_frame.jpg?n=${n}&layers=${layerParam()}&width=${S.playWidth}&q=${S.quality}&quality=${S.qualityMode}&rt=${Date.now()}${want3d}`;
}

function loadRtFrame(n, retries = 1) {
  $("#frame-no").value = n;
  $("#scrub").value = n;
  updateFrameInfo();
  return loadFrameImage(rtFrameUrl(n), retries).then(() => {
    if (window.BSK3D && window.BSK3D.enabled) window.BSK3D.showFrame(n);
  });
}

// 面板数据低频刷新（只读缓存，不触发计算）
async function refreshRtMeta(n, dtMs) {
  try {
    const r = await fetch(`/api/videos/${S.vid}/rt_meta.json?n=${n}&quality=${S.qualityMode}&width=${S.playWidth}`);
    if (!r.ok) return;
    updateRealtimePanels(await r.json(), dtMs);
  } catch (_) { /* 面板刷新失败不影响播放 */ }
}

// Playback（逐帧拉取实时分析结果，播到哪算到哪）
function play() {
  if (!S.vid) {
    toast("请先选择视频", "error");
    return;
  }

  S.playing = true;
  S.reqId += 1;
  // 播放默认 960：1280 的服务端耗时虽可接受，但浏览器 JPEG 解码会拖到 ~19fps。
  S.playWidth = Math.min(S.width, 960);
  S.loadFails = 0;
  S.playT0 = performance.now();
  S.playFrames = 0;
  S.stats.playFps = 0;
  const token = S.reqId;
  const speed = parseFloat($("#speed").value) || 1;
  const step = Math.max(1, Math.round(speed));
  const interval = 1000 / ((S.fps || 29.97) * speed);

  $("#btn-play").innerHTML = "停止";
  $("#btn-play").classList.add("btn-secondary");
  status(`实时分析 ${speed}x`);

  const tick = async () => {
    if (!S.playing || token !== S.reqId) return;
    const t0 = performance.now();
    try {
      await loadRtFrame(S.frame);
      S.loadFails = 0;
    } catch (e) {
      // 单帧游标竞争/解码抖动不要直接停播：同帧再试，连续失败才降级跳过。
      S.loadFails += 1;
      if (S.loadFails <= 2) {
        status(`帧 ${S.frame} 暂时未取到，重试 ${S.loadFails}/2...`);
        setTimeout(tick, 180);
        return;
      }
      S.loadFails = 0;
      S.frame = Math.min(S.nframes, S.frame + 1);
      if (S.frame >= S.nframes) {
        toast("实时帧加载失败: " + e.message, "error");
        stop();
        return;
      }
      status(`帧 ${S.frame - 1} 解码失败，已跳过`);
      setTimeout(tick, 80);
      return;
    }
    if (!S.playing || token !== S.reqId) return;

    const dt = performance.now() - t0;
    S.lastFrameMs = dt;
    S.stats.procFps = 1000 / Math.max(dt, 1);
    // 播放 FPS 按墙钟计算，不能用处理耗时；处理快不等于视频快放。
    S.playFrames += 1;
    const wall = Math.max(0.001, (performance.now() - S.playT0) / 1000);
    S.stats.playFps = S.playFrames / wall;
    updateStats();

    if (S.frame % 5 === 0) refreshRtMeta(S.frame, dt);

    // 自适应降档：持续达不到 30fps 时降低播放宽度（每档 160，最低 800）
    if (dt > 36) S.noFpsCount = (S.noFpsCount || 0) + 1; else S.noFpsCount = 0;
    if (S.noFpsCount >= 12 && S.playWidth > 800) {
      S.playWidth = Math.max(800, S.playWidth - 160);
    if (window.BSK3D) window.BSK3D.setVideo(S.vid, { quality: S.qualityMode, fps: S.fps, width: S.playWidth });
      S.noFpsCount = 0;
      toast(`帧耗时偏高，已自动降档到 ${S.playWidth} 宽以保持流畅`, "info");
      status(`实时分析 ${speed}x · ${S.playWidth}宽`);
    }

    // 严格按源视频时间轴推进：1x 每次只前进一帧，不额外追帧加速。
    S.frame = Math.min(S.nframes, S.frame + step);
    if (S.frame >= S.nframes) {
      stop();
      return;
    }
    const wait = Math.max(0, interval - dt);
    setTimeout(tick, wait);
  };
  tick();
}

function stop() {
  if (!S.playing) return;
  S.playing = false;
  S.reqId += 1;
  $("#btn-play").innerHTML = "播放";
  $("#btn-play").classList.remove("btn-secondary");
  goto(S.frame);
}

// Update Stats Display
function updateStats() {
  const playersEl = $("#stat-players");
  const ballsEl = $("#stat-balls");
  const fpsEl = $("#stat-fps");
  if (playersEl) playersEl.textContent = S.stats.players;
  if (ballsEl) ballsEl.textContent = S.stats.balls;
  if (fpsEl) {
    const fps = S.playing ? S.stats.playFps : (S.fps || 0);
    fpsEl.textContent = fps ? fps.toFixed(1) : "0";
  }
}

// 用响应头里的分析摘要填充右侧面板（统计 / 帧数据 / 检测表）
function updateRealtimePanels(meta, dtMs) {
  S.stats.players = meta.players ?? 0;
  S.stats.balls = meta.balls ?? 0;
  const jerseyInfo = meta.jersey || {};
  if ((jerseyInfo.published || 0) > S.stats.jerseyPublished) {
    const nums = Object.values(jerseyInfo.numbers || {}).join(", ");
    toast(`号码已发布${nums ? `：#${nums}` : ""}`, "success");
  }
  S.stats.jerseyPublished = jerseyInfo.published || 0;
  if (dtMs) {
    const inst = 1000 / Math.max(dtMs, 1);
    S.stats.procFps = S.stats.procFps ? S.stats.procFps * 0.7 + inst * 0.3 : inst;
  }
  updateStats();

  const metaEl = $("#meta");
  if (metaEl) {
    const d3 = meta["3d"] || {};
    const c2 = meta.court2d || {};
    const rows = [
      ["帧号", `${meta.frame} / ${S.nframes}`],
      ["时间码", $("#timecode")?.textContent || "-"],
      ["质量档", "RTMO + RF-DETR"],
      ["检测对象", meta.players + (meta.balls ? ` + ${meta.balls} 球` : "")],
      ["骨架", `${meta.poses} 人`],
      ["号码", `${jerseyInfo.published || 0} 已发布 / ${jerseyInfo.crops || 0} 采样`],
      ["2D坐标", c2.ok ? `${c2.n} 人` : `关闭 (${c2.reason || "no_calib"})`],
      ["3D", d3.ok ? `启用 ${d3.n} 人` : `关闭 (${d3.why || "disabled"})`],
      ["运行档", meta.profile || "-"],
      ["处理延迟", dtMs ? `${dtMs.toFixed(0)} ms` : "-"],
    ];
    metaEl.innerHTML = rows.map(([k, v]) =>
      `<div class="kv-item"><span class="kv-key">${k}</span><span class="kv-value">${v}</span></div>`
    ).join("");
  }

  const tablesEl = $("#tables");
  if (tablesEl && Array.isArray(meta.objects)) {
    const rows = meta.objects.map((o) => {
      const box = o.box.map((v) => Math.round(v)).join(",");
      const pos = (o.x !== undefined && o.y !== undefined)
        ? `${o.x.toFixed(2)}, ${o.y.toFixed(2)}` : "—";
      const jersey = o.jersey !== undefined ? `#${o.jersey}` : "—";
      const identity = o.jersey !== undefined ? `#${o.jersey}` : `#${o.id}`;
      return `<tr><td>${identity}</td><td>${o.cat}</td><td>${jersey}</td><td>${(o.conf * 100).toFixed(0)}%</td><td>${box}</td><td>${pos}</td></tr>`;
    }).join("");
    tablesEl.innerHTML = `<table class="det-table">
      <thead><tr><th>ID/号码</th><th>类别</th><th>号码</th><th>置信</th><th>框 (x0,y0,x1,y1)</th><th>场地 x,y (m)</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="6">无检测</td></tr>'}</tbody></table>`;
  }
}

function showSplit3D(on) {
  const host = document.querySelector(".viewer-container");
  host?.classList.toggle("split-3d", on);
  $("#frame").style.display = "block";
  $("#view3d").style.display = on ? "block" : "none";
}

// Toggle 3D View：2D 与真实 3D 骨架并排，播放不中断。
function toggle3D() {
  const btn = $("#btn-toggle-3d");
  const sel = $("#court-model-select");
  if (!window.BSK3D) {
    toast("3D 模块未加载（gui/view3d.js / three.min.js 缺失）", "error");
    return;
  }
  if (!window.BSK3D.available) {
    toast("3D 不可用：" + (window.BSK3D.error || "缺少 Three.js"), "error");
    return;
  }
  const on = !window.BSK3D.enabled;
  if (!on && window.BSK3D.demo) window.BSK3D.stopDemo();
  // 先切显示，再让 3D 初始化（初始化要读 canvas 的 clientWidth/Height）
  showSplit3D(on);
  btn.classList.toggle("active", on);
  btn.textContent = on ? "2D" : "3D";
  if (sel) sel.classList.toggle("visible", on);

  window.BSK3D.setVideo(S.vid, { quality: S.qualityMode, fps: S.fps, width: S.playWidth });
  window.BSK3D.setEnabled(on);

  if (on) {
    // 同步到当前帧；播放中继续由 loadRtFrame 逐帧驱动
    window.BSK3D.showFrame(S.frame);
    status("2D / 3D 并排：真实识别骨架（拖拽旋转 / 滚轮缩放 / 双击复位）");
  } else {
    // 恢复 2D 显示当前帧（不重启播放）
    if (!S.playing) goto(S.frame);
    else status(`实时分析 ${$("#speed").value || 1}x`);
  }
}

function startDemo3D() {
  if (!window.BSK3D?.available) {
    toast("3D 模块不可用", "error");
    return;
  }
  if (S.vid) {
    if (!window.BSK3D.enabled) toggle3D();
    status("3D 演示已接入真实识别骨架");
    return;
  }
  if (S.playing) stop();
  const img = $("#frame"), canvas = $("#view3d");
  img.style.display = "none";
  canvas.style.display = "block";
  $("#btn-toggle-3d").classList.add("active");
  $("#btn-toggle-3d").textContent = "2D";
  $("#court-model-select").classList.add("visible");
  document.querySelector(".viewer-container").classList.add("demo-mode");
  window.BSK3D.startDemo();
  status("3D 演示：拖拽旋转 / 滚轮缩放 / 双击复位");
}

function stopDemo3D() {
  if (!window.BSK3D) return;
  if (S.vid) {
    if (window.BSK3D.enabled) toggle3D();
    return;
  }
  window.BSK3D.stopDemo();
  window.BSK3D.setEnabled(false);
  $("#frame").style.display = "block";
  $("#view3d").style.display = "none";
  $("#btn-toggle-3d").classList.remove("active");
  $("#btn-toggle-3d").textContent = "3D";
  $("#court-model-select").classList.remove("visible");
  document.querySelector(".viewer-container").classList.remove("demo-mode");
  if (!S.playing) goto(S.frame);
}

// === Event Bindings ===
function bindEvents() {
  // Header
  $("#btn-add").onclick = addVideo;
  $("#btn-refresh").onclick = refreshVideos;
  $("#path").addEventListener("keydown", (e) => {
    if (e.key === "Enter") addVideo();
  });

  // Playback
  $("#btn-play").onclick = () => (S.playing ? stop() : play());

  // Frame navigation
  $$("[data-step]").forEach((btn) => {
    btn.onclick = () => goto(S.frame + parseInt(btn.dataset.step));
  });

  $("#frame-no").onchange = () => goto(parseInt($("#frame-no").value) || 1);
  $("#scrub").oninput = () => {
    S.frame = parseInt($("#scrub").value);
    $("#frame-no").value = S.frame;
    updateFrameInfo();
  };
  $("#scrub").onchange = () => goto(S.frame);

  // Shot navigation
  $("#btn-shot-prev").onclick = () => {
    if (!S.shots.length) return;
    const cur = S.shots.find((s) => S.frame >= s.start_frame && S.frame <= s.end_frame);
    const idx = cur ? S.shots.indexOf(cur) : S.shots.length;
    goto(idx > 0 ? S.shots[idx - 1].start_frame : 1);
  };

  $("#btn-shot-next").onclick = () => {
    if (!S.shots.length) return;
    const cur = S.shots.find((s) => S.frame >= s.start_frame && S.frame <= s.end_frame);
    const idx = cur ? S.shots.indexOf(cur) : -1;
    goto(idx < S.shots.length - 1 ? S.shots[idx + 1].start_frame : S.nframes);
  };

  // Controls
  $("#width").oninput = () => {
    S.width = parseInt($("#width").value);
    $("#width-v").textContent = S.width;
    if (!S.playing) goto(S.frame);
  };

  $("#quality").oninput = () => {
    S.quality = parseInt($("#quality").value);
    $("#quality-v").textContent = S.quality;
  };

  // 推理质量档：统一使用 RTMO + RF-DETR，fast 仅作为旧 API 兼容别名。
  $("#quality-mode").onchange = () => {
    S.qualityMode = $("#quality-mode").value;
    if (window.BSK3D) window.BSK3D.setQuality(S.qualityMode);
    status("RTMO + RF-DETR 模式");
    if (!S.playing) goto(S.frame);
  };

  // 3D toggle
  $("#btn-toggle-3d").onclick = toggle3D;
  $("#btn-demo-3d").onclick = () => window.BSK3D?.demo ? stopDemo3D() : startDemo3D();
  window.addEventListener("bsk3ddemo", (e) => {
    const active = !!e.detail?.active;
    $("#btn-demo-3d").textContent = active ? "退出演示" : "3D 演示";
    $("#view3d-demo-info").classList.toggle("visible", active);
    document.querySelector(".viewer-container").classList.toggle("demo-mode", active);
  });

  // 球场型号（只影响 3D 视图画出的地面几何；标定 H 的型号以服务端为准）
  const courtSel = $("#court-model-select");
  if (courtSel) {
    courtSel.onchange = () => {
      const model = courtSel.value;
      if (window.BSK3D) window.BSK3D.setCourtModel(model);
      toast(`3D 球场：${model}`, "info");
    };
  }

  // Keyboard shortcuts
  document.onkeydown = (e) => {
    if (e.target.tagName === "INPUT") return;

    switch (e.key) {
      case "ArrowLeft":
        goto(S.frame - (e.shiftKey ? 10 : 1));
        break;
      case "ArrowRight":
        goto(S.frame + (e.shiftKey ? 10 : 1));
        break;
      case " ":
        e.preventDefault();
        S.playing ? stop() : play();
        break;
      case "[":
        $("#btn-shot-prev").click();
        break;
      case "]":
        $("#btn-shot-next").click();
        break;
      case "v":
      case "V":
        toggle3D();
        break;
    }
  };
}

// === Initialization ===
async function init() {
  buildLayerChips();
  bindEvents();
  const initialQs = new URLSearchParams(location.search);
  if (initialQs.get("demo3d") === "1") {
    const model = initialQs.get("court");
    if (model && window.BSK3D) window.BSK3D.setCourtModel(model);
    startDemo3D();
    console.log("BSK Detect 3D demo initialized");
    return;
  }
  await refreshVideos();

  // Auto-select first video if available
  const videos = await api("/api/videos");
  if (videos.videos && videos.videos.length > 0) {
    await selectVideo(videos.videos[0].vid);

    // 可分享/验收用的深链：?frame=2060&view=3d&court=FIBA
    const qs = new URLSearchParams(location.search);
    const wantFrame = parseInt(qs.get("frame") || "0", 10);
    const want3d = (qs.get("view") || "").toLowerCase() === "3d";
    const wantCourt = qs.get("court");
    if (wantFrame > 0) goto(wantFrame);
    if (wantCourt) {
      const sel = $("#court-model-select");
      if (sel) sel.value = wantCourt;
      if (window.BSK3D) window.BSK3D.setCourtModel(wantCourt);
    }
    if (want3d) toggle3D();

    // 探测实时接口：连到旧服务实例时给出明确提示
    try {
      const probe = await fetch(`/api/videos/${S.vid}/rt_frame.jpg?n=1&layers=hud&width=480&q=40`);
      if (!probe.ok) {
        toast(`当前服务缺少实时接口（HTTP ${probe.status}），请访问运行实时管线的新服务端口（如 :8000）`, "error");
      }
    } catch (_) { /* 探测失败不阻塞界面 */ }
  }

  console.log("BSK Detect initialized");
}

// Start when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
