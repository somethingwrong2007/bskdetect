// === Modern Basketball Analytics UI ===
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const LAYERS = [
  ["hud", "HUD"],
  ["boxes", "检测框"],
  ["pose", "姿态"],
  ["seg", "分割"],
  ["floor", "地板"],
  ["motion", "运动"],
  ["trail", "轨迹"],
];

const S = {
  vid: null,
  meta: null,
  nframes: 1,
  fps: 29.97,
  frame: 1,
  shots: [],
  layers: new Set(["hud", "boxes", "pose"]),
  width: 960,
  quality: 72,
  qualityMode: "accurate",
  playing: false,
  strip: null,
  reqId: 0,
  analyzed: new Set(),
  stats: { players: 0, balls: 0, fps: 0 },
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

// Go to Frame
function goto(n) {
  if (!S.vid) return;
  n = Math.max(1, Math.min(S.nframes, n));
  S.frame = n;
  
  const img = $("#frame");
  img.onload = () => refreshRtMeta(n, 0);
  img.src = `/api/videos/${S.vid}/frame.jpg?n=${n}&layers=${layerParam()}&width=${S.width}&q=${S.quality}&quality=${S.qualityMode}&realtime=1`;
  
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
  return `/api/videos/${S.vid}/rt_frame.jpg?n=${n}&layers=${layerParam()}&width=${S.width}&q=${S.quality}&quality=${S.qualityMode}&rt=${Date.now()}`;
}

function loadRtFrame(n) {
  return new Promise((resolve, reject) => {
    const img = $("#frame");
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("加载或解码失败（确认访问的是实时服务端口，如 :8000）"));
    img.src = rtFrameUrl(n);
    $("#frame-no").value = n;
    $("#scrub").value = n;
    updateFrameInfo();
  });
}

// 面板数据低频刷新（只读缓存，不触发计算）
async function refreshRtMeta(n, dtMs) {
  try {
    const r = await fetch(`/api/videos/${S.vid}/rt_meta.json?n=${n}&quality=${S.qualityMode}`);
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
  const token = S.reqId;
  const speed = parseFloat($("#speed").value) || 1;
  const step = Math.max(1, Math.round(speed));
  const interval = 1000 / ((S.fps || 29.97) * speed);
  
  $("#btn-play").innerHTML = "⏸ 停止";
  $("#btn-play").classList.add("btn-secondary");
  status(`实时分析 ${speed}x`);

  const tick = async () => {
    if (!S.playing || token !== S.reqId) return;
    const t0 = performance.now();
    try {
      await loadRtFrame(S.frame);
    } catch (e) {
      toast("实时帧加载失败: " + e.message, "error");
      stop();
      return;
    }
    if (!S.playing || token !== S.reqId) return;

    const dt = performance.now() - t0;
    S.lastFrameMs = dt;
    // 本端统计实际处理帧率
    const inst = 1000 / Math.max(dt, 1);
    S.stats.fps = S.stats.fps ? S.stats.fps * 0.8 + inst * 0.2 : inst;
    updateStats();

    if (S.frame % 5 === 0) refreshRtMeta(S.frame, dt);

    // 落后超过两个帧间隔时多跳一帧，按墙钟时间追赶而不是越播越慢
    let advance = step;
    if (dt > interval * 2.2) advance += 1;
    S.frame = Math.min(S.nframes, S.frame + advance);
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
  $("#btn-play").innerHTML = "▶ 播放";
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
  if (fpsEl) fpsEl.textContent = S.stats.fps ? S.stats.fps.toFixed(1) : "0";
}

// 用响应头里的分析摘要填充右侧面板（统计 / 帧数据 / 检测表）
function updateRealtimePanels(meta, dtMs) {
  S.stats.players = meta.players ?? 0;
  S.stats.balls = meta.balls ?? 0;
  if (dtMs) {
    const inst = 1000 / Math.max(dtMs, 1);
    S.stats.fps = S.stats.fps ? S.stats.fps * 0.7 + inst * 0.3 : inst;
  }
  updateStats();

  const metaEl = $("#meta");
  if (metaEl) {
    const rows = [
      ["帧号", `${meta.frame} / ${S.nframes}`],
      ["时间码", $("#timecode")?.textContent || "-"],
      ["质量档", meta.backend === "accurate" ? "准确 (RTMO)" : "快速 (YOLO)"],
      ["检测对象", meta.players + (meta.balls ? ` + ${meta.balls} 球` : "")],
      ["骨架", `${meta.poses} 人`],
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
      return `<tr><td>#${o.id}</td><td>${o.cat}</td><td>${(o.conf * 100).toFixed(0)}%</td><td>${box}</td></tr>`;
    }).join("");
    tablesEl.innerHTML = `<table class="det-table">
      <thead><tr><th>ID</th><th>类别</th><th>置信</th><th>框 (x0,y0,x1,y1)</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4">无检测</td></tr>'}</tbody></table>`;
  }
}

// Toggle 3D View
function toggle3D() {
  const img = $("#frame");
  const canvas = $("#view3d");
  const btn = $("#btn-toggle-3d");
  
  if (canvas.style.display === "none") {
    img.style.display = "none";
    canvas.style.display = "block";
    btn.classList.add("active");
    init3D();
  } else {
    img.style.display = "block";
    canvas.style.display = "none";
    btn.classList.remove("active");
  }
}

// Initialize 3D (placeholder)
function init3D() {
  // Would integrate with view3d.js
  console.log("3D view initialized");
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
  
  // 推理质量档：准确(RTMO) / 快速(YOLOv8n-pose)
  $("#quality-mode").onchange = () => {
    S.qualityMode = $("#quality-mode").value;
    status(S.qualityMode === "accurate" ? "准确模式" : "快速模式");
    if (!S.playing) goto(S.frame);
  };

  // 3D toggle
  $("#btn-toggle-3d").onclick = toggle3D;
  
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
    }
  };
}

// === Initialization ===
async function init() {
  buildLayerChips();
  bindEvents();
  await refreshVideos();
  
  // Auto-select first video if available
  const videos = await api("/api/videos");
  if (videos.videos && videos.videos.length > 0) {
    await selectVideo(videos.videos[0].vid);

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
