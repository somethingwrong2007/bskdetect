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
  width: 1280,
  quality: 72,
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
  img.src = `/api/videos/${S.vid}/frame.jpg?n=${n}&layers=${layerParam()}&width=${S.width}&q=${S.quality}`;
  
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

// Playback
function play() {
  if (!S.vid) {
    toast("请先选择视频", "error");
    return;
  }
  
  S.playing = true;
  const speed = parseFloat($("#speed").value);
  const img = $("#frame");
  
  img.onload = () => {
    if (S.playing) {
      S.frame += 1;
      if (S.frame > S.nframes) {
        stop();
        return;
      }
      $("#frame-no").value = S.frame;
      $("#scrub").value = S.frame;
      updateFrameInfo();
      updateStats();
    }
  };
  
  img.src = `/api/videos/${S.vid}/stream.mjpeg?from=${S.frame}&count=${Math.min(5000, Math.max(1, S.nframes - S.frame + 1))}&layers=${layerParam()}&width=${S.width}&fps=${(S.fps * speed).toFixed(3)}&q=${S.quality}`;
  
  $("#btn-play").innerHTML = "⏸ 停止";
  $("#btn-play").classList.add("btn-secondary");
  status(`播放中 ${speed}x`);
}

function stop() {
  if (!S.playing) return;
  S.playing = false;
  $("#frame").src = "";
  $("#btn-play").innerHTML = "▶ 播放";
  $("#btn-play").classList.remove("btn-secondary");
  goto(S.frame);
}

// Update Stats Display
function updateStats() {
  // This would be updated with real data from backend
  // For now, just placeholders
  const playersEl = $("#stat-players");
  const ballsEl = $("#stat-balls");
  
  if (playersEl) playersEl.textContent = S.stats.players;
  if (ballsEl) ballsEl.textContent = S.stats.balls;
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
  }
  
  console.log("BSK Detect initialized");
}

// Start when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
