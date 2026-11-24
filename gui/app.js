const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const LAYERS = [
  ["hud", "HUD"],
  ["boxes", "Boxes"],
  ["pose", "Pose"],
  ["seg", "Segment"],
  ["floor", "Floor"],
  ["motion", "Motion"],
  ["trail", "Trail"],
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
  playing: false,
  strip: null,
  reqId: 0,
  analyzed: new Set(),
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 180)}`);
  return r.json();
}

let toastTimer = null;
function toast(msg, kind = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.style.borderColor = kind === "bad" ? "var(--error)" : kind === "ok" ? "var(--success)" : "var(--warning)";
  clearTimeout(toastTimer);
  if (msg) toastTimer = setTimeout(() => (el.textContent = ""), 4000);
}

function status(msg) {
  $("#status").textContent = msg;
}

function layerParam() {
  return [...S.layers].join(",");
}

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

async function loadVideos() {
  try {
    const data = await api("/api/videos");
    const wrap = $("#videos");
    wrap.innerHTML = "";
    for (const v of data.videos) {
      const card = document.createElement("div");
      card.className = "video-card" + (v.vid === S.vid ? " active" : "");
      card.innerHTML = `
        <div class="name">${v.name}</div>
        <div class="meta">
          <span>${v.width}x${v.height}</span>
          <span>${v.nframes} frames</span>
          <span>${v.shots} shots</span>
          <span class="status-${v.status}">${v.status}</span>
        </div>
      `;
      card.onclick = () => selectVideo(v.vid);
      wrap.appendChild(card);
    }
  } catch (e) {
    toast(`Load failed: ${e.message}`, "bad");
  }
}

async function addVideo() {
  const path = $("#path").value.trim();
  if (!path) return toast("Enter video path", "bad");
  try {
    const data = await api("/api/videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    toast(`Imported: ${data.vid}`, "ok");
    $("#path").value = "";
    await loadVideos();
    selectVideo(data.vid);
  } catch (e) {
    toast(`Import failed: ${e.message}`, "bad");
  }
}

async function selectVideo(vid) {
  S.vid = vid;
  S.analyzed.clear();
  try {
    const [meta, shots, strip] = await Promise.all([
      api(`/api/videos/${vid}`),
      api(`/api/videos/${vid}/shots`),
      api(`/api/videos/${vid}/strip.json`),
    ]);
    S.meta = meta;
    S.nframes = meta.nframes;
    S.fps = meta.fps;
    S.shots = shots.shots;
    S.strip = strip;
    $("#frame-no").max = S.nframes;
    $("#scrub").max = S.nframes;
    renderTimeline();
    drawStrip();
    goto(1);
    status(`${meta.name} · ${S.nframes} frames · ${S.shots.length} shots`);
  } catch (e) {
    toast(`Select failed: ${e.message}`, "bad");
  }
}

function renderTimeline() {
  const wrap = $("#timeline");
  wrap.innerHTML = "";
  for (const shot of S.shots) {
    const el = document.createElement("span");
    el.className = "shot";
    el.textContent = `#${shot.start_frame}-${shot.end_frame}`;
    el.onclick = () => goto(shot.start_frame);
    wrap.appendChild(el);
  }
}

function drawStrip() {
  const canvas = $("#strip");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "var(--accent)";
    ctx.lineWidth = 2;
    for (const shot of S.shots) {
      const x = (shot.start_frame / S.nframes) * canvas.width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
  };
  img.src = `/api/videos/${S.vid}/strip.png?t=${Date.now()}`;
}

async function goto(n) {
  if (!S.vid) return;
  n = Math.max(1, Math.min(S.nframes, n));
  S.frame = n;
  $("#frame-no").value = n;
  $("#scrub").value = n;
  
  const reqId = ++S.reqId;
  $("#busy").classList.remove("hidden");
  
  try {
    const img = $("#frame");
    img.src = `/api/videos/${S.vid}/frame.jpg?n=${n}&layers=${layerParam()}&width=${S.width}&q=86&realtime=0&_=${reqId}`;
    
    await loadData(n);
    
    if (reqId === S.reqId) {
      $("#busy").classList.add("hidden");
    }
  } catch (e) {
    if (reqId === S.reqId) {
      $("#busy").classList.add("hidden");
      toast(`Frame ${n} failed: ${e.message}`, "bad");
    }
  }
}

async function loadData(n) {
  try {
    const data = await api(`/api/videos/${S.vid}/frames?from=${Math.max(1, n - 40)}&to=${Math.min(S.nframes, n + 6)}&limit=80`);
    renderMeta(n, data);
    renderTables(n, data);
  } catch (e) {
    console.error("loadData", e);
  }
}

function renderMeta(n, data) {
  const wrap = $("#meta");
  const rec = data.frames.find((f) => f.frame === n);
  if (!rec) {
    wrap.innerHTML = `<div class="k">Frame</div><div class="v">${n}</div>`;
    return;
  }
  const tc = timecode(rec.pts_us);
  wrap.innerHTML = `
    <div class="k">Frame</div><div class="v">${n}</div>
    <div class="k">Time</div><div class="v">${tc}</div>
    <div class="k">Shot</div><div class="v">${rec.shot_id}</div>
    <div class="k">Objects</div><div class="v">${rec.objects.length}</div>
    <div class="k">Poses</div><div class="v">${data.poses.filter((p) => p.frame === n).length}</div>
    <div class="k">Segments</div><div class="v">${data.segments.filter((s) => s.frame === n).length}</div>
  `;
}

function renderTables(n, data) {
  const wrap = $("#tables");
  const rec = data.frames.find((f) => f.frame === n);
  const poses = data.poses.filter((p) => p.frame === n);
  const segs = data.segments.filter((s) => s.frame === n);
  
  let html = "";
  
  if (rec && rec.objects.length) {
    html += `<h3>Objects (${rec.objects.length})</h3><table><tr><th>ID</th><th>Cat</th><th>Conf</th><th>BBox</th></tr>`;
    for (const o of rec.objects.slice(0, 10)) {
      const bbox = o.bbox_px ? o.bbox_px.map((v) => v.toFixed(0)).join(",") : "-";
      html += `<tr><td>${o.track_id}</td><td>${o.category}</td><td>${o.conf.toFixed(2)}</td><td>${bbox}</td></tr>`;
    }
    html += `</table>`;
  }
  
  if (poses.length) {
    html += `<h3>Pose (${poses.length})</h3><table><tr><th>ID</th><th>Kpts</th><th>Conf</th></tr>`;
    for (const p of poses.slice(0, 10)) {
      const meanConf = p.kpts.reduce((s, k) => s + k[2], 0) / p.kpts.length;
      html += `<tr><td>${p.track_id}</td><td>${p.kpts.length}</td><td>${meanConf.toFixed(2)}</td></tr>`;
    }
    html += `</table>`;
  }
  
  if (segs.length) {
    html += `<h3>Segments (${segs.length})</h3><table><tr><th>ID</th><th>Cat</th><th>Area</th></tr>`;
    for (const s of segs.slice(0, 10)) {
      html += `<tr><td>${s.track_id}</td><td>${s.category}</td><td>${s.area_px.toFixed(0)}</td></tr>`;
    }
    html += `</table>`;
  }
  
  wrap.innerHTML = html;
}

function timecode(pts_us) {
  const total_s = pts_us / 1e6;
  const m = Math.floor(total_s / 60);
  const s = total_s % 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}

function play() {
  if (!S.vid) return toast("Select video first");
  S.playing = true;
  const speed = parseFloat($("#speed").value);
  const img = $("#frame");
  img.onload = () => {
    if (S.playing) {
      S.frame += 1;
      $("#frame-no").value = S.frame;
      $("#scrub").value = S.frame;
    }
  };
  img.src = `/api/videos/${S.vid}/stream.mjpeg?from=${S.frame}&count=${Math.min(5000, Math.max(1, S.nframes - S.frame + 1))}&layers=${layerParam()}&width=${S.width}&fps=${(S.fps * speed).toFixed(3)}&q=72`;
  $("#btn-play").textContent = "Stop";
  $("#btn-play").classList.add("primary");
  status(`Streaming from frame ${S.frame} · ${speed}x`);
}

function stop() {
  if (!S.playing) return;
  S.playing = false;
  $("#frame").src = "";
  $("#btn-play").textContent = "Play";
  $("#btn-play").classList.remove("primary");
  goto(S.frame);
}

function shotJump(dir) {
  const cur = S.shots.find((s) => S.frame >= s.start_frame && S.frame <= s.end_frame);
  if (!cur) return goto(dir > 0 ? S.nframes : 1);
  const next = S.shots[cur.id + (dir > 0 ? 1 : -1)];
  if (next) goto(dir > 0 ? next.start_frame : cur.start_frame);
}

buildLayerChips();
$("#btn-add").onclick = addVideo;
$("#btn-refresh").onclick = loadVideos;
$("#path").addEventListener("keydown", (e) => e.key === "Enter" && addVideo());
$("#btn-play").onclick = () => (S.playing ? stop() : play());
$("#speed").onchange = () => {
  if (S.playing) {
    stop();
    play();
  }
};
$$("[data-step]").forEach((b) => (b.onclick = () => goto(S.frame + parseInt(b.dataset.step, 10))));
$("#frame-no").addEventListener("change", (e) => goto(parseInt(e.target.value, 10) || 1));
$("#scrub").addEventListener("input", (e) => goto(parseInt(e.target.value, 10)));
$("#btn-shot-prev").onclick = () => shotJump(-1);
$("#btn-shot-next").onclick = () => shotJump(1);
$("#width").addEventListener("change", (e) => {
  S.width = parseInt(e.target.value, 10);
  $("#width-v").textContent = S.width;
  if (!S.playing) goto(S.frame);
});
$("#width-v").textContent = S.width;

$("#btn-auto-calib").onclick = async () => {
  if (!S.vid) return toast("No video selected", "bad");
  const btn = $("#btn-auto-calib");
  const statusEl = $("#calib-status");
  btn.disabled = true;
  btn.textContent = "Calibrating...";
  statusEl.textContent = "";
  try {
    const data = await api(`/api/videos/${S.vid}/auto_calibrate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: S.frame, court_model: "FIBA" }),
    });
    statusEl.textContent = `OK: error=${data.reproj_error_px.toFixed(1)}px, inliers=${data.keypoints_used}`;
    statusEl.className = "calib-status ok";
    toast("Calibration successful", "ok");
    goto(S.frame);
  } catch (e) {
    statusEl.textContent = `Failed: ${e.message}`;
    statusEl.className = "calib-status error";
    toast(`Calibration failed: ${e.message}`, "bad");
  } finally {
    btn.disabled = false;
    btn.textContent = "Auto Calibrate";
  }
};
$("#strip").onclick = (e) => {
  const r = e.target.getBoundingClientRect();
  goto(((e.clientX - r.left) / r.width) * S.nframes);
};

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") goto(S.frame - (e.shiftKey ? 10 : 1));
  else if (e.key === "ArrowRight") goto(S.frame + (e.shiftKey ? 10 : 1));
  else if (e.key === "[") shotJump(-1);
  else if (e.key === "]") shotJump(1);
  else if (e.key === " ") {
    e.preventDefault();
    S.playing ? stop() : play();
  } else if (e.key === "Home") goto(1);
  else if (e.key === "End") goto(S.nframes);
});

loadVideos();
