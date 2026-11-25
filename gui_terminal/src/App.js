/* 本文件由 gui_terminal/build.py 从同目录 App.jsx 预编译生成，请勿手改。
   编译器：@babel/standalone 7.26.7 (MIT)，preset=react
   改界面只改 App.jsx，然后跑 python gui_terminal/build.py 重新生成。 */

/* BSK Detect · 实时篮球视频分析调试台（React + Tailwind）
   界面：浅色冷调机能风，全站直角；数据来自本机 FastAPI（bsk/server.py）。
   实时管线不可用时自动降级到离线渲染 frame.jpg，并在界面上说明原因。 */

const {
  useState,
  useEffect,
  useRef,
  useCallback
} = React;

/* ---------------- 常量与工具 ---------------- */

// 实时档支持的图层；seg/trail 需要额外模型，未接入时后端会自己忽略
const LAYERS = [{
  key: "hud",
  label: "HUD"
}, {
  key: "boxes",
  label: "检测框"
}, {
  key: "pose",
  label: "姿态"
}, {
  key: "floor",
  label: "地板"
}, {
  key: "motion",
  label: "运动"
}];
const COURT_MODELS = ["FIBA", "NBA", "3x3"];
const SPEEDS = ["0.25", "0.5", "1", "2", "4"];
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 160)}`);
  return r.json();
}
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const pad = (n, w) => String(n).padStart(w, "0");
function timecode(frame, fps) {
  const t = frame / (fps || 29.97);
  return pad(Math.floor(t / 60), 2) + ":" + (t % 60).toFixed(3).padStart(6, "0");
}

/* ---------------- 通用小组件 ---------------- */

function SectionTitle({
  index,
  zh,
  en,
  tail
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 mb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-mono text-[10px] leading-none text-[#2FA8D8]"
  }, index), /*#__PURE__*/React.createElement("span", {
    className: "h-3 w-[3px] bg-[#2FA8D8]"
  }), /*#__PURE__*/React.createElement("span", {
    className: "font-display text-[13px] font-semibold uppercase tracking-[0.16em] text-[#1B2026]"
  }, zh), /*#__PURE__*/React.createElement("span", {
    className: "font-mono text-[9px] uppercase tracking-[0.18em] text-[#8A949F]"
  }, en), /*#__PURE__*/React.createElement("span", {
    className: "flex-1 h-px bg-[rgba(122,148,172,0.38)]"
  }), tail ? /*#__PURE__*/React.createElement("span", {
    className: "font-mono text-[10px] text-[#4C555F]"
  }, tail) : null);
}
function Corner({
  className
}) {
  return /*#__PURE__*/React.createElement("span", {
    "aria-hidden": "true",
    className: "pointer-events-none absolute h-3 w-3 border-[#2FA8D8]/70 " + className
  });
}

/* 画面四角的三角标记 + 沿边工程断线：贴边框走，不压内容 */
function FrameMarks() {
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Corner, {
    className: "left-0 top-0 border-l border-t"
  }), /*#__PURE__*/React.createElement(Corner, {
    className: "right-0 top-0 border-r border-t"
  }), /*#__PURE__*/React.createElement(Corner, {
    className: "left-0 bottom-0 border-l border-b"
  }), /*#__PURE__*/React.createElement(Corner, {
    className: "right-0 bottom-0 border-r border-b"
  }), /*#__PURE__*/React.createElement("span", {
    "aria-hidden": "true",
    className: "pointer-events-none absolute left-6 right-6 top-[3px] h-[3px] tick-strip opacity-50"
  }), /*#__PURE__*/React.createElement("span", {
    "aria-hidden": "true",
    className: "pointer-events-none absolute left-6 right-6 bottom-[3px] h-[3px] tick-strip opacity-35"
  }), /*#__PURE__*/React.createElement("span", {
    "aria-hidden": "true",
    className: "pointer-events-none absolute left-[3px] top-8 bottom-8 w-[3px] tick-strip-major opacity-30 [writing-mode:vertical-rl]"
  }));
}
function Divider({
  vertical
}) {
  return vertical ? /*#__PURE__*/React.createElement("span", {
    className: "mx-1 h-5 w-px bg-[rgba(122,148,172,0.5)]"
  }) : /*#__PURE__*/React.createElement("span", {
    className: "h-px w-full bg-[rgba(122,148,172,0.32)]"
  });
}

/* 无画面时的球场线稿底，有帧图后由图片盖住 */
function CourtWireframe() {
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 640 360",
    className: "absolute inset-0 h-full w-full",
    preserveAspectRatio: "xMidYMid slice",
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement("g", {
    fill: "none",
    stroke: "#2FA8D8",
    strokeOpacity: "0.16",
    strokeWidth: "1"
  }, /*#__PURE__*/React.createElement("rect", {
    x: "40",
    y: "30",
    width: "560",
    height: "300"
  }), /*#__PURE__*/React.createElement("line", {
    x1: "320",
    y1: "30",
    x2: "320",
    y2: "330"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "320",
    cy: "180",
    r: "46"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "40",
    y: "128",
    width: "120",
    height: "104"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M160 150 a34 34 0 0 1 0 60"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "90",
    cy: "180",
    r: "8"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "480",
    y: "128",
    width: "120",
    height: "104"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M480 150 a34 34 0 0 0 0 60"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "550",
    cy: "180",
    r: "8"
  })), /*#__PURE__*/React.createElement("g", {
    stroke: "#62CDEC",
    strokeOpacity: "0.22",
    strokeWidth: "1",
    strokeDasharray: "4 6"
  }, /*#__PURE__*/React.createElement("line", {
    x1: "40",
    y1: "180",
    x2: "600",
    y2: "180"
  })));
}
function Toasts({
  items
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "pointer-events-none fixed bottom-4 right-4 z-50 flex w-[340px] flex-col gap-1"
  }, items.map(t => /*#__PURE__*/React.createElement("div", {
    key: t.id,
    className: "glass border px-3 py-2 font-mono text-[11px] leading-relaxed " + (t.kind === "error" ? "border-[#B3261E]/70 text-[#7C1D17]" : t.kind === "warn" ? "border-[#C8952B]/70 text-[#7A5610]" : "border-[rgba(122,148,172,0.55)] text-[#1B2026]")
  }, t.msg)));
}

/* ---------------- 1 · 顶部导航 ---------------- */

function TopNav({
  onImport,
  onRefresh,
  busy
}) {
  const [path, setPath] = useState("");
  return /*#__PURE__*/React.createElement("header", {
    className: "relative z-10 flex h-14 items-center gap-6 border-b border-[rgba(122,148,172,0.45)] bg-white/70 px-4 backdrop-blur-md"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-baseline gap-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "h-6 w-[5px] bg-[#2FA8D8]"
  }), /*#__PURE__*/React.createElement("h1", {
    className: "font-display text-[26px] font-bold uppercase leading-none tracking-[0.02em] text-[#1B2026]"
  }, "BSK Detect"), /*#__PURE__*/React.createElement("span", {
    className: "font-mono text-[10px] uppercase tracking-[0.2em] text-[#8A949F]"
  }, "Real-time Basketball Analytics")), /*#__PURE__*/React.createElement("div", {
    className: "ml-auto flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("input", {
    value: path,
    onChange: e => setPath(e.target.value),
    onKeyDown: e => {
      if (e.key === "Enter") onImport(path, setPath);
    },
    spellCheck: "false",
    className: "h-8 w-[360px] border border-[rgba(122,148,172,0.5)] bg-white/70 px-2 font-mono text-[12px] text-[#1B2026] outline-none focus:border-[#2FA8D8]",
    placeholder: "\u89C6\u9891\u8DEF\u5F84 (e.g. E:\\videos\\game.mp4)"
  }), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => onImport(path, setPath),
    className: "btn-ice h-8 px-4 font-display text-[13px] font-semibold uppercase tracking-[0.1em] disabled:opacity-50"
  }, "\u5BFC\u5165\u89C6\u9891"), /*#__PURE__*/React.createElement("button", {
    title: "\u5237\u65B0\u89C6\u9891\u5217\u8868",
    onClick: onRefresh,
    className: "btn-steel h-8 w-8",
    "aria-label": "\u5237\u65B0"
  }, /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 16 16",
    className: "mx-auto h-3.5 w-3.5",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.6"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M13 8a5 5 0 1 1-1.6-3.7M13 2v3h-3"
  })))));
}

/* ---------------- 2 · 左侧控制面板 ---------------- */

function VideoList({
  videos,
  active,
  setActive
}) {
  return /*#__PURE__*/React.createElement("section", {
    className: "mb-4"
  }, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "01",
    zh: "\u89C6\u9891\u5E93",
    en: "library",
    tail: videos.length + " CLIPS"
  }), /*#__PURE__*/React.createElement("div", {
    className: "max-h-[236px] space-y-px overflow-y-auto pr-px"
  }, videos.length === 0 ? /*#__PURE__*/React.createElement("div", {
    className: "px-2.5 py-2 font-mono text-[11px] text-[#8A949F]"
  }, "\u6682\u65E0\u7D20\u6750 \xB7 \u5728\u4E0A\u65B9\u586B\u8DEF\u5F84\u5BFC\u5165") : null, videos.map(v => {
    const on = v.vid === active;
    const ready = v.status === "ready";
    return /*#__PURE__*/React.createElement("button", {
      key: v.vid,
      onClick: () => setActive(v.vid),
      className: "block w-full border-l-2 px-2.5 py-2 text-left transition " + (on ? "border-l-[#2FA8D8] bg-[rgba(47,168,216,0.13)]" : "border-l-[rgba(122,148,172,0.4)] hover:border-l-[#2FA8D8]/70 hover:bg-white/60")
    }, /*#__PURE__*/React.createElement("div", {
      className: "truncate text-[12px] leading-snug " + (on ? "font-semibold text-[#1B2026]" : "text-[#4C555F]")
    }, v.name), /*#__PURE__*/React.createElement("div", {
      className: "mt-1 flex items-center gap-3 font-mono text-[10px] tracking-[0.04em] text-[#8A949F]"
    }, /*#__PURE__*/React.createElement("span", null, (v.nframes || 0).toLocaleString(), " FR"), /*#__PURE__*/React.createElement("span", null, v.fps ? v.fps.toFixed(2) : "--", " FPS"), /*#__PURE__*/React.createElement("span", {
      className: ready ? "text-[#2FA8D8]" : "text-[#C8952B]"
    }, ready ? "READY" : (v.status || "INDEX").toUpperCase().slice(0, 6))));
  })));
}
function LayerGroup({
  on,
  toggle
}) {
  return /*#__PURE__*/React.createElement("section", {
    className: "mb-4"
  }, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "02",
    zh: "\u56FE\u5C42\u63A7\u5236",
    en: "layers",
    tail: [...on].length + "/" + LAYERS.length
  }), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-2 gap-px"
  }, LAYERS.map(l => {
    const active = on.has(l.key);
    return /*#__PURE__*/React.createElement("button", {
      key: l.key,
      onClick: () => toggle(l.key),
      className: "flex h-8 items-center justify-between px-2 text-[12px] transition " + (active ? "btn-ice" : "btn-quiet")
    }, /*#__PURE__*/React.createElement("span", {
      className: "font-display font-semibold uppercase tracking-[0.08em]"
    }, l.label), /*#__PURE__*/React.createElement("span", {
      className: "font-mono text-[9px] " + (active ? "text-[#08202C]/70" : "text-[#8A949F]")
    }, active ? "ON" : "OFF"));
  })));
}
function SliderRow({
  label,
  unit,
  value,
  min,
  max,
  step,
  onChange
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "mb-3 last:mb-0"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mb-1 flex items-baseline justify-between"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] uppercase tracking-[0.14em] text-[#4C555F]"
  }, label), /*#__PURE__*/React.createElement("span", {
    className: "font-mono text-[13px] font-medium tabular-nums text-[#1B2026]"
  }, value, /*#__PURE__*/React.createElement("span", {
    className: "ml-0.5 text-[10px] text-[#8A949F]"
  }, unit))), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: min,
    max: max,
    step: step,
    value: value,
    onChange: e => onChange(Number(e.target.value))
  }), /*#__PURE__*/React.createElement("div", {
    className: "mt-0.5 flex justify-between font-mono text-[9px] text-[#8A949F]"
  }, /*#__PURE__*/React.createElement("span", null, min), /*#__PURE__*/React.createElement("span", null, max)));
}
function DisplaySettings({
  width,
  setWidth,
  quality,
  setQuality
}) {
  return /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "03",
    zh: "\u663E\u793A\u8BBE\u7F6E",
    en: "render"
  }), /*#__PURE__*/React.createElement(SliderRow, {
    label: "\u5BBD\u5EA6",
    unit: "px",
    value: width,
    min: 480,
    max: 1920,
    step: 80,
    onChange: setWidth
  }), /*#__PURE__*/React.createElement(SliderRow, {
    label: "\u8D28\u91CF",
    unit: "q",
    value: quality,
    min: 40,
    max: 95,
    step: 1,
    onChange: setQuality
  }), /*#__PURE__*/React.createElement("div", {
    className: "mt-2 font-mono text-[9px] leading-relaxed text-[#8A949F]"
  }, "\u5BBD/\u8D28\u91CF\u53EA\u5F71\u54CD\u670D\u52A1\u7AEF\u70E7\u5F55\u7684\u8FD9\u5F20 JPEG\uFF0C\u4E0D\u6539\u6E90\u7D20\u6750\u3002"));
}
function LeftPanel(state) {
  return /*#__PURE__*/React.createElement("aside", {
    className: "glass relative z-10 min-h-0 overflow-y-auto p-3"
  }, /*#__PURE__*/React.createElement(VideoList, {
    videos: state.videos,
    active: state.active,
    setActive: state.setActive
  }), /*#__PURE__*/React.createElement(Divider, null), /*#__PURE__*/React.createElement("div", {
    className: "pt-3"
  }, /*#__PURE__*/React.createElement(LayerGroup, {
    on: state.layers,
    toggle: state.toggle
  })), /*#__PURE__*/React.createElement(Divider, null), /*#__PURE__*/React.createElement("div", {
    className: "pt-3"
  }, /*#__PURE__*/React.createElement(DisplaySettings, state)));
}

/* ---------------- 3 · 中间视频预览 ---------------- */

function Viewer({
  imgRef,
  frame,
  total,
  fps,
  playing,
  mode,
  statusMsg,
  outW,
  outH,
  q,
  pending,
  view3dOn,
  onToggle3d,
  courtModel,
  setCourtModel,
  hasFrame
}) {
  const tc = timecode(frame, fps);
  const res = outW ? outW + "×" + outH : "--";
  return /*#__PURE__*/React.createElement("div", {
    className: "glass relative z-10 flex min-h-0 flex-1 flex-col p-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "viewer-container scanline relative flex-1 overflow-hidden border border-[rgba(122,148,172,0.55)] bg-[#0E1418]"
  }, /*#__PURE__*/React.createElement(CourtWireframe, null), /*#__PURE__*/React.createElement("img", {
    ref: imgRef,
    id: "frame",
    alt: "",
    draggable: "false",
    className: "absolute inset-0 h-full w-full object-contain",
    style: {
      display: view3dOn ? "none" : "block",
      visibility: hasFrame ? "visible" : "hidden"
    }
  }), /*#__PURE__*/React.createElement("canvas", {
    id: "view3d"
  }), /*#__PURE__*/React.createElement("div", {
    id: "view3d-status",
    "data-kind": "info"
  }), /*#__PURE__*/React.createElement("div", {
    className: "pointer-events-none absolute left-3 top-3 z-10 flex items-center gap-3 font-mono text-[11px] tracking-[0.06em] text-white/85"
  }, /*#__PURE__*/React.createElement("span", {
    className: "bg-[rgba(47,168,216,0.18)] px-1.5 py-0.5 text-[#9BE4FF]"
  }, "FRAME ", pad(frame, 4), " / ", pad(total, 5)), /*#__PURE__*/React.createElement("span", {
    className: "text-white/70"
  }, tc)), /*#__PURE__*/React.createElement("button", {
    onClick: onToggle3d,
    className: "absolute right-3 top-3 z-20 h-7 px-3 font-display text-[12px] font-semibold uppercase tracking-[0.16em] " + (view3dOn ? "btn-ice" : "btn-steel")
  }, view3dOn ? "2D" : "3D"), /*#__PURE__*/React.createElement("select", {
    id: "court-model-select",
    value: courtModel,
    onChange: e => setCourtModel(e.target.value),
    className: "court-model-select" + (view3dOn ? " visible" : ""),
    title: "3D \u7403\u573A\u578B\u53F7"
  }, COURT_MODELS.map(m => /*#__PURE__*/React.createElement("option", {
    key: m,
    value: m
  }, m))), !hasFrame ? /*#__PURE__*/React.createElement("div", {
    className: "pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center gap-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "font-display text-[15px] font-semibold uppercase tracking-[0.3em] text-white/25"
  }, "no signal"), /*#__PURE__*/React.createElement("div", {
    className: "font-mono text-[10px] uppercase tracking-[0.18em] text-white/20"
  }, mode === "realtime" ? "等待实时管线出帧" : "等待服务端解码出帧")) : null, /*#__PURE__*/React.createElement("div", {
    className: "pointer-events-none absolute bottom-3 left-3 z-10 flex max-w-[80%] items-center gap-2 font-mono text-[10px] text-white/35"
  }, /*#__PURE__*/React.createElement("span", {
    className: "h-1.5 w-1.5 " + (pending ? "bg-[#C8952B]" : playing ? "bg-[#62CDEC]" : "bg-white/25")
  }), /*#__PURE__*/React.createElement("span", {
    className: "whitespace-nowrap"
  }, pending ? "DECODING" : playing ? "STREAMING" : "IDLE", " \xB7 ", res, " \xB7 Q", q, " \xB7 ", mode === "realtime" ? "RT" : "OFFLINE"), statusMsg ? /*#__PURE__*/React.createElement("span", {
    className: "truncate text-white/55"
  }, statusMsg) : null), /*#__PURE__*/React.createElement(FrameMarks, null)));
}

/* ---------------- 4 · 底部播放控制栏 ---------------- */

function StepButton({
  label,
  onClick,
  tone,
  wide
}) {
  return /*#__PURE__*/React.createElement("button", {
    onClick: onClick,
    title: wide ? label : undefined,
    className: "h-7 font-mono text-[11px] tabular-nums whitespace-nowrap transition " + (wide ? "min-w-[54px] px-1 text-[10px] " : "w-[30px] px-0 ") + (tone === "primary" ? "btn-ice" : "btn-quiet")
  }, label);
}
function PlaybackBar({
  frame,
  total,
  setFrame,
  playing,
  onPlayToggle,
  qualityMode,
  setQualityMode,
  speed,
  setSpeed,
  onShot
}) {
  const pct = total > 1 ? (frame - 1) / (total - 1) * 100 : 0;
  const jump = d => setFrame(clamp(frame + d, 1, total));
  return /*#__PURE__*/React.createElement("div", {
    className: "glass relative z-10 flex shrink-0 flex-wrap items-center gap-1.5 max-[1500px]:gap-1 p-2"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: onPlayToggle,
    className: "flex h-8 shrink-0 items-center gap-2 px-3 font-display text-[13px] font-semibold uppercase tracking-[0.1em] transition " + (playing ? "btn-ice" : "btn-steel")
  }, playing ? /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 12 12",
    className: "h-3 w-3",
    fill: "currentColor"
  }, /*#__PURE__*/React.createElement("rect", {
    x: "2",
    y: "1.5",
    width: "3",
    height: "9"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "7",
    y: "1.5",
    width: "3",
    height: "9"
  })) : /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 12 12",
    className: "h-3 w-3",
    fill: "currentColor"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M2 1.5 L11 6 L2 10.5 Z"
  })), playing ? "停止" : "播放"), /*#__PURE__*/React.createElement("div", {
    className: "flex shrink-0 items-center gap-px"
  }, [-30, -10, -1].map(s => /*#__PURE__*/React.createElement(StepButton, {
    key: s,
    label: (s > 0 ? "+" : "") + s,
    onClick: () => jump(s)
  })), /*#__PURE__*/React.createElement("input", {
    value: pad(frame, 4),
    onChange: e => {
      const n = Number(e.target.value.replace(/\D/g, ""));
      if (!Number.isNaN(n)) setFrame(clamp(n, 1, total));
    },
    className: "mx-px h-7 w-[62px] border border-[rgba(122,148,172,0.5)] bg-white/70 px-1 text-center font-mono text-[11px] tabular-nums text-[#1B2026] outline-none focus:border-[#2FA8D8]"
  }), [1, 10, 30].map(s => /*#__PURE__*/React.createElement(StepButton, {
    key: s,
    label: "+" + s,
    onClick: () => jump(s)
  }))), /*#__PURE__*/React.createElement("div", {
    className: "relative mx-1 h-9 min-w-[110px] flex-1"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute left-0 right-0 top-1/2 h-[3px] -translate-y-1/2 bg-[rgba(122,148,172,0.4)]"
  }), /*#__PURE__*/React.createElement("div", {
    className: "absolute left-0 right-0 top-1/2 h-[9px] -translate-y-1/2 tick-strip opacity-40"
  }), /*#__PURE__*/React.createElement("div", {
    className: "absolute left-0 top-1/2 h-[3px] -translate-y-1/2 bg-[#2FA8D8]",
    style: {
      width: pct + "%"
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "absolute top-1/2 h-[18px] w-[3px] -translate-y-1/2 bg-[#62CDEC] shadow-[0_0_0_1px_rgba(8,32,44,0.6)]",
    style: {
      left: "calc(" + pct + "% - 1.5px)"
    }
  }), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: 1,
    max: total,
    value: frame,
    onChange: e => setFrame(Number(e.target.value)),
    className: "absolute inset-0 h-full w-full opacity-0",
    "aria-label": "\u65F6\u95F4\u8F74"
  })), /*#__PURE__*/React.createElement(Divider, {
    vertical: true
  }), /*#__PURE__*/React.createElement("label", {
    className: "flex shrink-0 items-center gap-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[9px] uppercase tracking-[0.1em] text-[#4C555F]"
  }, "\u8D28\u91CF"), /*#__PURE__*/React.createElement("select", {
    value: qualityMode,
    onChange: e => setQualityMode(e.target.value),
    className: "sharp btn-quiet h-7 pl-2 pr-6 text-[11px]"
  }, /*#__PURE__*/React.createElement("option", {
    value: "accurate"
  }, "\u51C6\u786E"), /*#__PURE__*/React.createElement("option", {
    value: "fast"
  }, "\u5FEB\u901F"))), /*#__PURE__*/React.createElement("label", {
    className: "flex shrink-0 items-center gap-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[9px] uppercase tracking-[0.1em] text-[#4C555F]"
  }, "\u901F\u5EA6"), /*#__PURE__*/React.createElement("select", {
    value: speed,
    onChange: e => setSpeed(e.target.value),
    className: "sharp btn-quiet h-7 pl-2 pr-6 font-mono text-[11px]"
  }, SPEEDS.map(s => /*#__PURE__*/React.createElement("option", {
    key: s,
    value: s
  }, s, "x")))), /*#__PURE__*/React.createElement("div", {
    className: "ml-auto flex shrink-0 items-center gap-px"
  }, /*#__PURE__*/React.createElement(StepButton, {
    wide: true,
    label: "\u4E0A\u4E00\u955C\u5934",
    onClick: () => onShot(-1)
  }), /*#__PURE__*/React.createElement(StepButton, {
    wide: true,
    label: "\u4E0B\u4E00\u955C\u5934",
    onClick: () => onShot(1)
  })));
}

/* ---------------- 5 · 右侧面板 ---------------- */

function StatBlock({
  label,
  en,
  value,
  unit,
  sub
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "flex items-end justify-between border-l-2 border-l-[rgba(122,148,172,0.45)] py-1.5 pl-2.5"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "font-display text-[12px] uppercase tracking-[0.14em] text-[#4C555F]"
  }, label), /*#__PURE__*/React.createElement("div", {
    className: "font-mono text-[9px] uppercase tracking-[0.16em] text-[#8A949F]"
  }, en)), /*#__PURE__*/React.createElement("div", {
    className: "text-right"
  }, /*#__PURE__*/React.createElement("div", {
    className: "font-display text-[30px] font-semibold leading-none tabular-nums text-[#1B2026]"
  }, value, unit ? /*#__PURE__*/React.createElement("span", {
    className: "ml-0.5 font-mono text-[11px] font-normal text-[#8A949F]"
  }, unit) : null), sub ? /*#__PURE__*/React.createElement("div", {
    className: "mt-0.5 font-mono text-[9px] text-[#2FA8D8]"
  }, sub) : null));
}
function RightPanel({
  rt,
  stats,
  meta,
  frame,
  fps,
  width,
  qualityMode,
  mode,
  dtMs
}) {
  const d3 = rt && rt["3d"] || {};
  const objects = rt && rt.objects || [];
  const rows = [["时间码", meta ? timecode(frame, fps) : "--"], ["帧号 / 总数", meta ? pad(frame, 4) + " / " + pad(meta.nframes, 5) : "--"], ["输出分辨率", width + " × " + (meta && meta.height ? Math.round(width * meta.height / meta.width) : "--")], ["素材", meta ? meta.width + " × " + meta.height + " · " + (meta.fps ? meta.fps.toFixed(2) : "--") + "fps" : "--"], ["管线", mode === "realtime" ? "实时烧录 rt_frame" : "离线渲染 frame.jpg"], ["质量档", qualityMode === "accurate" ? "准确 (RTMO)" : "快速 (YOLO)"], ["推理引擎", rt && rt.engine ? rt.engine : "--"], ["检测对象", rt ? rt.players + (rt.balls ? " + " + rt.balls + " 球" : "") : "--"], ["骨架", rt ? rt.poses + " 人" : "--"], ["3D", d3.ok ? "启用 " + d3.n + " 人" : "关闭 (" + (d3.why || "disabled") + ")"], ["运动约束", d3.prior ? d3.prior.enabled ? "开启 · " + (d3.prior.hit_rate_pct || 0).toFixed(1) + "% 帧修正" : "关闭" : "--"], ["运行档", rt && rt.profile || "--"], ["处理延迟", dtMs ? dtMs.toFixed(0) + " ms" : "--"]];
  return /*#__PURE__*/React.createElement("aside", {
    className: "glass relative z-10 flex min-h-0 flex-col gap-3 overflow-y-auto p-3"
  }, /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "04",
    zh: "\u5B9E\u65F6\u7EDF\u8BA1",
    en: "live",
    tail: stats.procFps ? stats.procFps.toFixed(1) + " FPS" : mode === "realtime" ? "LIVE" : "离线"
  }), /*#__PURE__*/React.createElement("div", {
    className: "space-y-px"
  }, /*#__PURE__*/React.createElement(StatBlock, {
    label: "\u7403\u5458",
    en: "players",
    value: stats.players,
    sub: rt ? stats.players ? "TRACK OK" : "无检出" : "等待分析"
  }), /*#__PURE__*/React.createElement(StatBlock, {
    label: "\u7BEE\u7403",
    en: "balls",
    value: stats.balls,
    sub: rt && rt.balls ? "CONF " + (objects.find(o => o.cat === "ball") || {}).conf : null
  }), /*#__PURE__*/React.createElement(StatBlock, {
    label: "\u5E27\u7387",
    en: "fps",
    value: stats.playing ? stats.playFps.toFixed(1) : (fps || 0).toFixed(1),
    unit: "fps"
  }))), /*#__PURE__*/React.createElement(Divider, null), /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "05",
    zh: "\u5E27\u6570\u636E",
    en: "frame meta"
  }), /*#__PURE__*/React.createElement("dl", {
    className: "space-y-px"
  }, rows.map(([k, v]) => /*#__PURE__*/React.createElement("div", {
    key: k,
    className: "flex items-baseline justify-between gap-3 border-b border-[rgba(122,148,172,0.22)] py-1 last:border-b-0"
  }, /*#__PURE__*/React.createElement("dt", {
    className: "shrink-0 text-[11px] text-[#4C555F]"
  }, k), /*#__PURE__*/React.createElement("dd", {
    className: "truncate text-right font-mono text-[11px] tabular-nums text-[#1B2026]"
  }, v))))), /*#__PURE__*/React.createElement(Divider, null), /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "06",
    zh: "\u68C0\u6D4B\u7ED3\u679C",
    en: "detections",
    tail: objects.length + " ROWS"
  }), /*#__PURE__*/React.createElement("div", {
    className: "overflow-x-auto"
  }, /*#__PURE__*/React.createElement("table", {
    className: "w-full border-collapse font-mono text-[10px] tabular-nums"
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", {
    className: "border-b border-[rgba(122,148,172,0.6)] text-left"
  }, ["ID", "类别", "置信", "框 (x0,y0,x1,y1)", "号码"].map(h => /*#__PURE__*/React.createElement("th", {
    key: h,
    className: "whitespace-nowrap py-1 pr-2 font-display text-[10px] font-semibold uppercase tracking-[0.08em] text-[#4C555F]"
  }, h)))), /*#__PURE__*/React.createElement("tbody", null, objects.length === 0 ? /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("td", {
    colSpan: "5",
    className: "py-2 pr-2 text-[#8A949F]"
  }, mode === "realtime" ? "该帧尚无分析缓存" : "离线渲染需要先跑分析任务")) : objects.map(o => /*#__PURE__*/React.createElement("tr", {
    key: o.id,
    className: "border-b border-[rgba(122,148,172,0.2)] last:border-b-0 hover:bg-[rgba(47,168,216,0.10)]"
  }, /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#2FA8D8]"
  }, "#", pad(o.id, 2)), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#1B2026]"
  }, o.cat), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#4C555F]"
  }, o.conf.toFixed(2)), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#8A949F]"
  }, o.box.join(",")), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 font-semibold text-[#1B2026]"
  }, o.jersey || "—"))))))));
}

/* ---------------- 页面 ---------------- */

function App() {
  const [videos, setVideos] = useState([]);
  const [vid, setVid] = useState(null);
  const [meta, setMeta] = useState(null);
  const [shots, setShots] = useState([]);
  const [frame, setFrame] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [layers, setLayers] = useState(new Set(["hud", "boxes", "pose"]));
  const [width, setWidth] = useState(1280);
  const [q, setQ] = useState(72);
  const [qualityMode, setQualityMode] = useState("accurate");
  const [speed, setSpeed] = useState("1");
  const [mode, setMode] = useState("realtime"); // realtime | offline
  const modeRef = useRef("realtime"); // 同一份状态的可读快照，避开 setState 批处理带来的竞态
  const pickMode = useCallback(m => {
    modeRef.current = m;
    setMode(m);
  }, []);
  const [rt, setRt] = useState(null); // rt_meta.json 摘要
  const [dtMs, setDtMs] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [hasFrame, setHasFrame] = useState(false);
  const [pending, setPending] = useState(false); // 有一帧正在服务端解码/烧录
  const [toasts, setToasts] = useState([]);
  const [busy, setBusy] = useState(false);
  const [view3dOn, setView3dOn] = useState(false);
  const [courtModel, setCourtModel] = useState("FIBA");
  const imgRef = useRef(null);
  const loop = useRef({
    token: 0,
    playT0: 0,
    playFrames: 0,
    loadFails: 0,
    noFpsCount: 0,
    playWidth: 1280,
    reqId: 0
  });
  const stats = useRef({
    players: 0,
    balls: 0,
    playFps: 0,
    procFps: 0
  });
  const playingRef = useRef(0); // 播放令牌：0 表示未播放，非 0 是本次播放的 token
  const frameRef = useRef(1); // 播放循环里的权威帧号，避免等 React 提交后才更新
  const toastSeq = useRef(0);
  const [statTick, setStatTick] = useState(0); // 只用来触发统计区重绘

  const total = meta ? meta.nframes : 1;
  const fps = meta ? meta.fps : 29.97;
  const toast = useCallback((msg, kind) => {
    const id = ++toastSeq.current;
    setToasts(prev => [...prev.slice(-3), {
      id,
      msg,
      kind: kind || "info"
    }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4200);
  }, []);
  const layerParam = useCallback(() => [...layers].join(","), [layers]);

  /* ── 帧图 URL ── */
  const want3d = view3dOn ? "&enable_3d=1" : "";
  const rtUrl = useCallback(n => `/api/videos/${vid}/rt_frame.jpg?n=${n}&layers=${layerParam()}&width=${loop.current.playWidth}` + `&q=${q}&quality=${qualityMode}&rt=${Date.now()}${want3d}`, [vid, layerParam, q, qualityMode, want3d]);
  const offUrl = useCallback(n => {
    // 播放时用降档宽度：离线渲染每帧都要从关键帧重解，宽度小一点能明显提速
    const w = playingRef.current ? loop.current.playWidth : width;
    return `/api/videos/${vid}/frame.jpg?n=${n}&layers=${layerParam()}&width=${w}&q=${q}` + `&quality=${qualityMode}&realtime=0${want3d}`;
  }, [vid, layerParam, width, q, qualityMode, want3d]);

  /* ── 单帧加载：直接给 <img> 设 src，浏览器原生解码 ──
     离线档不设超时：服务端每帧要从关键帧重解，超时后浏览器放弃、服务端却还在算，
     连续几次就会把解码线程堆满，整个服务被拖死。离线只串行、等到底。 */
  const loadFrame = useCallback((n, useRealtime, retries) => {
    return new Promise((resolve, reject) => {
      const img = imgRef.current;
      if (!img || !vid) return reject(new Error("未选择视频"));
      const req = ++loop.current.reqId;
      let settled = false;
      const left = retries === undefined ? useRealtime ? 1 : 0 : retries;
      const cleanup = () => {
        img.onload = null;
        img.onerror = null;
        clearTimeout(timer);
      };
      const finish = err => {
        if (settled) return;
        settled = true;
        cleanup();
        err ? reject(err) : resolve();
      };
      const timer = useRealtime ? setTimeout(() => {
        if (req === loop.current.reqId) finish(new Error("帧请求超时"));
      }, 8000) : 0;
      img.onload = () => {
        if (req === loop.current.reqId) {
          setHasFrame(true);
          setPending(false);
          finish(null);
        }
      };
      img.onerror = () => {
        if (req !== loop.current.reqId) return;
        if (left > 0) {
          cleanup();
          setTimeout(() => loadFrame(n, useRealtime, left - 1).then(resolve, reject), 120);
          return;
        }
        setHasFrame(false);
        setPending(false);
        finish(new Error(useRealtime ? "实时帧加载失败" : "帧解码失败"));
      };
      setPending(true);
      img.src = useRealtime ? rtUrl(n) : offUrl(n);
    });
  }, [vid, rtUrl, offUrl]);

  /* ── 面板摘要：只读服务端缓存，204 表示该帧还没算过 ── */
  const refreshMeta = useCallback(async n => {
    if (!vid || modeRef.current !== "realtime") return;
    try {
      const r = await fetch(`/api/videos/${vid}/rt_meta.json?n=${n}&quality=${qualityMode}`);
      if (!r.ok) return;
      const data = await r.json();
      setRt(data);
      stats.current.players = data.players || 0;
      stats.current.balls = data.balls || 0;
      setStatTick(t => t + 1);
    } catch (_) {/* 面板刷新失败不影响看帧 */}
  }, [vid, qualityMode]);
  const goto = useCallback(n => {
    if (!vid) return;
    const clamped = clamp(n, 1, total);
    const useRealtime = modeRef.current === "realtime";
    frameRef.current = clamped;
    setFrame(clamped);
    loadFrame(clamped, useRealtime).then(() => refreshMeta(clamped)).catch(e => {
      setHasFrame(false);
      if (useRealtime) {
        // 实时档跑不了（缺权重/依赖）不是错误终点：换离线渲染继续出图
        pickMode("offline");
        setStatusMsg("实时管线不可用，已降级离线渲染");
        toast("实时管线不可用（缺模型权重或推理依赖），已降级为离线渲染", "warn");
        loadFrame(clamped, false).then(() => setStatusMsg("")).catch(e2 => setStatusMsg(e2.message));
      } else {
        setStatusMsg(e.message);
      }
    });
  }, [vid, total, loadFrame, refreshMeta, toast, pickMode]);

  /* ── 播放：逐帧拉取，按源帧率推进；实时档达不到 30fps 时自动降播放宽度 ── */
  const play = useCallback(() => {
    if (!vid) {
      toast("请先在视频库选择素材", "error");
      return;
    }
    const useRealtime = modeRef.current === "realtime";
    const sp = parseFloat(speed) || 1;
    const L = loop.current;
    L.token += 1;
    const step = Math.max(1, Math.round(sp));
    const interval = 1000 / ((fps || 29.97) * sp);
    // 播放默认 960：1280 的服务端耗时虽可接受，但浏览器 JPEG 解码会拖到 ~19fps。
    // 离线档每帧都要从关键帧重解（本机素材 GOP≈150），降到 480 并串行请求才不压垮服务。
    L.playWidth = Math.min(width, useRealtime ? 960 : 480);
    L.loadFails = 0;
    L.playT0 = performance.now();
    L.playFrames = 0;
    setPlaying(true);
    setStatusMsg(useRealtime ? "实时分析 " + sp + "x" : "离线逐帧：每帧重解 GOP，实际帧率见右侧统计");
    if (!useRealtime) {
      toast("离线档本机单帧解码约需 1 分钟，播放会非常慢；补上实时权重后走 rt_frame 才是设计路径", "warn");
    }
    const tick = async () => {
      if (!playingRef.current || L.token !== playingRef.current) return;
      const t0 = performance.now();
      const at = frameRef.current;
      try {
        await loadFrame(at, useRealtime);
        L.loadFails = 0;
      } catch (e) {
        L.loadFails += 1;
        if (L.loadFails <= 2) {
          setTimeout(tick, 180);
          return;
        }
        L.loadFails = 0;
        const skipped = clamp(at + 1, 1, total);
        frameRef.current = skipped;
        setFrame(skipped);
        if (skipped >= total) {
          stopPlay();
          return;
        }
        setTimeout(tick, 80);
        return;
      }
      if (!playingRef.current || L.token !== playingRef.current) return;
      const dt = performance.now() - t0;
      setDtMs(dt);
      const inst = 1000 / Math.max(dt, 1);
      stats.current.procFps = stats.current.procFps ? stats.current.procFps * 0.7 + inst * 0.3 : inst;
      L.playFrames += 1;
      const wall = Math.max(0.001, (performance.now() - L.playT0) / 1000);
      stats.current.playFps = L.playFrames / wall; // 播放 FPS 必须按墙钟算
      setStatTick(t => t + 1);
      if (at % 5 === 0) refreshMeta(at);
      if (useRealtime) {
        if (dt > 36) L.noFpsCount += 1;else L.noFpsCount = 0;
        if (L.noFpsCount >= 12 && L.playWidth > 800) {
          L.playWidth = Math.max(800, L.playWidth - 160);
          L.noFpsCount = 0;
          toast("帧耗时偏高，已降档到 " + L.playWidth + " 宽保流畅", "info");
        }
      }

      // 严格按源视频时间轴推进：1x 每次只前进一帧，不额外追帧加速
      const next = clamp(at + step, 1, total);
      if (next >= total) {
        stopPlay();
        return;
      }
      frameRef.current = next;
      setFrame(next);
      setTimeout(tick, Math.max(0, interval - dt));
    };
    playingRef.current = L.token;
    tick();
  }, [vid, speed, fps, width, total, loadFrame, refreshMeta, toast]);
  const stopPlay = useCallback(() => {
    playingRef.current = 0;
    loop.current.token += 1;
    setPlaying(false);
    setStatusMsg("");
  }, []);

  /* ── 素材 ── */
  const refreshVideos = useCallback(async () => {
    try {
      const data = await api("/api/videos");
      setVideos(data.videos || []);
      return data.videos || [];
    } catch (e) {
      toast("读取视频列表失败: " + e.message, "error");
      return [];
    }
  }, [toast]);
  const selectVideo = useCallback(async id => {
    setVid(id);
    setHasFrame(false);
    setRt(null);
    try {
      const m = await api(`/api/videos/${id}`);
      setMeta(m);
      frameRef.current = 1;
      setFrame(1);
      loop.current.noFpsCount = 0;
      stats.current = {
        players: 0,
        balls: 0,
        playFps: 0,
        procFps: 0
      };
      try {
        const s = await api(`/api/videos/${id}/shots`);
        setShots(s.shots || []);
      } catch (_) {
        setShots([]);
      }
      if (window.BSK3D) window.BSK3D.setVideo(id, {
        quality: qualityMode,
        fps: m.fps
      });
      toast("已选择: " + m.name, "success");
    } catch (e) {
      toast("读取素材信息失败: " + e.message, "error");
    }
  }, [qualityMode, toast]);
  const importVideo = useCallback(async (path, setPath) => {
    if (!path || !path.trim()) {
      toast("请输入视频路径", "error");
      return;
    }
    setBusy(true);
    try {
      await api("/api/videos", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          path: path.trim()
        })
      });
      setPath("");
      toast("已提交导入，正在索引（探测完成后列表会刷新）", "success");
      setTimeout(refreshVideos, 2500);
      setTimeout(refreshVideos, 8000);
    } catch (e) {
      toast("导入失败: " + e.message, "error");
    } finally {
      setBusy(false);
    }
  }, [refreshVideos, toast]);

  /* ── 镜头跳转 ── */
  const onShot = useCallback(dir => {
    if (!shots.length) {
      toast("该素材还没有镜头切分结果", "warn");
      return;
    }
    const cur = shots.findIndex(s => frame >= s.start_frame && frame <= s.end_frame);
    if (dir < 0) goto(cur > 0 ? shots[cur - 1].start_frame : 1);else goto(cur >= 0 && cur < shots.length - 1 ? shots[cur + 1].start_frame : total);
  }, [shots, frame, total, goto, toast]);

  /* ── 3D ── */
  const toggle3D = useCallback(() => {
    const B = window.BSK3D;
    if (!B) {
      toast("3D 模块未加载（src/view3d.js 缺失）", "error");
      return;
    }
    if (!B.available) {
      toast("3D 不可用：" + (B.error || "缺少 Three.js"), "error");
      return;
    }
    const on = !view3dOn;
    setView3dOn(on);
    if (on) B.init();
    B.setVideo(vid, {
      quality: qualityMode,
      fps: fps
    });
    B.setCourtModel(courtModel);
    B.setEnabled(on);
    if (on) {
      B.showFrame(frame);
      setStatusMsg("3D 视图（拖拽旋转 / 滚轮缩放 / 双击复位）");
      if (modeRef.current !== "realtime") {
        // 离线档不重跑检测：/rt_3d 会读取当前帧 poses.jsonl 并现场标定补算。
        B.setStatus("离线 3D：读取当前帧姿态并标定；无姿态或标定失败时无骨架", "warn");
      }
    } else if (!playingRef.current) loadFrame(frame, modeRef.current === "realtime").catch(() => {});
  }, [view3dOn, vid, qualityMode, fps, courtModel, frame, loadFrame, toast]);
  useEffect(() => {
    if (window.BSK3D) window.BSK3D.setQuality(qualityMode);
  }, [qualityMode]);
  useEffect(() => {
    if (window.BSK3D && view3dOn) window.BSK3D.setCourtModel(courtModel);
  }, [courtModel, view3dOn]);
  useEffect(() => {
    if (view3dOn && window.BSK3D && !playingRef.current) window.BSK3D.showFrame(frame);
  }, [frame, view3dOn]);

  /* ── 首次装载：列表 → 探测实时档 → 选素材 → 深链 ── */
  useEffect(() => {
    (async () => {
      const qs = new URLSearchParams(location.search);
      const list = await refreshVideos();
      if (!list.length) return;
      const wantVid = qs.get("vid") || list[0].vid;

      // 先探一次实时接口：省掉第一帧白跑的实时请求
      try {
        const probe = await fetch(`/api/videos/${wantVid}/rt_frame.jpg?n=1&layers=hud&width=480&q=40`);
        if (!probe.ok) {
          pickMode("offline");
          setStatusMsg("实时管线不可用，走离线渲染");
          toast("实时管线不可用（HTTP " + probe.status + "，通常是缺 rtmo/rtmdet 权重），已降级为离线渲染", "warn");
        }
      } catch (_) {
        pickMode("offline");
        toast("实时接口无响应，已降级为离线渲染", "warn");
      }
      await selectVideo(wantVid);
      const wantFrame = parseInt(qs.get("frame") || "0", 10);
      const wantCourt = qs.get("court");
      if (wantCourt && COURT_MODELS.includes(wantCourt)) setCourtModel(wantCourt);
      if (qs.get("view") === "3d") setTimeout(toggle3D, 300);
      if (wantFrame > 1) setTimeout(() => goto(wantFrame), 400);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── 参数变化时重画当前帧（播放中交给循环自己推进）──
     mode 不进依赖：降级路径在 goto 里已经自己重画过一次，再触发一次只会白跑请求。 */
  useEffect(() => {
    if (playingRef.current) return;
    if (!vid || !meta) return;
    goto(frameRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, width, q, qualityMode, vid, meta]);

  /* ── 键盘快捷键 ── */
  useEffect(() => {
    const onKey = e => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "ArrowLeft") goto(frameRef.current - (e.shiftKey ? 10 : 1));else if (e.key === "ArrowRight") goto(frameRef.current + (e.shiftKey ? 10 : 1));else if (e.key === " ") {
        e.preventDefault();
        playingRef.current ? stopPlay() : play();
      } else if (e.key === "[") onShot(-1);else if (e.key === "]") onShot(1);else if (e.key === "v" || e.key === "V") toggle3D();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [goto, play, stopPlay, onShot, toggle3D]);
  const toggle = k => setLayers(prev => {
    const next = new Set(prev);
    next.has(k) ? next.delete(k) : next.add(k);
    return next;
  });
  const outH = meta && meta.width ? Math.round(width * meta.height / meta.width) : 0;
  return /*#__PURE__*/React.createElement("div", {
    className: "dot-grid h-full overflow-auto bg-[#E9EDF1] font-sans text-[#1B2026]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex h-full min-w-[1180px] flex-col"
  }, /*#__PURE__*/React.createElement(TopNav, {
    onImport: importVideo,
    onRefresh: refreshVideos,
    busy: busy
  }), /*#__PURE__*/React.createElement("main", {
    className: "relative z-10 grid min-h-0 flex-1 grid-cols-[248px_minmax(0,1fr)_312px] max-[1500px]:grid-cols-[224px_minmax(0,1fr)_288px] gap-3 p-3"
  }, /*#__PURE__*/React.createElement(LeftPanel, {
    videos: videos,
    active: vid,
    setActive: selectVideo,
    layers: layers,
    toggle: toggle,
    width: width,
    setWidth: setWidth,
    quality: q,
    setQuality: setQ
  }), /*#__PURE__*/React.createElement("section", {
    className: "flex min-h-0 min-w-0 flex-col gap-3"
  }, /*#__PURE__*/React.createElement(Viewer, {
    imgRef: imgRef,
    frame: frame,
    total: total,
    fps: fps,
    playing: playing,
    mode: mode,
    statusMsg: statusMsg,
    outW: width,
    outH: outH,
    q: q,
    pending: pending,
    view3dOn: view3dOn,
    onToggle3d: toggle3D,
    courtModel: courtModel,
    setCourtModel: setCourtModel,
    hasFrame: hasFrame
  }), /*#__PURE__*/React.createElement(PlaybackBar, {
    frame: frame,
    total: total,
    setFrame: n => {
      if (playingRef.current) stopPlay();
      goto(n);
    },
    playing: playing,
    onPlayToggle: () => playingRef.current ? stopPlay() : play(),
    qualityMode: qualityMode,
    setQualityMode: setQualityMode,
    speed: speed,
    setSpeed: setSpeed,
    onShot: onShot
  })), /*#__PURE__*/React.createElement(RightPanel, {
    rt: rt,
    stats: {
      ...stats.current,
      playing: playing
    },
    meta: meta,
    frame: frame,
    fps: fps,
    width: width,
    qualityMode: qualityMode,
    mode: mode,
    dtMs: dtMs,
    statTick: statTick
  }))), /*#__PURE__*/React.createElement(Toasts, {
    items: toasts
  }));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));
