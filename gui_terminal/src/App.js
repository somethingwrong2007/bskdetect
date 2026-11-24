/* 本文件由 gui_terminal/build.py 从同目录 App.jsx 预编译生成，请勿手改。
   编译器：@babel/standalone 7.26.7 (MIT)，preset=react
   改界面只改 App.jsx，然后跑 python gui_terminal/build.py 重新生成。 */

/* BSK Detect · 实时篮球视频分析调试台 —— 静态 UI（React + Tailwind）
   只做界面还原：数据全部是写死的样例，没有取流、解码与推理逻辑。
   图层开关 / 滑块 / 播放态属于纯前端交互状态，方便评审界面反馈。 */

const {
  useState
} = React;

/* ---------------- 静态样例数据 ---------------- */

const VIDEOS = [{
  vid: "51",
  title: "阵前再亮旧时剑，哈登对阵步行者51分集锦.mp4",
  frames: 8455,
  fps: 59.94,
  state: "ready"
}, {
  vid: "47",
  title: "末节 13-0 反超：勇士 vs 湖人 关键回合.mp4",
  frames: 6120,
  fps: 29.97,
  state: "ready"
}, {
  vid: "33",
  title: "挡拆外弹解析：约基奇低位策应 12 例.mp4",
  frames: 4380,
  fps: 30.0,
  state: "indexing"
}, {
  vid: "12",
  title: "快攻推进路线：雷霆转换进攻片段.mp4",
  frames: 2960,
  fps: 59.94,
  state: "ready"
}];
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
const FRAME_ROWS = [["时间码", "00:00.017"], ["帧号 / 总数", "0001 / 08455"], ["输出分辨率", "1280 × 720"], ["机位", "CAM-01 · 底线高机位"], ["标定残差", "3.42 px"], ["推理耗时", "det 6.1 / pose 4.3 ms"], ["检测目标", "11（球员 10 · 球 1）"], ["号码识别", "已发布 6 / 10"]];
const DET_ROWS = [["#01", "player", "0.97", "812,168,906,402", "23"], ["#02", "player", "0.95", "640,152,738,396", "11"], ["#03", "player", "0.93", "994,174,1088,408", "7"], ["#04", "player", "0.91", "452,160,548,392", "30"], ["#05", "player", "0.88", "1140,180,1232,410", "—"], ["#06", "ball", "0.86", "704,236,738,270", "—"]];

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

/* ---------------- 1 · 顶部导航 ---------------- */

function TopNav({
  onImport
}) {
  const [path, setPath] = useState("E:\\videos\\harden_vs_walkers_51.mp4");
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
    spellCheck: "false",
    className: "h-8 w-[360px] border border-[rgba(122,148,172,0.5)] bg-white/70 px-2 font-mono text-[12px] text-[#1B2026] outline-none focus:border-[#2FA8D8]",
    placeholder: "\u89C6\u9891\u8DEF\u5F84 (e.g. E:\\videos\\game.mp4)"
  }), /*#__PURE__*/React.createElement("button", {
    onClick: onImport,
    className: "btn-ice h-8 px-4 font-display text-[13px] font-semibold uppercase tracking-[0.1em]"
  }, "\u5BFC\u5165\u89C6\u9891"), /*#__PURE__*/React.createElement("button", {
    title: "\u5237\u65B0\u5217\u8868",
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
  active,
  setActive
}) {
  return /*#__PURE__*/React.createElement("section", {
    className: "mb-4"
  }, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "01",
    zh: "\u89C6\u9891\u5E93",
    en: "library",
    tail: VIDEOS.length + " CLIPS"
  }), /*#__PURE__*/React.createElement("div", {
    className: "max-h-[236px] space-y-px overflow-y-auto pr-px"
  }, VIDEOS.map(v => {
    const on = v.vid === active;
    return /*#__PURE__*/React.createElement("button", {
      key: v.vid,
      onClick: () => setActive(v.vid),
      className: "block w-full border-l-2 px-2.5 py-2 text-left transition " + (on ? "border-l-[#2FA8D8] bg-[rgba(47,168,216,0.13)]" : "border-l-[rgba(122,148,172,0.4)] hover:border-l-[#2FA8D8]/70 hover:bg-white/60")
    }, /*#__PURE__*/React.createElement("div", {
      className: "truncate text-[12px] leading-snug " + (on ? "font-semibold text-[#1B2026]" : "text-[#4C555F]")
    }, v.title), /*#__PURE__*/React.createElement("div", {
      className: "mt-1 flex items-center gap-3 font-mono text-[10px] tracking-[0.04em] text-[#8A949F]"
    }, /*#__PURE__*/React.createElement("span", null, v.frames.toLocaleString(), " FR"), /*#__PURE__*/React.createElement("span", null, v.fps.toFixed(2), " FPS"), /*#__PURE__*/React.createElement("span", {
      className: v.state === "ready" ? "text-[#2FA8D8]" : "text-[#C8952B]"
    }, v.state === "ready" ? "READY" : "INDEX")));
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
  }));
}
function LeftPanel(state) {
  return /*#__PURE__*/React.createElement("aside", {
    className: "glass relative z-10 min-h-0 overflow-y-auto p-3"
  }, /*#__PURE__*/React.createElement(VideoList, {
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
function Viewer({
  frame,
  total,
  playing
}) {
  const tc = "00:00." + String(Math.round(frame / 59.94 * 1000)).padStart(3, "0");
  return /*#__PURE__*/React.createElement("div", {
    className: "glass relative z-10 flex min-h-0 flex-1 flex-col p-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "scanline relative flex-1 overflow-hidden border border-[rgba(122,148,172,0.55)] bg-[#0E1418]"
  }, /*#__PURE__*/React.createElement(CourtWireframe, null), /*#__PURE__*/React.createElement("div", {
    className: "absolute left-3 top-3 flex items-center gap-3 font-mono text-[11px] tracking-[0.06em] text-white/85"
  }, /*#__PURE__*/React.createElement("span", {
    className: "bg-[rgba(47,168,216,0.18)] px-1.5 py-0.5 text-[#9BE4FF]"
  }, "FRAME ", String(frame).padStart(4, "0"), " / ", String(total).padStart(5, "0")), /*#__PURE__*/React.createElement("span", {
    className: "text-white/70"
  }, tc)), /*#__PURE__*/React.createElement("button", {
    className: "btn-steel absolute right-3 top-3 h-7 px-3 font-display text-[12px] font-semibold uppercase tracking-[0.16em]"
  }, "3D"), /*#__PURE__*/React.createElement("div", {
    className: "absolute inset-0 flex flex-col items-center justify-center gap-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "font-display text-[15px] font-semibold uppercase tracking-[0.3em] text-white/25"
  }, "no signal"), /*#__PURE__*/React.createElement("div", {
    className: "font-mono text-[10px] uppercase tracking-[0.18em] text-white/20"
  }, "\u9759\u6001\u754C\u9762 \xB7 \u672A\u63A5\u5165\u5B9E\u65F6\u7BA1\u7EBF")), /*#__PURE__*/React.createElement("div", {
    className: "absolute bottom-3 left-3 flex items-center gap-2 font-mono text-[10px] text-white/35"
  }, /*#__PURE__*/React.createElement("span", {
    className: "h-1.5 w-1.5 " + (playing ? "bg-[#62CDEC]" : "bg-white/25")
  }), playing ? "STREAMING @ 1280×720" : "IDLE · 1280×720 · Q72"), /*#__PURE__*/React.createElement(FrameMarks, null)));
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
  setPlaying
}) {
  const pct = (frame - 1) / (total - 1) * 100;
  const [quality, setQuality] = useState("accurate");
  const [speed, setSpeed] = useState("1");
  const jump = d => setFrame(Math.min(total, Math.max(1, frame + d)));
  return /*#__PURE__*/React.createElement("div", {
    className: "glass relative z-10 flex shrink-0 flex-wrap items-center gap-1.5 max-[1500px]:gap-1 p-2"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setPlaying(!playing),
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
    value: String(frame).padStart(4, "0"),
    onChange: e => {
      const n = Number(e.target.value.replace(/\D/g, ""));
      if (!Number.isNaN(n)) setFrame(Math.min(total, Math.max(1, n)));
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
    value: quality,
    onChange: e => setQuality(e.target.value),
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
  }, ["0.25", "0.5", "1", "2", "4"].map(s => /*#__PURE__*/React.createElement("option", {
    key: s,
    value: s
  }, s, "x")))), /*#__PURE__*/React.createElement("div", {
    className: "ml-auto flex shrink-0 items-center gap-px"
  }, /*#__PURE__*/React.createElement(StepButton, {
    wide: true,
    label: "\u4E0A\u4E00\u955C\u5934",
    onClick: () => jump(-120)
  }), /*#__PURE__*/React.createElement(StepButton, {
    wide: true,
    label: "\u4E0B\u4E00\u955C\u5934",
    onClick: () => jump(120)
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
function RightPanel() {
  return /*#__PURE__*/React.createElement("aside", {
    className: "glass relative z-10 flex min-h-0 flex-col gap-3 overflow-y-auto p-3"
  }, /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "04",
    zh: "\u5B9E\u65F6\u7EDF\u8BA1",
    en: "live",
    tail: "30.0 FPS"
  }), /*#__PURE__*/React.createElement("div", {
    className: "space-y-px"
  }, /*#__PURE__*/React.createElement(StatBlock, {
    label: "\u7403\u5458",
    en: "players",
    value: "10",
    sub: "TRACK OK"
  }), /*#__PURE__*/React.createElement(StatBlock, {
    label: "\u7BEE\u7403",
    en: "balls",
    value: "1",
    sub: "CONF 0.86"
  }), /*#__PURE__*/React.createElement(StatBlock, {
    label: "\u5E27\u7387",
    en: "fps",
    value: "59.9",
    unit: "fps"
  }))), /*#__PURE__*/React.createElement(Divider, null), /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement(SectionTitle, {
    index: "05",
    zh: "\u5E27\u6570\u636E",
    en: "frame meta"
  }), /*#__PURE__*/React.createElement("dl", {
    className: "space-y-px"
  }, FRAME_ROWS.map(([k, v]) => /*#__PURE__*/React.createElement("div", {
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
    tail: DET_ROWS.length + " ROWS"
  }), /*#__PURE__*/React.createElement("div", {
    className: "overflow-x-auto"
  }, /*#__PURE__*/React.createElement("table", {
    className: "w-full border-collapse font-mono text-[10px] tabular-nums"
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", {
    className: "border-b border-[rgba(122,148,172,0.6)] text-left"
  }, ["ID", "类别", "置信", "框 (x0,y0,x1,y1)", "号码"].map(h => /*#__PURE__*/React.createElement("th", {
    key: h,
    className: "whitespace-nowrap py-1 pr-2 font-display text-[10px] font-semibold uppercase tracking-[0.08em] text-[#4C555F]"
  }, h)))), /*#__PURE__*/React.createElement("tbody", null, DET_ROWS.map(r => /*#__PURE__*/React.createElement("tr", {
    key: r[0],
    className: "border-b border-[rgba(122,148,172,0.2)] last:border-b-0 hover:bg-[rgba(47,168,216,0.10)]"
  }, /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#2FA8D8]"
  }, r[0]), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#1B2026]"
  }, r[1]), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#4C555F]"
  }, r[2]), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 text-[#8A949F]"
  }, r[3]), /*#__PURE__*/React.createElement("td", {
    className: "whitespace-nowrap py-1 pr-2 font-semibold text-[#1B2026]"
  }, r[4]))))))));
}

/* ---------------- 页面 ---------------- */

function App() {
  const [active, setActive] = useState("51");
  const [layers, setLayers] = useState(new Set(["hud", "boxes", "pose"]));
  const [width, setWidth] = useState(1280);
  const [quality, setQuality] = useState(72);
  const [frame, setFrame] = useState(1);
  const [playing, setPlaying] = useState(false);
  const total = (VIDEOS.find(v => v.vid === active) || VIDEOS[0]).frames;
  const toggle = k => setLayers(prev => {
    const next = new Set(prev);
    next.has(k) ? next.delete(k) : next.add(k);
    return next;
  });
  return /*#__PURE__*/React.createElement("div", {
    className: "dot-grid h-full overflow-auto bg-[#E9EDF1] font-sans text-[#1B2026]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex h-full min-w-[1180px] flex-col"
  }, /*#__PURE__*/React.createElement(TopNav, null), /*#__PURE__*/React.createElement("main", {
    className: "relative z-10 grid min-h-0 flex-1 grid-cols-[248px_minmax(0,1fr)_312px] max-[1500px]:grid-cols-[224px_minmax(0,1fr)_288px] gap-3 p-3"
  }, /*#__PURE__*/React.createElement(LeftPanel, {
    active: active,
    setActive: setActive,
    layers: layers,
    toggle: toggle,
    width: width,
    setWidth: setWidth,
    quality: quality,
    setQuality: setQuality
  }), /*#__PURE__*/React.createElement("section", {
    className: "flex min-h-0 min-w-0 flex-col gap-3"
  }, /*#__PURE__*/React.createElement(Viewer, {
    frame: frame,
    total: total,
    playing: playing
  }), /*#__PURE__*/React.createElement(PlaybackBar, {
    frame: frame,
    total: total,
    setFrame: setFrame,
    playing: playing,
    setPlaying: setPlaying
  })), /*#__PURE__*/React.createElement(RightPanel, null))));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));
