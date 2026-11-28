/* BSK Detect · 实时篮球视频分析调试台（React + Tailwind）
   界面：浅色冷调机能风，全站直角；数据来自本机 FastAPI（bsk/server.py）。
   实时管线不可用时自动降级到离线渲染 frame.jpg，并在界面上说明原因。 */

const { useState, useEffect, useRef, useCallback } = React;

/* ---------------- 常量与工具 ---------------- */

// 实时档支持的图层；seg/trail 需要额外模型，未接入时后端会自己忽略
const LAYERS = [
  { key: "hud", label: "HUD" },
  { key: "boxes", label: "检测框" },
  { key: "pose", label: "姿态" },
  { key: "floor", label: "地板" },
  { key: "motion", label: "运动" },
];

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

function SectionTitle({ index, zh, en, tail }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="font-mono text-[10px] leading-none text-[#2FA8D8]">{index}</span>
      <span className="h-3 w-[3px] bg-[#2FA8D8]" />
      <span className="font-display text-[13px] font-semibold uppercase tracking-[0.16em] text-[#1B2026]">
        {zh}
      </span>
      <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-[#8A949F]">{en}</span>
      <span className="flex-1 h-px bg-[rgba(122,148,172,0.38)]" />
      {tail ? <span className="font-mono text-[10px] text-[#4C555F]">{tail}</span> : null}
    </div>
  );
}

function Corner({ className }) {
  return (
    <span
      aria-hidden="true"
      className={"pointer-events-none absolute h-3 w-3 border-[#2FA8D8]/70 " + className}
    />
  );
}

/* 画面四角的三角标记 + 沿边工程断线：贴边框走，不压内容 */
function FrameMarks() {
  return (
    <React.Fragment>
      <Corner className="left-0 top-0 border-l border-t" />
      <Corner className="right-0 top-0 border-r border-t" />
      <Corner className="left-0 bottom-0 border-l border-b" />
      <Corner className="right-0 bottom-0 border-r border-b" />
      <span aria-hidden="true" className="pointer-events-none absolute left-6 right-6 top-[3px] h-[3px] tick-strip opacity-50" />
      <span aria-hidden="true" className="pointer-events-none absolute left-6 right-6 bottom-[3px] h-[3px] tick-strip opacity-35" />
      <span aria-hidden="true" className="pointer-events-none absolute left-[3px] top-8 bottom-8 w-[3px] tick-strip-major opacity-30 [writing-mode:vertical-rl]" />
    </React.Fragment>
  );
}

function Divider({ vertical }) {
  return vertical
    ? <span className="mx-1 h-5 w-px bg-[rgba(122,148,172,0.5)]" />
    : <span className="h-px w-full bg-[rgba(122,148,172,0.32)]" />;
}

/* 无画面时的球场线稿底，有帧图后由图片盖住 */
function CourtWireframe() {
  return (
    <svg viewBox="0 0 640 360" className="absolute inset-0 h-full w-full" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <g fill="none" stroke="#2FA8D8" strokeOpacity="0.16" strokeWidth="1">
        <rect x="40" y="30" width="560" height="300" />
        <line x1="320" y1="30" x2="320" y2="330" />
        <circle cx="320" cy="180" r="46" />
        <rect x="40" y="128" width="120" height="104" />
        <path d="M160 150 a34 34 0 0 1 0 60" />
        <circle cx="90" cy="180" r="8" />
        <rect x="480" y="128" width="120" height="104" />
        <path d="M480 150 a34 34 0 0 0 0 60" />
        <circle cx="550" cy="180" r="8" />
      </g>
      <g stroke="#62CDEC" strokeOpacity="0.22" strokeWidth="1" strokeDasharray="4 6">
        <line x1="40" y1="180" x2="600" y2="180" />
      </g>
    </svg>
  );
}

function Toasts({ items }) {
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[340px] flex-col gap-1">
      {items.map((t) => (
        <div
          key={t.id}
          className={
            "glass border px-3 py-2 font-mono text-[11px] leading-relaxed " +
            (t.kind === "error"
              ? "border-[#B3261E]/70 text-[#7C1D17]"
              : t.kind === "warn"
                ? "border-[#C8952B]/70 text-[#7A5610]"
                : "border-[rgba(122,148,172,0.55)] text-[#1B2026]")
          }
        >
          {t.msg}
        </div>
      ))}
    </div>
  );
}

/* ---------------- 1 · 顶部导航 ---------------- */

function TopNav({ onImport, onRefresh, busy }) {
  const [path, setPath] = useState("");
  return (
    <header className="relative z-10 flex h-14 items-center gap-6 border-b border-[rgba(122,148,172,0.45)] bg-white/70 px-4 backdrop-blur-md">
      <div className="flex items-baseline gap-3">
        <span className="h-6 w-[5px] bg-[#2FA8D8]" />
        <h1 className="font-display text-[26px] font-bold uppercase leading-none tracking-[0.02em] text-[#1B2026]">
          BSK Detect
        </h1>
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8A949F]">
          Real-time Basketball Analytics
        </span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") onImport(path, setPath); }}
          spellCheck="false"
          className="h-8 w-[360px] border border-[rgba(122,148,172,0.5)] bg-white/70 px-2 font-mono text-[12px] text-[#1B2026] outline-none focus:border-[#2FA8D8]"
          placeholder="视频路径 (e.g. E:\videos\game.mp4)"
        />
        <button
          disabled={busy}
          onClick={() => onImport(path, setPath)}
          className="btn-ice h-8 px-4 font-display text-[13px] font-semibold uppercase tracking-[0.1em] disabled:opacity-50"
        >
          导入视频
        </button>
        <button title="刷新视频列表" onClick={onRefresh} className="btn-steel h-8 w-8" aria-label="刷新">
          <svg viewBox="0 0 16 16" className="mx-auto h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M13 8a5 5 0 1 1-1.6-3.7M13 2v3h-3" />
          </svg>
        </button>
      </div>
    </header>
  );
}

/* ---------------- 2 · 左侧控制面板 ---------------- */

function VideoList({ videos, active, setActive }) {
  return (
    <section className="mb-4">
      <SectionTitle index="01" zh="视频库" en="library" tail={videos.length + " CLIPS"} />
      <div className="max-h-[236px] space-y-px overflow-y-auto pr-px">
        {videos.length === 0 ? (
          <div className="px-2.5 py-2 font-mono text-[11px] text-[#8A949F]">
            暂无素材 · 在上方填路径导入
          </div>
        ) : null}
        {videos.map((v) => {
          const on = v.vid === active;
          const ready = v.status === "ready";
          return (
            <button
              key={v.vid}
              onClick={() => setActive(v.vid)}
              className={
                "block w-full border-l-2 px-2.5 py-2 text-left transition " +
                (on
                  ? "border-l-[#2FA8D8] bg-[rgba(47,168,216,0.13)]"
                  : "border-l-[rgba(122,148,172,0.4)] hover:border-l-[#2FA8D8]/70 hover:bg-white/60")
              }
            >
              <div className={"truncate text-[12px] leading-snug " + (on ? "font-semibold text-[#1B2026]" : "text-[#4C555F]")}>
                {v.name}
              </div>
              <div className="mt-1 flex items-center gap-3 font-mono text-[10px] tracking-[0.04em] text-[#8A949F]">
                <span>{(v.nframes || 0).toLocaleString()} FR</span>
                <span>{v.fps ? v.fps.toFixed(2) : "--"} FPS</span>
                <span className={ready ? "text-[#2FA8D8]" : "text-[#C8952B]"}>
                  {ready ? "READY" : (v.status || "INDEX").toUpperCase().slice(0, 6)}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function LayerGroup({ on, toggle }) {
  return (
    <section className="mb-4">
      <SectionTitle index="02" zh="图层控制" en="layers" tail={[...on].length + "/" + LAYERS.length} />
      <div className="grid grid-cols-2 gap-px">
        {LAYERS.map((l) => {
          const active = on.has(l.key);
          return (
            <button
              key={l.key}
              onClick={() => toggle(l.key)}
              className={
                "flex h-8 items-center justify-between px-2 text-[12px] transition " +
                (active ? "btn-ice" : "btn-quiet")
              }
            >
              <span className="font-display font-semibold uppercase tracking-[0.08em]">{l.label}</span>
              <span className={"font-mono text-[9px] " + (active ? "text-[#08202C]/70" : "text-[#8A949F]")}>
                {active ? "ON" : "OFF"}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function SliderRow({ label, unit, value, min, max, step, onChange }) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[11px] uppercase tracking-[0.14em] text-[#4C555F]">{label}</span>
        <span className="font-mono text-[13px] font-medium tabular-nums text-[#1B2026]">
          {value}
          <span className="ml-0.5 text-[10px] text-[#8A949F]">{unit}</span>
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="mt-0.5 flex justify-between font-mono text-[9px] text-[#8A949F]">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function DisplaySettings({ width, setWidth, quality, setQuality }) {
  return (
    <section>
      <SectionTitle index="03" zh="显示设置" en="render" />
      <SliderRow label="宽度" unit="px" value={width} min={480} max={1920} step={80} onChange={setWidth} />
      <SliderRow label="质量" unit="q" value={quality} min={40} max={95} step={1} onChange={setQuality} />
      <div className="mt-2 font-mono text-[9px] leading-relaxed text-[#8A949F]">
        宽/质量只影响服务端烧录的这张 JPEG，不改源素材。
      </div>
    </section>
  );
}

function LeftPanel(state) {
  return (
    <aside className="glass relative z-10 min-h-0 overflow-y-auto p-3">
      <VideoList videos={state.videos} active={state.active} setActive={state.setActive} />
      <Divider />
      <div className="pt-3">
        <LayerGroup on={state.layers} toggle={state.toggle} />
      </div>
      <Divider />
      <div className="pt-3">
        <DisplaySettings {...state} />
      </div>
    </aside>
  );
}

/* ---------------- 3 · 中间视频预览 ---------------- */

function Viewer({
  imgRef, frame, total, fps, playing, mode, statusMsg, outW, outH, q, pending,
  view3dOn, onToggle3d, courtModel, setCourtModel, hasFrame,
}) {
  const tc = timecode(frame, fps);
  const res = outW ? outW + "×" + outH : "--";
  return (
    <div className="glass relative z-10 flex min-h-0 flex-1 flex-col p-2">
      <div className="viewer-container scanline relative flex-1 overflow-hidden border border-[rgba(122,148,172,0.55)] bg-[#0E1418]">
        <CourtWireframe />

        {/* 帧图与 3D 画布：id 由 view3d.js 读取，不能改名 */}
        <img
          ref={imgRef}
          id="frame"
          alt=""
          draggable="false"
          className="absolute inset-0 h-full w-full object-contain"
          style={{ display: view3dOn ? "none" : "block", visibility: hasFrame ? "visible" : "hidden" }}
        />
        <canvas id="view3d" />
        <div id="view3d-status" data-kind="info" />

        {/* 左上：帧号 / 总帧数 / 时间码 */}
        <div className="pointer-events-none absolute left-3 top-3 z-10 flex items-center gap-3 font-mono text-[11px] tracking-[0.06em] text-white/85">
          <span className="bg-[rgba(47,168,216,0.18)] px-1.5 py-0.5 text-[#9BE4FF]">
            FRAME {pad(frame, 4)} / {pad(total, 5)}
          </span>
          <span className="text-white/70">{tc}</span>
        </div>

        {/* 右上：3D 切换 + 球场型号 */}
        <button
          onClick={onToggle3d}
          className={
            "absolute right-3 top-3 z-20 h-7 px-3 font-display text-[12px] font-semibold uppercase tracking-[0.16em] " +
            (view3dOn ? "btn-ice" : "btn-steel")
          }
        >
          {view3dOn ? "2D" : "3D"}
        </button>
        <select
          id="court-model-select"
          value={courtModel}
          onChange={(e) => setCourtModel(e.target.value)}
          className={"court-model-select" + (view3dOn ? " visible" : "")}
          title="3D 球场型号"
        >
          {COURT_MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>

        {/* 中央：还没有任何一帧解码成功时的占位 */}
        {!hasFrame ? (
          <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center gap-2">
            <div className="font-display text-[15px] font-semibold uppercase tracking-[0.3em] text-white/25">
              no signal
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/20">
              {mode === "realtime" ? "等待实时管线出帧" : "等待服务端解码出帧"}
            </div>
          </div>
        ) : null}

        {/* 左下：管线状态，不遮挡画面主体 */}
        <div className="pointer-events-none absolute bottom-3 left-3 z-10 flex max-w-[80%] items-center gap-2 font-mono text-[10px] text-white/35">
          <span className={"h-1.5 w-1.5 " + (pending ? "bg-[#C8952B]" : playing ? "bg-[#62CDEC]" : "bg-white/25")} />
          <span className="whitespace-nowrap">
            {pending ? "DECODING" : playing ? "STREAMING" : "IDLE"} · {res} · Q{q} · {mode === "realtime" ? "RT" : "OFFLINE"}
          </span>
          {statusMsg ? <span className="truncate text-white/55">{statusMsg}</span> : null}
        </div>

        <FrameMarks />
      </div>
    </div>
  );
}

/* ---------------- 4 · 底部播放控制栏 ---------------- */

function StepButton({ label, onClick, tone, wide }) {
  return (
    <button
      onClick={onClick}
      title={wide ? label : undefined}
      className={
        "h-7 font-mono text-[11px] tabular-nums whitespace-nowrap transition " +
        (wide ? "min-w-[54px] px-1 text-[10px] " : "w-[30px] px-0 ") +
        (tone === "primary" ? "btn-ice" : "btn-quiet")
      }
    >
      {label}
    </button>
  );
}

function PlaybackBar({
  frame, total, setFrame, playing, onPlayToggle,
  qualityMode, setQualityMode, speed, setSpeed, onShot,
}) {
  const pct = total > 1 ? ((frame - 1) / (total - 1)) * 100 : 0;
  const jump = (d) => setFrame(clamp(frame + d, 1, total));

  return (
    <div className="glass relative z-10 flex shrink-0 flex-wrap items-center gap-1.5 max-[1500px]:gap-1 p-2">
      <button
        onClick={onPlayToggle}
        className={
          "flex h-8 shrink-0 items-center gap-2 px-3 font-display text-[13px] font-semibold uppercase tracking-[0.1em] transition " +
          (playing ? "btn-ice" : "btn-steel")
        }
      >
        {playing ? (
          <svg viewBox="0 0 12 12" className="h-3 w-3" fill="currentColor"><rect x="2" y="1.5" width="3" height="9" /><rect x="7" y="1.5" width="3" height="9" /></svg>
        ) : (
          <svg viewBox="0 0 12 12" className="h-3 w-3" fill="currentColor"><path d="M2 1.5 L11 6 L2 10.5 Z" /></svg>
        )}
        {playing ? "停止" : "播放"}
      </button>

      <div className="flex shrink-0 items-center gap-px">
        {[-30, -10, -1].map((s) => (
          <StepButton key={s} label={(s > 0 ? "+" : "") + s} onClick={() => jump(s)} />
        ))}
        <input
          value={pad(frame, 4)}
          onChange={(e) => {
            const n = Number(e.target.value.replace(/\D/g, ""));
            if (!Number.isNaN(n)) setFrame(clamp(n, 1, total));
          }}
          className="mx-px h-7 w-[62px] border border-[rgba(122,148,172,0.5)] bg-white/70 px-1 text-center font-mono text-[11px] tabular-nums text-[#1B2026] outline-none focus:border-[#2FA8D8]"
        />
        {[1, 10, 30].map((s) => (
          <StepButton key={s} label={"+" + s} onClick={() => jump(s)} />
        ))}
      </div>

      {/* 进度条 */}
      <div className="relative mx-1 h-9 min-w-[110px] flex-1">
        <div className="absolute left-0 right-0 top-1/2 h-[3px] -translate-y-1/2 bg-[rgba(122,148,172,0.4)]" />
        <div className="absolute left-0 right-0 top-1/2 h-[9px] -translate-y-1/2 tick-strip opacity-40" />
        <div
          className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 bg-[#2FA8D8]"
          style={{ width: pct + "%" }}
        />
        <div
          className="absolute top-1/2 h-[18px] w-[3px] -translate-y-1/2 bg-[#62CDEC] shadow-[0_0_0_1px_rgba(8,32,44,0.6)]"
          style={{ left: "calc(" + pct + "% - 1.5px)" }}
        />
        <input
          type="range"
          min={1}
          max={total}
          value={frame}
          onChange={(e) => setFrame(Number(e.target.value))}
          className="absolute inset-0 h-full w-full opacity-0"
          aria-label="时间轴"
        />
      </div>

      <Divider vertical />

      <label className="flex shrink-0 items-center gap-1">
        <span className="text-[9px] uppercase tracking-[0.1em] text-[#4C555F]">质量</span>
        <select
          value={qualityMode}
          onChange={(e) => setQualityMode(e.target.value)}
          className="sharp btn-quiet h-7 pl-2 pr-6 text-[11px]"
        >
          <option value="accurate">准确</option>
          <option value="fast">快速</option>
        </select>
      </label>

      <label className="flex shrink-0 items-center gap-1">
        <span className="text-[9px] uppercase tracking-[0.1em] text-[#4C555F]">速度</span>
        <select value={speed} onChange={(e) => setSpeed(e.target.value)} className="sharp btn-quiet h-7 pl-2 pr-6 font-mono text-[11px]">
          {SPEEDS.map((s) => (
            <option key={s} value={s}>{s}x</option>
          ))}
        </select>
      </label>

      <div className="ml-auto flex shrink-0 items-center gap-px">
        <StepButton wide label="上一镜头" onClick={() => onShot(-1)} />
        <StepButton wide label="下一镜头" onClick={() => onShot(1)} />
      </div>
    </div>
  );
}

/* ---------------- 5 · 右侧面板 ---------------- */

function StatBlock({ label, en, value, unit, sub }) {
  return (
    <div className="flex items-end justify-between border-l-2 border-l-[rgba(122,148,172,0.45)] py-1.5 pl-2.5">
      <div>
        <div className="font-display text-[12px] uppercase tracking-[0.14em] text-[#4C555F]">{label}</div>
        <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-[#8A949F]">{en}</div>
      </div>
      <div className="text-right">
        <div className="font-display text-[30px] font-semibold leading-none tabular-nums text-[#1B2026]">
          {value}
          {unit ? <span className="ml-0.5 font-mono text-[11px] font-normal text-[#8A949F]">{unit}</span> : null}
        </div>
        {sub ? <div className="mt-0.5 font-mono text-[9px] text-[#2FA8D8]">{sub}</div> : null}
      </div>
    </div>
  );
}

function RightPanel({ rt, stats, meta, frame, fps, width, qualityMode, mode, dtMs }) {
  const d3 = (rt && rt["3d"]) || {};
  const objects = (rt && rt.objects) || [];
  const rows = [
    ["时间码", meta ? timecode(frame, fps) : "--"],
    ["帧号 / 总数", meta ? pad(frame, 4) + " / " + pad(meta.nframes, 5) : "--"],
    ["输出分辨率", width + " × " + (meta && meta.height ? Math.round(width * meta.height / meta.width) : "--")],
    ["素材", meta ? meta.width + " × " + meta.height + " · " + (meta.fps ? meta.fps.toFixed(2) : "--") + "fps" : "--"],
    ["管线", mode === "realtime" ? "实时烧录 rt_frame" : "离线渲染 frame.jpg"],
    ["质量档", qualityMode === "accurate" ? "准确 (RTMO)" : "快速 (YOLO)"],
    ["推理引擎", rt && rt.engine ? rt.engine : "--"],
    ["检测对象", rt ? rt.players + (rt.balls ? " + " + rt.balls + " 球" : "") : "--"],
    ["骨架", rt ? rt.poses + " 人" : "--"],
    ["3D", d3.ok ? "启用 " + d3.n + " 人" : "关闭 (" + (d3.why || "disabled") + ")"],
    ["运动约束", d3.prior ? (d3.prior.enabled
      ? "开启 · " + (d3.prior.hit_rate_pct || 0).toFixed(1) + "% 帧修正" : "关闭") : "--"],
    ["运行档", (rt && rt.profile) || "--"],
    ["处理延迟", dtMs ? dtMs.toFixed(0) + " ms" : "--"],
  ];

  return (
    <aside className="glass relative z-10 flex min-h-0 flex-col gap-3 overflow-y-auto p-3">
      <section>
        <SectionTitle
          index="04" zh="实时统计" en="live"
          tail={stats.procFps ? stats.procFps.toFixed(1) + " FPS" : (mode === "realtime" ? "LIVE" : "离线")}
        />
        <div className="space-y-px">
          <StatBlock label="球员" en="players" value={stats.players}
            sub={rt ? (stats.players ? "TRACK OK" : "无检出") : "等待分析"} />
          <StatBlock label="篮球" en="balls" value={stats.balls}
            sub={rt && rt.balls ? "CONF " + (objects.find((o) => o.cat === "ball") || {}).conf : null} />
          <StatBlock label="帧率" en="fps"
            value={stats.playing ? stats.playFps.toFixed(1) : (fps || 0).toFixed(1)} unit="fps" />
        </div>
      </section>

      <Divider />

      <section>
        <SectionTitle index="05" zh="帧数据" en="frame meta" />
        <dl className="space-y-px">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between gap-3 border-b border-[rgba(122,148,172,0.22)] py-1 last:border-b-0">
              <dt className="shrink-0 text-[11px] text-[#4C555F]">{k}</dt>
              <dd className="truncate text-right font-mono text-[11px] tabular-nums text-[#1B2026]">{v}</dd>
            </div>
          ))}
        </dl>
      </section>

      <Divider />

      <section>
        <SectionTitle index="06" zh="检测结果" en="detections" tail={objects.length + " ROWS"} />
        <div className="overflow-x-auto">
          <table className="w-full border-collapse font-mono text-[10px] tabular-nums">
            <thead>
              <tr className="border-b border-[rgba(122,148,172,0.6)] text-left">
                {["ID", "类别", "置信", "框 (x0,y0,x1,y1)", "号码"].map((h) => (
                  <th key={h} className="whitespace-nowrap py-1 pr-2 font-display text-[10px] font-semibold uppercase tracking-[0.08em] text-[#4C555F]">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {objects.length === 0 ? (
                <tr><td colSpan="5" className="py-2 pr-2 text-[#8A949F]">
                  {mode === "realtime" ? "该帧尚无分析缓存" : "离线渲染需要先跑分析任务"}
                </td></tr>
              ) : objects.map((o) => (
                <tr key={o.id} className="border-b border-[rgba(122,148,172,0.2)] last:border-b-0 hover:bg-[rgba(47,168,216,0.10)]">
                  <td className="whitespace-nowrap py-1 pr-2 text-[#2FA8D8]">#{pad(o.id, 2)}</td>
                  <td className="whitespace-nowrap py-1 pr-2 text-[#1B2026]">{o.cat}</td>
                  <td className="whitespace-nowrap py-1 pr-2 text-[#4C555F]">{o.conf.toFixed(2)}</td>
                  <td className="whitespace-nowrap py-1 pr-2 text-[#8A949F]">{o.box.join(",")}</td>
                  <td className="whitespace-nowrap py-1 pr-2 font-semibold text-[#1B2026]">{o.jersey || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </aside>
  );
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

  const [mode, setMode] = useState("realtime");     // realtime | offline
  const modeRef = useRef("realtime");               // 同一份状态的可读快照，避开 setState 批处理带来的竞态
  const pickMode = useCallback((m) => { modeRef.current = m; setMode(m); }, []);
  const [rt, setRt] = useState(null);               // rt_meta.json 摘要
  const [dtMs, setDtMs] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [hasFrame, setHasFrame] = useState(false);
  const [pending, setPending] = useState(false);   // 有一帧正在服务端解码/烧录
  const [toasts, setToasts] = useState([]);
  const [busy, setBusy] = useState(false);

  const [view3dOn, setView3dOn] = useState(false);
  const [courtModel, setCourtModel] = useState("FIBA");

  const imgRef = useRef(null);
  const loop = useRef({ token: 0, playT0: 0, playFrames: 0, loadFails: 0, noFpsCount: 0, playWidth: 1280, reqId: 0 });
  const stats = useRef({ players: 0, balls: 0, playFps: 0, procFps: 0 });
  const playingRef = useRef(0);   // 播放令牌：0 表示未播放，非 0 是本次播放的 token
  const frameRef = useRef(1);     // 播放循环里的权威帧号，避免等 React 提交后才更新
  const toastSeq = useRef(0);
  const [statTick, setStatTick] = useState(0);      // 只用来触发统计区重绘

  const total = meta ? meta.nframes : 1;
  const fps = meta ? meta.fps : 29.97;

  const toast = useCallback((msg, kind) => {
    const id = ++toastSeq.current;
    setToasts((prev) => [...prev.slice(-3), { id, msg, kind: kind || "info" }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4200);
  }, []);

  const layerParam = useCallback(() => [...layers].join(","), [layers]);

  /* ── 帧图 URL ── */
  const want3d = view3dOn ? "&enable_3d=1" : "";
  const rtUrl = useCallback((n) =>
    `/api/videos/${vid}/rt_frame.jpg?n=${n}&layers=${layerParam()}&width=${loop.current.playWidth}` +
    `&q=${q}&quality=${qualityMode}&rt=${Date.now()}${want3d}`,
    [vid, layerParam, q, qualityMode, want3d]);
  const offUrl = useCallback((n) => {
    // 播放时用降档宽度：离线渲染每帧都要从关键帧重解，宽度小一点能明显提速
    const w = playingRef.current ? loop.current.playWidth : width;
    return `/api/videos/${vid}/frame.jpg?n=${n}&layers=${layerParam()}&width=${w}&q=${q}` +
      `&quality=${qualityMode}&realtime=0${want3d}`;
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
      const left = retries === undefined ? (useRealtime ? 1 : 0) : retries;
      // 先离屏解码，成功后才原子替换画面；失败时保留上一帧，不闪黑。
      const next = new Image();
      const cleanup = () => {
        next.onload = null; next.onerror = null;
        img.onload = null; img.onerror = null;
        clearTimeout(timer);
      };
      const finish = (err) => { if (settled) return; settled = true; cleanup(); err ? reject(err) : resolve(); };
      const timer = useRealtime ? setTimeout(() => {
        if (req === loop.current.reqId) finish(new Error("帧请求超时"));
      }, 8000) : 0;
      next.onload = () => {
        if (req !== loop.current.reqId) return;
        img.src = next.src;
        setHasFrame(true);
        setPending(false);
        finish(null);
      };
      next.onerror = () => {
        if (req !== loop.current.reqId) return;
        if (left > 0) {
          cleanup();
          setTimeout(() => loadFrame(n, useRealtime, left - 1).then(resolve, reject), 120);
          return;
        }
        setPending(false);
        finish(new Error(useRealtime ? "实时帧加载失败" : "帧解码失败"));
      };
      setPending(true);
      next.src = useRealtime ? rtUrl(n) : offUrl(n);
    });
  }, [vid, rtUrl, offUrl]);

  /* ── 面板摘要：只读服务端缓存，204 表示该帧还没算过 ── */
  const refreshMeta = useCallback(async (n) => {
    if (!vid || modeRef.current !== "realtime") return;
    try {
      const r = await fetch(`/api/videos/${vid}/rt_meta.json?n=${n}&quality=${qualityMode}`);
      if (!r.ok) return;
      const data = await r.json();
      setRt(data);
      setStatusMsg(data.analysis_error || "");
      stats.current.players = data.players || 0;
      stats.current.balls = data.balls || 0;
      setStatTick((t) => t + 1);
    } catch (_) { /* 面板刷新失败不影响看帧 */ }
  }, [vid, qualityMode]);

  const goto = useCallback((n) => {
    if (!vid) return;
    const clamped = clamp(n, 1, total);
    const useRealtime = modeRef.current === "realtime";
    frameRef.current = clamped;
    setFrame(clamped);
    loadFrame(clamped, useRealtime)
      .then(() => refreshMeta(clamped))
      .catch((e) => {
        if (useRealtime) {
          // 实时档跑不了（缺权重/依赖）不是错误终点：换离线渲染继续出图
          pickMode("offline");
          setStatusMsg("实时管线不可用，已降级离线渲染");
          toast("实时管线不可用（缺模型权重或推理依赖），已降级为离线渲染", "warn");
          loadFrame(clamped, false)
            .then(() => setStatusMsg(""))
            .catch((e2) => setStatusMsg(e2.message));
        } else {
          setStatusMsg(e.message);
        }
      });
  }, [vid, total, loadFrame, refreshMeta, toast, pickMode]);

  /* ── 播放：按墙钟时间推进，1x 目标 30fps；落后时丢帧而不是累积延迟 ── */
  const play = useCallback(() => {
    if (!vid) { toast("请先在视频库选择素材", "error"); return; }
    const useRealtime = modeRef.current === "realtime";
    const sp = parseFloat(speed) || 1;
    const sourceFps = fps || 29.97;
    const targetFps = Math.min(30, sourceFps);
    const outputInterval = 1000 / targetFps;
    const sourceRate = sourceFps * sp;
    const sourceStep = sourceRate / targetFps;
    const L = loop.current;
    L.token += 1;
    // 播放默认 960；离线档每帧重解 GOP，降到 480 并串行请求。
    L.playWidth = Math.min(width, useRealtime ? 960 : 480);
    L.loadFails = 0; L.noFpsCount = 0; L.playFrames = 0;
    L.playT0 = performance.now();
    const playStartFrame = frameRef.current;
    setPlaying(true);
    setStatusMsg(useRealtime
      ? `实时播放 ${sp}x · 目标 ${Math.round(targetFps)}fps`
      : "离线逐帧：每帧重解 GOP，实际帧率见右侧统计");
    if (!useRealtime) {
      toast("离线档本机单帧解码约需 1 分钟，播放会非常慢；补上实时权重后走 rt_frame 才是设计路径", "warn");
    }

    const tick = async () => {
      if (!playingRef.current || L.token !== playingRef.current) return;
      const t0 = performance.now();
      // 请求当前时间对应的源帧；浏览器或服务端落后时直接跳过中间帧。
      const dueFrame = playStartFrame + (t0 - L.playT0) / 1000 * sourceRate;
      const at = clamp(Math.max(frameRef.current, Math.round(dueFrame)), 1, total);
      try {
        await loadFrame(at, useRealtime);
        L.loadFails = 0;
      } catch (e) {
        L.loadFails += 1;
        if (L.loadFails <= 2) { setTimeout(tick, 180); return; }
        L.loadFails = 0;
        const skipped = clamp(Math.round(at + sourceStep), 1, total);
        frameRef.current = skipped;
        setFrame(skipped);
        if (skipped >= total) { stopPlay(); return; }
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
      stats.current.playFps = L.playFrames / wall;
      setStatTick((t) => t + 1);

      if (L.playFrames % 10 === 0) refreshMeta(at);

      if (useRealtime) {
        if (dt > outputInterval * 1.15) L.noFpsCount += 1; else L.noFpsCount = 0;
        if (L.noFpsCount >= 8 && L.playWidth > 480) {
          L.playWidth = Math.max(480, L.playWidth - 160);
          L.noFpsCount = 0;
          toast("帧耗时偏高，已降档到 " + L.playWidth + " 宽保 30fps", "info");
        }
      }

      const next = clamp(Math.round(at + sourceStep), 1, total);
      if (next >= total) { stopPlay(); return; }
      frameRef.current = next;
      setFrame(next);
      // 对齐固定输出时隙，避免 setTimeout 误差逐帧累积。
      const slot = Math.floor((performance.now() - L.playT0) / outputInterval) + 1;
      const wait = Math.max(0, L.playT0 + slot * outputInterval - performance.now());
      setTimeout(tick, wait);
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

  const selectVideo = useCallback(async (id) => {
    setVid(id);
    setHasFrame(false);
    setRt(null);
    try {
      const m = await api(`/api/videos/${id}`);
      setMeta(m);
      frameRef.current = 1;
      setFrame(1);
      loop.current.noFpsCount = 0;
      stats.current = { players: 0, balls: 0, playFps: 0, procFps: 0 };
      try {
        const s = await api(`/api/videos/${id}/shots`);
        setShots(s.shots || []);
      } catch (_) { setShots([]); }
      if (window.BSK3D) window.BSK3D.setVideo(id, { quality: qualityMode, fps: m.fps });
      toast("已选择: " + m.name, "success");
    } catch (e) {
      toast("读取素材信息失败: " + e.message, "error");
    }
  }, [qualityMode, toast]);

  const importVideo = useCallback(async (path, setPath) => {
    if (!path || !path.trim()) { toast("请输入视频路径", "error"); return; }
    setBusy(true);
    try {
      await api("/api/videos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path.trim() }),
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
  const onShot = useCallback((dir) => {
    if (!shots.length) { toast("该素材还没有镜头切分结果", "warn"); return; }
    const cur = shots.findIndex((s) => frame >= s.start_frame && frame <= s.end_frame);
    if (dir < 0) goto(cur > 0 ? shots[cur - 1].start_frame : 1);
    else goto(cur >= 0 && cur < shots.length - 1 ? shots[cur + 1].start_frame : total);
  }, [shots, frame, total, goto, toast]);

  /* ── 3D ── */
  const toggle3D = useCallback(() => {
    const B = window.BSK3D;
    if (!B) { toast("3D 模块未加载（src/view3d.js 缺失）", "error"); return; }
    if (!B.available) { toast("3D 不可用：" + (B.error || "缺少 Three.js"), "error"); return; }
    const on = !view3dOn;
    setView3dOn(on);
    if (on) B.init();
    B.setVideo(vid, { quality: qualityMode, fps: fps });
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

  useEffect(() => { if (window.BSK3D) window.BSK3D.setQuality(qualityMode); }, [qualityMode]);
  useEffect(() => { if (window.BSK3D && view3dOn) window.BSK3D.setCourtModel(courtModel); }, [courtModel, view3dOn]);
  useEffect(() => { if (view3dOn && window.BSK3D && !playingRef.current) window.BSK3D.showFrame(frame); }, [frame, view3dOn]);

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
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "ArrowLeft") goto(frameRef.current - (e.shiftKey ? 10 : 1));
      else if (e.key === "ArrowRight") goto(frameRef.current + (e.shiftKey ? 10 : 1));
      else if (e.key === " ") { e.preventDefault(); playingRef.current ? stopPlay() : play(); }
      else if (e.key === "[") onShot(-1);
      else if (e.key === "]") onShot(1);
      else if (e.key === "v" || e.key === "V") toggle3D();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [goto, play, stopPlay, onShot, toggle3D]);

  const toggle = (k) =>
    setLayers((prev) => {
      const next = new Set(prev);
      next.has(k) ? next.delete(k) : next.add(k);
      return next;
    });

  const outH = meta && meta.width ? Math.round(width * meta.height / meta.width) : 0;

  return (
    <div className="dot-grid h-full overflow-auto bg-[#E9EDF1] font-sans text-[#1B2026]">
      {/* B 端调试台按 1180px 起排；更窄的窗口横向滚动，避免三栏被压变形 */}
      <div className="flex h-full min-w-[1180px] flex-col">
        <TopNav onImport={importVideo} onRefresh={refreshVideos} busy={busy} />

        <main className="relative z-10 grid min-h-0 flex-1 grid-cols-[248px_minmax(0,1fr)_312px] max-[1500px]:grid-cols-[224px_minmax(0,1fr)_288px] gap-3 p-3">
          <LeftPanel
            videos={videos}
            active={vid}
            setActive={selectVideo}
            layers={layers}
            toggle={toggle}
            width={width}
            setWidth={setWidth}
            quality={q}
            setQuality={setQ}
          />

          <section className="flex min-h-0 min-w-0 flex-col gap-3">
            <Viewer
              imgRef={imgRef}
              frame={frame}
              total={total}
              fps={fps}
              playing={playing}
              mode={mode}
              statusMsg={statusMsg}
              outW={width}
              outH={outH}
              q={q}
              pending={pending}
              view3dOn={view3dOn}
              onToggle3d={toggle3D}
              courtModel={courtModel}
              setCourtModel={setCourtModel}
              hasFrame={hasFrame}
            />
            <PlaybackBar
              frame={frame}
              total={total}
              setFrame={(n) => { if (playingRef.current) stopPlay(); goto(n); }}
              playing={playing}
              onPlayToggle={() => (playingRef.current ? stopPlay() : play())}
              qualityMode={qualityMode}
              setQualityMode={setQualityMode}
              speed={speed}
              setSpeed={setSpeed}
              onShot={onShot}
            />
          </section>

          <RightPanel
            rt={rt}
            stats={{ ...stats.current, playing: playing }}
            meta={meta}
            frame={frame}
            fps={fps}
            width={width}
            qualityMode={qualityMode}
            mode={mode}
            dtMs={dtMs}
            statTick={statTick}
          />
        </main>
      </div>

      <Toasts items={toasts} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
