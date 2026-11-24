/* BSK Detect · 实时篮球视频分析调试台 —— 静态 UI（React + Tailwind）
   只做界面还原：数据全部是写死的样例，没有取流、解码与推理逻辑。
   图层开关 / 滑块 / 播放态属于纯前端交互状态，方便评审界面反馈。 */

const { useState } = React;

/* ---------------- 静态样例数据 ---------------- */

const VIDEOS = [
  { vid: "51", title: "阵前再亮旧时剑，哈登对阵步行者51分集锦.mp4", frames: 8455, fps: 59.94, state: "ready" },
  { vid: "47", title: "末节 13-0 反超：勇士 vs 湖人 关键回合.mp4", frames: 6120, fps: 29.97, state: "ready" },
  { vid: "33", title: "挡拆外弹解析：约基奇低位策应 12 例.mp4", frames: 4380, fps: 30.0, state: "indexing" },
  { vid: "12", title: "快攻推进路线：雷霆转换进攻片段.mp4", frames: 2960, fps: 59.94, state: "ready" },
];

const LAYERS = [
  { key: "hud", label: "HUD" },
  { key: "boxes", label: "检测框" },
  { key: "pose", label: "姿态" },
  { key: "floor", label: "地板" },
  { key: "motion", label: "运动" },
];

const FRAME_ROWS = [
  ["时间码", "00:00.017"],
  ["帧号 / 总数", "0001 / 08455"],
  ["输出分辨率", "1280 × 720"],
  ["机位", "CAM-01 · 底线高机位"],
  ["标定残差", "3.42 px"],
  ["推理耗时", "det 6.1 / pose 4.3 ms"],
  ["检测目标", "11（球员 10 · 球 1）"],
  ["号码识别", "已发布 6 / 10"],
];

const DET_ROWS = [
  ["#01", "player", "0.97", "812,168,906,402", "23"],
  ["#02", "player", "0.95", "640,152,738,396", "11"],
  ["#03", "player", "0.93", "994,174,1088,408", "7"],
  ["#04", "player", "0.91", "452,160,548,392", "30"],
  ["#05", "player", "0.88", "1140,180,1232,410", "—"],
  ["#06", "ball", "0.86", "704,236,738,270", "—"],
];

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

/* ---------------- 1 · 顶部导航 ---------------- */

function TopNav({ onImport }) {
  const [path, setPath] = useState("E:\\videos\\harden_vs_walkers_51.mp4");
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
          spellCheck="false"
          className="h-8 w-[360px] border border-[rgba(122,148,172,0.5)] bg-white/70 px-2 font-mono text-[12px] text-[#1B2026] outline-none focus:border-[#2FA8D8]"
          placeholder="视频路径 (e.g. E:\videos\game.mp4)"
        />
        <button onClick={onImport} className="btn-ice h-8 px-4 font-display text-[13px] font-semibold uppercase tracking-[0.1em]">
          导入视频
        </button>
        <button title="刷新列表" className="btn-steel h-8 w-8" aria-label="刷新">
          <svg viewBox="0 0 16 16" className="mx-auto h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M13 8a5 5 0 1 1-1.6-3.7M13 2v3h-3" />
          </svg>
        </button>
      </div>
    </header>
  );
}

/* ---------------- 2 · 左侧控制面板 ---------------- */

function VideoList({ active, setActive }) {
  return (
    <section className="mb-4">
      <SectionTitle index="01" zh="视频库" en="library" tail={VIDEOS.length + " CLIPS"} />
      <div className="max-h-[236px] space-y-px overflow-y-auto pr-px">
        {VIDEOS.map((v) => {
          const on = v.vid === active;
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
                {v.title}
              </div>
              <div className="mt-1 flex items-center gap-3 font-mono text-[10px] tracking-[0.04em] text-[#8A949F]">
                <span>{v.frames.toLocaleString()} FR</span>
                <span>{v.fps.toFixed(2)} FPS</span>
                <span className={v.state === "ready" ? "text-[#2FA8D8]" : "text-[#C8952B]"}>
                  {v.state === "ready" ? "READY" : "INDEX"}
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
    </section>
  );
}

function LeftPanel(state) {
  return (
    <aside className="glass relative z-10 min-h-0 overflow-y-auto p-3">
      <VideoList active={state.active} setActive={state.setActive} />
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

function Viewer({ frame, total, playing }) {
  const tc = "00:00." + String(Math.round((frame / 59.94) * 1000)).padStart(3, "0");
  return (
    <div className="glass relative z-10 flex min-h-0 flex-1 flex-col p-2">
      <div className="scanline relative flex-1 overflow-hidden border border-[rgba(122,148,172,0.55)] bg-[#0E1418]">
        <CourtWireframe />

        {/* 左上：帧号 / 总帧数 / 时间码 */}
        <div className="absolute left-3 top-3 flex items-center gap-3 font-mono text-[11px] tracking-[0.06em] text-white/85">
          <span className="bg-[rgba(47,168,216,0.18)] px-1.5 py-0.5 text-[#9BE4FF]">
            FRAME {String(frame).padStart(4, "0")} / {String(total).padStart(5, "0")}
          </span>
          <span className="text-white/70">{tc}</span>
        </div>

        {/* 右上：3D 按钮 */}
        <button className="btn-steel absolute right-3 top-3 h-7 px-3 font-display text-[12px] font-semibold uppercase tracking-[0.16em]">
          3D
        </button>

        {/* 中央：静态占位（无取流） */}
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <div className="font-display text-[15px] font-semibold uppercase tracking-[0.3em] text-white/25">
            no signal
          </div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/20">
            静态界面 · 未接入实时管线
          </div>
        </div>

        {/* 左下：极弱的状态刻度，不遮挡画面主体 */}
        <div className="absolute bottom-3 left-3 flex items-center gap-2 font-mono text-[10px] text-white/35">
          <span className={"h-1.5 w-1.5 " + (playing ? "bg-[#62CDEC]" : "bg-white/25")} />
          {playing ? "STREAMING @ 1280×720" : "IDLE · 1280×720 · Q72"}
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

function PlaybackBar({ frame, total, setFrame, playing, setPlaying }) {
  const pct = ((frame - 1) / (total - 1)) * 100;
  const [quality, setQuality] = useState("accurate");
  const [speed, setSpeed] = useState("1");
  const jump = (d) => setFrame(Math.min(total, Math.max(1, frame + d)));

  return (
    <div className="glass relative z-10 flex shrink-0 flex-wrap items-center gap-1.5 max-[1500px]:gap-1 p-2">
      <button
        onClick={() => setPlaying(!playing)}
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
          value={String(frame).padStart(4, "0")}
          onChange={(e) => {
            const n = Number(e.target.value.replace(/\D/g, ""));
            if (!Number.isNaN(n)) setFrame(Math.min(total, Math.max(1, n)));
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
        <select value={quality} onChange={(e) => setQuality(e.target.value)} className="sharp btn-quiet h-7 pl-2 pr-6 text-[11px]">
          <option value="accurate">准确</option>
          <option value="fast">快速</option>
        </select>
      </label>

      <label className="flex shrink-0 items-center gap-1">
        <span className="text-[9px] uppercase tracking-[0.1em] text-[#4C555F]">速度</span>
        <select value={speed} onChange={(e) => setSpeed(e.target.value)} className="sharp btn-quiet h-7 pl-2 pr-6 font-mono text-[11px]">
          {["0.25", "0.5", "1", "2", "4"].map((s) => (
            <option key={s} value={s}>{s}x</option>
          ))}
        </select>
      </label>

      <div className="ml-auto flex shrink-0 items-center gap-px">
        <StepButton wide label="上一镜头" onClick={() => jump(-120)} />
        <StepButton wide label="下一镜头" onClick={() => jump(120)} />
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

function RightPanel() {
  return (
    <aside className="glass relative z-10 flex min-h-0 flex-col gap-3 overflow-y-auto p-3">
      <section>
        <SectionTitle index="04" zh="实时统计" en="live" tail="30.0 FPS" />
        <div className="space-y-px">
          <StatBlock label="球员" en="players" value="10" sub="TRACK OK" />
          <StatBlock label="篮球" en="balls" value="1" sub="CONF 0.86" />
          <StatBlock label="帧率" en="fps" value="59.9" unit="fps" />
        </div>
      </section>

      <Divider />

      <section>
        <SectionTitle index="05" zh="帧数据" en="frame meta" />
        <dl className="space-y-px">
          {FRAME_ROWS.map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between gap-3 border-b border-[rgba(122,148,172,0.22)] py-1 last:border-b-0">
              <dt className="shrink-0 text-[11px] text-[#4C555F]">{k}</dt>
              <dd className="truncate text-right font-mono text-[11px] tabular-nums text-[#1B2026]">{v}</dd>
            </div>
          ))}
        </dl>
      </section>

      <Divider />

      <section>
        <SectionTitle index="06" zh="检测结果" en="detections" tail={DET_ROWS.length + " ROWS"} />
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
              {DET_ROWS.map((r) => (
                <tr key={r[0]} className="border-b border-[rgba(122,148,172,0.2)] last:border-b-0 hover:bg-[rgba(47,168,216,0.10)]">
                  <td className="whitespace-nowrap py-1 pr-2 text-[#2FA8D8]">{r[0]}</td>
                  <td className="whitespace-nowrap py-1 pr-2 text-[#1B2026]">{r[1]}</td>
                  <td className="whitespace-nowrap py-1 pr-2 text-[#4C555F]">{r[2]}</td>
                  <td className="whitespace-nowrap py-1 pr-2 text-[#8A949F]">{r[3]}</td>
                  <td className="whitespace-nowrap py-1 pr-2 font-semibold text-[#1B2026]">{r[4]}</td>
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
  const [active, setActive] = useState("51");
  const [layers, setLayers] = useState(new Set(["hud", "boxes", "pose"]));
  const [width, setWidth] = useState(1280);
  const [quality, setQuality] = useState(72);
  const [frame, setFrame] = useState(1);
  const [playing, setPlaying] = useState(false);

  const total = (VIDEOS.find((v) => v.vid === active) || VIDEOS[0]).frames;
  const toggle = (k) =>
    setLayers((prev) => {
      const next = new Set(prev);
      next.has(k) ? next.delete(k) : next.add(k);
      return next;
    });

  return (
    <div className="dot-grid h-full overflow-auto bg-[#E9EDF1] font-sans text-[#1B2026]">
      {/* B 端调试台按 1180px 起排；更窄的窗口横向滚动，避免三栏被压变形 */}
      <div className="flex h-full min-w-[1180px] flex-col">
        <TopNav />

        <main className="relative z-10 grid min-h-0 flex-1 grid-cols-[248px_minmax(0,1fr)_312px] max-[1500px]:grid-cols-[224px_minmax(0,1fr)_288px] gap-3 p-3">
          <LeftPanel
            active={active}
            setActive={setActive}
            layers={layers}
            toggle={toggle}
            width={width}
            setWidth={setWidth}
            quality={quality}
            setQuality={setQuality}
          />

          <section className="flex min-h-0 min-w-0 flex-col gap-3">
            <Viewer frame={frame} total={total} playing={playing} />
            <PlaybackBar
              frame={frame}
              total={total}
              setFrame={setFrame}
              playing={playing}
              setPlaying={setPlaying}
            />
          </section>

          <RightPanel />
        </main>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
