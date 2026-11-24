# gui_terminal · 实时分析调试台（静态 UI 概念稿）

React + Tailwind 写的**纯界面**调试台，风格是浅色冷调机能风（参考明日方舟罗德岛终端）。
和 `gui/` 的关系：`gui/` 是接后端的生产前端，这里是独立的概念稿，**不取流、不解码、不推理**，
所有帧号、统计、检测表都是写死的样例数据。

## 打开方式

三种都行，任选：

```bash
# 1) 直接双击 gui_terminal/index.html（已验证 file:// 可用，无需服务器）

# 2) 仓库根起一个静态服务
python -m http.server 8010 --bind 127.0.0.1
#    打开 http://127.0.0.1:8010/gui_terminal/

# 3) 挂进现有 FastAPI（想和后端同域时用，需要在 bsk/server.py 加一行）
#    app.mount("/gui_terminal", StaticFiles(directory="gui_terminal", html=True), name="gui_terminal")
```

不需要 Node，不需要 `npm install`，运行时依赖已经本地 vendor 在 `vendor/`。

## 目录

```
index.html            入口：Tailwind 配置（全站禁用圆角）+ React/ReactDOM + src/App.js
src/App.jsx           界面源码（唯一需要改的文件）
src/App.js            App.jsx 的预编译产物，由 build.py 生成，勿手改
src/styles.css        波点纹理、玻璃面板、钢蓝按钮、工程刻度尺、滑块与下拉的直角化
src/fonts.css         Oswald / JetBrains Mono 的 @font-face（可变字体，声明 200–700 全字重）
fonts/                两个 woff2 子集 + 各自的 OFL-1.1 许可证原文
vendor/               react / react-dom 生产版 UMD + tailwind Play（合计约 549KB）
build.py              App.jsx → App.js 的预编译脚本
LICENSES.md           依赖与字体许可
```

## 改界面

改 `src/App.jsx`，然后：

```bash
python gui_terminal/build.py      # 重新生成 src/App.js
python gui_terminal/build.py --check   # 只校验产物与源码是否一致（CI 可用）
```

编译走 headless Chrome 执行 Babel standalone（首次会自动下载约 3MB，已 gitignore，
只在构建期存在）。这样仓库里不用放 3MB 的浏览器端编译器，页面也不需要运行时 Babel。
装好 Node 之后可以整段丢掉：`src/App.jsx` 直接进 Vite 项目即可（见下）。

## 视觉规范落点

- 底色 `#E9EDF1` 浅灰白 + 极低透明度像素波点纹理（`.dot-grid::before`，4px 网格、透明度 .34）
- 面板：`rgba(255,255,255,.58)` + `backdrop-filter: blur(7px) saturate(1.2)` + 0.65px 冷蓝描边
- 按钮：炭灰/冷蓝两档实心，微弱上沿高光；激活态冰蓝高亮
- 全部直角：Tailwind `corePlugins.borderRadius = false`，自定义 CSS 里 `border-radius: 0`
- 栅格：`grid-cols-[248px_minmax(0,1fr)_312px]`，窄屏（≤1500px）收成 `224px / 1fr / 288px`
- 字重对比：Oswald 标题 + JetBrains Mono 数字 + Inter 正文，字号层级靠字重与间距而非装饰
- 画面区四角冰蓝角标与上下工程刻度条都是 `pointer-events-none`，不遮挡画面

## 迁进 frontend/ 的做法

仓库根已有 `frontend/`（Vite + TS，目前只有 package.json）。要正式工程化时：

```
把 src/App.jsx 放进 frontend/src/，index.html 换成 Vite 入口
npm i -D tailwindcss postcss autoprefixer && npx tailwindcss init -p
npm i react react-dom
tailwind.config.js 里加 corePlugins: { borderRadius: false } 与本文件同款的 fontFamily
删掉 vendor/、build.py、src/App.js（Vite 自己编译 JSX）
```

## 已验证

- 1920×1080 / 1600×900 / 1366×768 三档 headless 截图，五块模块位置与参考结构一致，控制栏单行不换行
- 计算样式扫描：非零 `border-radius` 元素 **0 个**；玻璃面板与波点纹理参数生效
- 交互态：地板图层 OFF→ON、播放→停止（冰蓝高亮）、+10 后帧号变 `FRAME 0011 / 08455`、
  宽度滑杆联动 `1280px` 与 `Q72` 标记
- 预编译产物与浏览器内编译结果逐字节一致（169 处 `React.createElement`，无残留 JSX）
