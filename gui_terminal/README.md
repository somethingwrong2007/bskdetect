# gui_terminal · 实时分析调试台前端

React + Tailwind 的 B 端调试台界面，浅色冷调机能风、全站直角。这是仓库的**正式前端**：
`bsk/server.py` 的 `GET /` 就返回本目录的 `index.html`（旧的 `gui/` 已移除）。

界面本身零构建：运行时依赖本地 vendor，不需要 Node，也不需要浏览器端 Babel。

## 打开

```bash
# 正常用法：起后端，前端由后端提供
.venv/Scripts/python.exe -u -m uvicorn bsk.server:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000/

# 只想看界面壳（所有接口 404，画面区显示 no signal）
python -m http.server 8010
# 打开 http://127.0.0.1:8010/gui_terminal/，或直接双击 gui_terminal/index.html
```

## 用到的接口

| 用途 | 接口 |
|---|---|
| 视频库 / 导入 | `GET /api/videos`、`POST /api/videos`、`GET /api/videos/{vid}` |
| 镜头切分 | `GET /api/videos/{vid}/shots` |
| 实时帧（主路径） | `GET /api/videos/{vid}/rt_frame.jpg?n&layers&width&q&quality&rt` |
| 实时摘要（右侧面板） | `GET /api/videos/{vid}/rt_meta.json?n&quality`（无缓存返回 204） |
| 3D 骨架 | `GET /api/videos/{vid}/rt_3d?n&quality&court_model`，由 `src/view3d.js` 轮询 |
| 离线渲染（降级路径） | `GET /api/videos/{vid}/frame.jpg?...&realtime=0` |

## 两条管线与自动降级

启动时先探一次 `rt_frame.jpg`：

- 正常 → **实时档**：逐帧请求服务端烧录叠加图，播放按源帧率推进，帧耗时持续偏高时自动降播放宽度
  （960 起，每档 160，最低 800），右侧统计来自 `rt_meta.json`。
- 探测失败（缺 `data/models/rtmo_s_coco.onnx` / `rtmdet_ins_s_640.onnx`，或没装推理依赖）→ **离线档**：
  改走 `frame.jpg?realtime=0`，界面左下角标 `OFFLINE`，并 toast 说明原因。检测/骨架/3D 在离线档没有数据，
  面板显示 `--` 而不是假数字。

本机实测（CPU、无实时权重）：离线档单帧首次请求约 68–84 秒（要从关键帧重解 + 烧录叠加），
同一帧第二次命中服务端图像缓存只要 4 毫秒。所以离线适合逐帧核对，不适合播放；
点播放会明确提示很慢、实际帧率见右侧统计。

## 交互

- 播放 / 停止、-30/-10/-1/+1/+10/+30、帧号输入、时间轴拖拽
- 上一镜头 / 下一镜头（按 `shots.json` 的 `start_frame` 跳）
- 图层 HUD / 检测框 / 姿态 / 地板 / 运动：切换后立刻重画当前帧
- 宽度 480–1920（步进 80）、JPEG 质量 40–95、质量档 准确/快速、速度 0.25x–4x
- 3D 按钮切 `#view3d` 画布（拖拽旋转 / 滚轮缩放 / 双击复位），球场型号下拉 FIBA / NBA / 3x3
- 键盘：`←/→` 单帧、`Shift+←/→` 十帧、空格 播放/停止、`[` `]` 镜头、`v` 切 3D
- 深链：`/?vid=51&frame=2060&view=3d&court=FIBA`

## 目录与构建

```
index.html            Tailwind 配置（corePlugins.borderRadius=false）+ React/ReactDOM + three + view3d + App.js
src/App.jsx           界面与数据层，唯一需要改的文件
src/App.js            App.jsx 的预编译产物，由 build.py 生成，勿手改
src/view3d.js         3D 复刻视图（原 gui/view3d.js，靠 window.BSK3D 暴露接口）
src/styles.css        波点纹理、玻璃面板、钢蓝按钮、工程刻度尺、滑块与下拉直角化
src/fonts.css         Oswald / JetBrains Mono 的 @font-face
fonts/                两个 latin 子集 woff2 + 各自 OFL-1.1 原文
vendor/               react / react-dom / tailwind / three.min.js
build.py              App.jsx → App.js 预编译，带 --check
```

改界面：改 `src/App.jsx` → `python gui_terminal/build.py`（首次会下载约 3MB 的
@babel/standalone，已 gitignore，只在构建期存在）→ 刷新页面。
`python gui_terminal/build.py --check` 可校验产物与源码一致，适合挂 CI。
装好 Node 后可以整段丢掉：`src/App.jsx` 直接进 `frontend/` 的 Vite 项目，删掉 `vendor/` 与 `build.py`。

## 已知边界

- `stream.mjpeg` 在本仓库当前实现里是坏的（`stream_mjpeg` 函数体里有一句游离的 `yield`，
  使整个视图成了生成器，FastAPI 只把收尾边界序列化返回）。本前端不使用它，播放走逐帧请求。
- 检测结果表的「号码」列读 `objects[].jersey`，字段来自球衣号码识别那条 PR；没合并前显示 `—`。
- 3D 依赖实时管线写入的缓存，离线档下会直接提示「离线渲染没有 3D 数据」。
- Tailwind 用的是 Play CDN 脚本（浏览器端 JIT），官方定位原型用途；正式工程化换成构建期 Tailwind。
