# 依赖与许可

本目录自研代码（`index.html`、`src/`、`build.py`、文档）随仓库采用 **Apache-2.0**，
与仓库根 `LICENSE` 一致。第三方资产如下：

| 资产 | 版本 | 许可 | 位置 | 是否入库 |
|---|---|---|---|---|
| React | 18.3.1 | MIT | `vendor/react.production.min.js` | 是 |
| React DOM | 18.3.1 | MIT | `vendor/react-dom.production.min.js` | 是 |
| Tailwind CSS（Play CDN 构建） | 3.4.17 | MIT | `vendor/tailwind.js` | 是 |
| Babel Standalone | 7.26.7 | MIT | `vendor/babel.min.js` | 否，仅构建期下载，已 `.gitignore` |
| Oswald（可变字体 latin 子集） | Google Fonts 当前版 | SIL OFL 1.1 | `fonts/oswald-latin.woff2` | 是 |
| JetBrains Mono（可变字体 latin 子集） | Google Fonts 当前版 | SIL OFL 1.1 | `fonts/jetbrainsmono-latin.woff2` | 是 |

字体原文许可证：`fonts/OFL-Oswald.txt`、`fonts/OFL-JetBrainsMono.txt`。
Oswald 版权归 The Oswald Project Authors，JetBrains Mono 版权归 The JetBrains Mono Project Authors；
两者均按 OFL-1.1 允许嵌入与再分发，前提是保留许可证文本且不对字体本身改名，本目录满足该条件。

三个 JS 依赖都是 MIT，仓库无 AGPL/GPL 依赖。Tailwind Play 脚本本质是浏览器端 JIT，
官方定位是原型/开发用途；正式工程化请按 `README.md` 的「迁进 frontend/ 的做法」换成
构建期 Tailwind，届时 `vendor/` 整个目录删除。
