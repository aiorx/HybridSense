# 工程验证 / Engineering validation

本记录区分可复现的工程检查与作者归因效果评测。测试示例没有作者标签，不用于计算识别准确率、误报率、召回率或校准误差。

## 发布检查：2026-10-20

| 检查 | 结果与范围 |
| --- | --- |
| Python 回归 | **42 项通过**；使用 `models/` 发布资产，覆盖六语言真实分类器、真实 MCP、特征对齐、源码不执行、计划与证据门禁、通用对话、意图路由、多个代码块、AST 边界、Unicode 偏移、覆盖率与重叠计权、取消与恢复 |
| 推理资产完整性 | 六套模型与特征顺序表齐全；24 个资产文件的 SHA-256 与清单一致；JSON 顺序保持、原始 CSV 表头兼容、重复列拒绝；发布资产中无训练 CSV |
| 前端生产构建 | TypeScript 检查与 Vite 生产构建通过 |
| Python 依赖 | `uv pip check`：126 个已安装包兼容 |

复现命令（先按 README 安装环境并准备 CodeBERTa）：

```powershell
$env:HYBRIDSENSE_MODEL_ROOT = 'models'
.venv\Scripts\python.exe -m pytest tests -q
uv pip check --python .venv\Scripts\python.exe
cd frontend
npm run build
```

模型与 MCP 测试调用实际分类器和本地编码器；智能体单元测试使用可控的测试模型，无需供应商 API Key。

## 已完成的本地集成验收

以下为此前在 Windows、Microsoft Edge 与配置好的 DeepSeek 连接上完成的验收，**不是每次单元测试都会执行的在线检查**：

- 普通聊天与后续回合记忆；明确要求解释附件时读取源码并解释；无文字上传 Python 与 TypeScript 时自动进入 MCP 检测并生成证据报告。
- 真实 DeepAgents 计划、工具调用、证据读取与门禁；保存检测证据后重试报告，可复用证据，避免重复推理。
- API 拒绝空源码、超大文件与非 UTF-8 输入；支持取消、恢复、Markdown 与 JSON 导出，并记录编码器 revision。
- 浏览器验证代码粘贴、TypeScript 文件上传、多附件增删、片段排序与源码预览、行定位、筛选、报告、历史与刷新恢复。
- 桌面和 390 px 移动端页面未出现横向溢出或未捕获 JavaScript 错误。
- `start.cmd` 已在 CMD 中启动成功；避免直接调用 `.ps1` 时因文件关联而仅打开编辑器。

可重复执行的集成脚本：`scripts/api_smoke.py`、`scripts/accept_v2.py`、`scripts/browser_v2.cjs`。后两项使用真实供应商连接，会消耗 API 额度；浏览器脚本需要 Playwright、Microsoft Edge，以及在线验收生成的示例报告。

私人会话、任务标识、源码历史、检查点与原始机器可读验收记录留在本地 `runtime/`，不随仓库提交。公开截图 `docs/images/workspace.png` 与 `docs/images/hotspots.png` 分别展示通用欢迎界面和合成示例的片段复核视图。

## 科学解释的边界

整体与片段输出是**未校准的 AI 生成倾向分数**；跨文件汇总与局部计权分数不是 AI 编写行数占比。模型成员间的分数范围不是统计置信区间。工程阈值尚未经独立验证集选择，函数与窗口定位属于探索性分析。

正式报告效果需要独立、有作者来源标签的数据，按仓库、作者与生成模型隔离，报告误报率、召回率、校准误差、语言分层与混写/重构条件。详见 [RESEARCH.md](RESEARCH.md)。
