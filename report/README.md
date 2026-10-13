# 第二次作业报告

- `模式识别第二次作业-实验报告.docx`：正式 Word 报告；公式为 MathType OLE 对象，表格为三线表。
- `模式识别第二次作业-实验报告.pdf`：使用 Microsoft Word 导出的同版 PDF。
- `homework2_report.md`：正文可读源文件；具体实验数据、图表来自 `../results/homework2/`。

第一次报告保留在本地原路径，未修改。第二次沿用其封面、A4 页面、页边距及主要字体格式。

## 重新构建

实验 Notebook 使用项目 `.venv` 与根目录 `requirements.txt`。报告构建另需 Python 的 `python-docx`、Pillow，以及 Windows 上的 Microsoft Word、MathType 桌面版；这些是文档工具，不是运行 Notebook 的依赖。

1. 执行 `tools/build_homework2_report.py`，读取 Markdown、图片以及本地第一次报告封面，生成 Word 排版和 `tmp/homework2_report/equations.json`。若没有本地第一次报告，使用简化封面。
2. 执行 `tools/homework2_mathtype.ps1`，根据 JSON 将标记替换成可编辑 MathType 公式，保存 Word 并导出 PDF。该步骤依赖本机 MathType 安装路径，具体参数见脚本。
3. 检查导出 PDF 的每一页，尤其是公式、三线表、图注和分页。仅生成文件不等同于完成排版验证。

报告中的性能来自外层五折测试；内层五折仅用于选参。SVM 分数没有解释为概率。t-SNE 仅用于描述特征分布，不作为预测输入或性能证据。
