# UESETC模式识别作业：基于身高、体重的最小错误率贝叶斯分类

使用身高（cm）和体重（kg）两项特征，完成参数估计、概率密度绘图、男女分类和交互预测。

## 查看与运行

`bayes_height_weight.ipynb` 包含完整步骤、公式、图表、测试结果和交互输入。全部结果在 Notebook 内展示。

交互输入需在运行中的Notebook 内使用。

## 实现内容

1. 读取 Excel，排除非法、无效或极端数值的样本，得到共计得到1266条样本。
2. 按类别以约 80% / 20% 划分训练集和测试集（可修改）。
3. 分别绘制男女生身高、体重直方图，以最大似然估计和贝叶斯估计计算并展示一维正态概率密度。
4. 使用最大似然估计得到的二维均值和完整协方差构建分类器，依据最小错误率贝叶斯决策分类。
5. 展示决策区域、测试结果与可交互样本预测。

## 首次安装环境

使用独立 `.venv` 环境，建议使用 Python 3.10。在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

数据文件 `pattern_recognition_homework_dataset.xlsx` 需要与 Notebook 保持在同一目录。
