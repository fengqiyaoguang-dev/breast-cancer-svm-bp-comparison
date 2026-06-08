# 乳腺癌良恶性分类 - SVM 与 BP 神经网络对比实验

基于 Wisconsin Breast Cancer Dataset 的数据挖掘课程实验项目，对比 **SVM（支持向量机）** 与 **BP 神经网络（MLP）** 两种分类模型在乳腺癌良恶性诊断上的性能。

## 📋 项目简介

本项目使用 `scikit-learn` 内置的 **Breast Cancer Wisconsin (Diagnostic) 数据集，基于细胞核的 30 个特征对乳腺肿瘤进行良恶性二分类。通过网格搜索进行超参数调优，从准确率、精确率、召回率、F1 值、AUC、训练时间等多个维度对两种模型进行综合对比。

## 🗂️ 目录结构

```
数据挖掘/
├── data_exploration.py       # 数据探索与预处理（标准化、PCA、数据集划分）
├── svm_model.py              # SVM 模型训练与评估
├── bp_nn_model.py            # BP 神经网络（MLP）模型训练与评估
├── compare_models.py         # 两模型性能综合对比图表生成
├── requirements.txt          # 依赖清单
├── 实验报告.html             # HTML 版实验报告
└── outputs/                  # 运行时生成的图表与中间数据
```

## 🚀 运行步骤

按顺序执行以下脚本（推荐使用 Python 3.9+）：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据探索与预处理
python data_exploration.py

# 3. 训练 SVM
python svm_model.py

# 4. 训练 BP 神经网络
python bp_nn_model.py

# 5. 生成综合对比图表
python compare_models.py
```

## 📊 实验流程

1. **数据探索**：加载数据集，绘制类别分布、特征直方图、相关性热力图
2. **预处理**：标准化 + PCA 降维可视化，按 8:2 划分训练/测试集
3. **模型训练**：
   - **SVM**：网格搜索 `kernel`、`C`、`gamma`
   - **BP-NN (MLP)**：网格搜索隐藏层结构、激活函数、`alpha`、学习率
4. **评估指标**：混淆矩阵、分类报告、ROC 曲线、学习曲线、交叉验证
5. **综合对比**：指标柱状图、雷达图、训练/预测耗时对比

## 🛠️ 依赖

- `numpy`, `pandas` — 数据处理
- `scikit-learn` — 模型与评估
- `matplotlib`, `seaborn` — 可视化
- `python-docx` — 生成 Word 报告

## 📝 说明

本项目为数据挖掘课程的学习实验，代码中已打印关键指标并将图表/JSON 结果保存在 `outputs/` 目录，便于撰写实验报告。
