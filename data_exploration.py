"""
数据探索与预处理
数据集：Breast Cancer Wisconsin (Diagnostic)
任务：基于细胞核特征预测乳腺肿瘤良恶性
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")
target_names = data.target_names  # ['malignant', 'benign']
print("数据集大小:", X.shape)
print("特征数:", X.shape[1])
print("类别分布:")
print(y.value_counts().rename(index=lambda i: f"{i}={target_names[i]}"))
print("\n特征名:")
for i, c in enumerate(X.columns):
    print(f"  {i+1:2d}. {c}")
print("\n基本统计:")
print(X.describe().T[["mean", "std", "min", "max"]].round(3).to_string())

X.to_csv(os.path.join(OUT, "features.csv"), index=False, encoding="utf-8-sig")
y.to_csv(os.path.join(OUT, "target.csv"), index=False, encoding="utf-8-sig", header=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
counts = y.value_counts().sort_index()
counts.index = [target_names[i] for i in counts.index]
colors = ["#d62728", "#2ca02c"]
ax.bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
for i, (name, v) in enumerate(counts.items()):
    ax.text(i, v + 3, str(v), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("样本数")
ax.set_title("类别分布（良性 vs 恶性）")
ax.set_ylim(0, counts.max() * 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax = axes[1]
ax.pie(counts.values, labels=counts.index, autopct="%.1f%%",
       colors=colors, startangle=90, textprops={"fontsize": 11})
ax.set_title("类别占比")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(5, 6, figsize=(16, 11))
axes = axes.flatten()
for i, col in enumerate(X.columns):
    axes[i].hist(X[y == 1][col], bins=25, alpha=0.6, label="良性", color="#2ca02c", density=True)
    axes[i].hist(X[y == 0][col], bins=25, alpha=0.6, label="恶性", color="#d62728", density=True)
    axes[i].set_title(col, fontsize=8)
    axes[i].tick_params(axis="both", labelsize=7)
axes[0].legend(fontsize=8, loc="upper right")
for j in range(len(X.columns), len(axes)):
    axes[j].axis("off")
fig.suptitle("各特征在良/恶性样本上的分布", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "feature_histograms.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(8, 6))
for i, c in enumerate(["#d62728", "#2ca02c"]):
    mask = y == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=c, label=target_names[i],
               alpha=0.7, edgecolors="k", linewidths=0.4, s=55)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("前两个主成分的散点图（PCA）")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pca_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
imp = np.abs(pca.components_[0])
order = np.argsort(imp)[::-1][:15]
ax.barh(range(len(order)), imp[order], color="#1f77b4", edgecolor="black")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([X.columns[j] for j in order], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("|PC1 载荷|")
ax.set_title("前 15 个最重要特征（按 PC1 载荷绝对值）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "feature_importance_pca.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

corr = X.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.7}, ax=ax,
            xticklabels=[c[:18] for c in X.columns],
            yticklabels=[c[:18] for c in X.columns])
ax.tick_params(axis="both", labelsize=7)
ax.set_title("特征相关系数热力图")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.25, random_state=42, stratify=y.values
)
np.savez(os.path.join(OUT, "data_split.npz"),
         X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
         feature_names=X.columns.values, target_names=np.array(target_names))
print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
print("已保存 data_split.npz / features.csv / target.csv")
print("图表文件: class_distribution / feature_histograms / pca_scatter / feature_importance_pca / correlation_heatmap")
