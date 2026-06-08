"""
模型性能对比与综合图表：SVM vs BP 神经网络
- 指标对比柱状图
- ROC 曲线对比
- 雷达图对比
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "outputs")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

with open(os.path.join(OUT, "svm_results.json"), "r", encoding="utf-8") as f:
    svm = json.load(f)
with open(os.path.join(OUT, "mlp_results.json"), "r", encoding="utf-8") as f:
    mlp = json.load(f)

metrics = ["test_acc", "precision", "recall", "f1", "auc"]
labels = ["准确率", "精确率", "召回率", "F1", "AUC"]
svm_vals = [svm[m] for m in metrics]
mlp_vals = [mlp[m] for m in metrics]

print("=== 模型性能对比 ===")
print(f"{'指标':<8} {'SVM':>8} {'BP-NN':>8}")
for lb, s, m in zip(labels, svm_vals, mlp_vals):
    print(f"{lb:<8} {s:>8.4f} {m:>8.4f}")
print(f"训练耗时(s): SVM={svm['train_time_s']:.3f}, BP-NN={mlp['train_time_s']:.3f}")
print(f"预测耗时(ms): SVM={svm['pred_time_ms']:.3f}, BP-NN={mlp['pred_time_ms']:.3f}")

x = np.arange(len(labels))
width = 0.36
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width/2, svm_vals, width, label="SVM", color="#1f77b4", edgecolor="black")
b2 = ax.bar(x + width/2, mlp_vals, width, label="BP 神经网络", color="#ff7f0e", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("得分"); ax.set_title("SVM 与 BP 神经网络性能指标对比")
ax.set_ylim(0.80, 1.02)
ax.legend(loc="lower right"); ax.grid(axis="y", linestyle="--", alpha=0.4)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
svm_closed = svm_vals + [svm_vals[0]]
mlp_closed = mlp_vals + [mlp_vals[0]]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
ax.plot(angles, svm_closed, "o-", color="#1f77b4", lw=2, label="SVM")
ax.fill(angles, svm_closed, color="#1f77b4", alpha=0.22)
ax.plot(angles, mlp_closed, "s-", color="#ff7f0e", lw=2, label="BP 神经网络")
ax.fill(angles, mlp_closed, color="#ff7f0e", alpha=0.22)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.85, 1.0)
ax.set_title("模型性能雷达图", fontsize=13)
ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0.0))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "radar_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

times_labels = ["训练耗时(s)", "预测耗时(ms)"]
times_svm = [svm["train_time_s"], svm["pred_time_ms"]]
times_mlp = [mlp["train_time_s"], mlp["pred_time_ms"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for i, (name, unit, vals) in enumerate([
    ("训练耗时", "秒 (s)", [svm["train_time_s"], mlp["train_time_s"]]),
    ("预测耗时", "毫秒 (ms)", [svm["pred_time_ms"], mlp["pred_time_ms"]]),
]):
    ax = axes[i]
    bars = ax.bar(["SVM", "BP 神经网络"], vals,
                  color=["#1f77b4", "#ff7f0e"], edgecolor="black", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2., v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(f"{name}对比"); ax.set_ylabel(unit)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "time_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n图表已生成: metrics_comparison.png / radar_comparison.png / time_comparison.png")
