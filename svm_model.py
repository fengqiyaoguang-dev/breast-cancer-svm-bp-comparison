"""
SVM 模型：乳腺癌良恶性分类
- 网格搜索超参（kernel, C, gamma）
- 混淆矩阵 / 分类报告 / ROC / 学习曲线 / 交叉验证
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import os, json, time

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

npz = np.load(os.path.join(OUT, "data_split.npz"))
X_train, X_test = npz["X_train"], npz["X_test"]
y_train, y_test = npz["y_train"], npz["y_test"]
target_names = npz["target_names"]

print("=== SVM 训练与评估 ===")
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

param_grid = [
    {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10, 100]},
    {"kernel": ["rbf"],    "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1, 1]},
    {"kernel": ["poly"],   "C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale"]},
]

grid = GridSearchCV(SVC(random_state=42, probability=True),
                    param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring="accuracy", n_jobs=-1, refit=True, verbose=0)

t0 = time.time()
grid.fit(X_train, y_train)
t_train = time.time() - t0

print(f"\n最佳参数: {grid.best_params_}")
print(f"最佳 CV 准确率: {grid.best_score_:.4f}")

best_clf = grid.best_estimator_
t0 = time.time()
y_pred = best_clf.predict(X_test)
t_pred = time.time() - t0
y_prob = best_clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\n测试集准确率: {acc:.4f}")
print(f"精确率: {prec:.4f}  召回率: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
print(f"训练耗时: {t_train:.3f}s, 预测耗时: {t_pred*1000:.2f}ms")
print("\n混淆矩阵:")
print(cm)
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

results = {
    "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v))
                    for k, v in grid.best_params_.items()},
    "best_cv_acc": float(grid.best_score_),
    "test_acc": float(acc), "precision": float(prec),
    "recall": float(rec), "f1": float(f1), "auc": float(auc),
    "confusion_matrix": cm.tolist(),
    "train_time_s": float(t_train),
    "pred_time_ms": float(t_pred * 1000),
    "target_names": list(target_names),
}
with open(os.path.join(OUT, "svm_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.set_title("SVM 混淆矩阵")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(target_names)
ax.set_yticks([0, 1]); ax.set_yticklabels(target_names)
ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "svm_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(fpr, tpr, color="#d62728", lw=2.2,
        label=f"ROC (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机基线")
ax.set_xlabel("假阳性率 FPR"); ax.set_ylabel("真阳性率 TPR")
ax.set_title("SVM ROC 曲线")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "svm_roc.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

train_sizes, train_scores, val_scores = learning_curve(
    best_clf, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
)
train_mean, train_std = train_scores.mean(1), train_scores.std(1)
val_mean, val_std = val_scores.mean(1), val_scores.std(1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_mean, "o-", color="#1f77b4", label="训练准确率")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="#1f77b4")
ax.plot(train_sizes, val_mean, "s-", color="#ff7f0e", label="验证准确率")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="#ff7f0e")
ax.set_xlabel("训练集样本数"); ax.set_ylabel("准确率")
ax.set_title("SVM 学习曲线")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "svm_learning_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n图表已生成: svm_confusion_matrix.png / svm_roc.png / svm_learning_curve.png")
print("指标已保存: svm_results.json")
