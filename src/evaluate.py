import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src.features import FEATURE_COLUMNS

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_evaluation_reports(base_dir: str):
    data_dir = os.path.join(base_dir, "data", "processed")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(figures_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["at_risk"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["at_risk"]

    with open(os.path.join(models_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Style do thi sach se, hien dai
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    models = {name.replace("_", " "): joblib.load(os.path.join(models_dir, f"{name}.joblib"))
              for name in metadata["default_metrics"]}
    probs = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}

    # 1. Bieu do so sanh cac metrics cua 3 model
    metrics_data = metadata["default_metrics"]
    df_metrics = pd.DataFrame(metrics_data).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    df_metrics.index = ["Logistic Regression", "Random Forest", "LightGBM"]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    df_metrics.plot(kind="bar", ax=ax, width=0.75, colormap="viridis")
    ax.set_title("Model comparison — held-out synthetic test set", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Diem So (0.0 - 1.0)", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_xticklabels(df_metrics.index, rotation=0, fontsize=10)
    ax.legend(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], frameon=True, facecolor="white", edgecolor="#ddd")
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "model_comparison.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"Da luu bieu do so sanh: {chart1_path}")

    # 2. Bieu do ROC Curves va PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=180)
    colors = ["#2563eb", "#10b981", "#f59e0b"]

    for (name, prob), color in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        
        auc_roc = metadata["default_metrics"][name.replace(" ", "_")]["roc_auc"]
        auc_pr = metadata["default_metrics"][name.replace(" ", "_")]["pr_auc"]
        
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_roc:.3f})", color=color, linewidth=1.8)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {auc_pr:.3f})", color=color, linewidth=1.8)

    ax1.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax1.set_title("ROC Curves", fontsize=11, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=9)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=9)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.set_title("Precision-Recall Curves", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=9)
    ax2.set_ylabel("Precision", fontsize=9)
    ax2.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "roc_pr_curves.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"Da luu bieu do ROC-PR: {chart2_path}")

    # 3. Confusion Matrix: Truoc va sau khi tinh chinh nguong (Threshold Tuning)
    best_prob = probs[metadata["best_model_name"].replace("_", " ")]
    opt_th = metadata["optimal_threshold"]

    cm_default = confusion_matrix(y_test, (best_prob >= 0.5).astype(int))
    cm_tuned = confusion_matrix(y_test, (best_prob >= opt_th).astype(int))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=180)
    ConfusionMatrixDisplay(cm_default, display_labels=["An Toan", "Nguy Co"]).plot(ax=ax1, cmap="Blues", colorbar=False)
    ax1.set_title("Default Threshold (0.50)", fontsize=10.5, fontweight="bold")
    ax1.grid(False)

    ConfusionMatrixDisplay(cm_tuned, display_labels=["An Toan", "Nguy Co"]).plot(ax=ax2, cmap="Greens", colorbar=False)
    ax2.set_title(f"Optimized Threshold ({opt_th:.2f})", fontsize=10.5, fontweight="bold")
    ax2.grid(False)

    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(chart3_path)
    plt.close()
    print(f"Da luu bieu do Confusion Matrix: {chart3_path}")

    # 4. Feature Importance tu Random Forest
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importances
    }).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=180)
    ax.barh(feat_df["feature"], feat_df["importance"], color="#3b82f6", edgecolor="#1d4ed8", height=0.65)
    ax.set_title("Xep Hang Tam Quan Trong Cua Cac Dac Trung (Random Forest Gini)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=9.5)
    plt.tight_layout()
    chart4_path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(chart4_path)
    plt.close()
    print(f"Da luu bieu do Feature Importance: {chart4_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_evaluation_reports(base_dir)
