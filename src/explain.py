import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

try:
    from src.features import FEATURE_COLUMNS, engineer_features
except ImportError:
    from features import FEATURE_COLUMNS, engineer_features

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_shap_reports(base_dir: str):
    data_dir = os.path.join(base_dir, "data", "processed")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(figures_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["at_risk"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["at_risk"]

    # Su dung LightGBM de tao TreeExplainer toi uu
    lgb_model = LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.04, class_weight="balanced",
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer(X_test)

    # 1. Ve SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 6.5), dpi=180)
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.title("SHAP Feature Attribution (Yeu To Quyen Dinh Nguy Co Rot Mon)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    chart_summary_path = os.path.join(figures_dir, "shap_summary.png")
    plt.savefig(chart_summary_path)
    plt.close()
    print(f"Da luu bieu do SHAP Summary: {chart_summary_path}")

    # 2. Ve SHAP Waterfall cho 1 sinh vien co nguy co cao dien hinh
    high_risk_indices = np.where((y_test == 1) & (lgb_model.predict_proba(X_test)[:, 1] > 0.85))[0]
    sample_idx = int(high_risk_indices[0]) if len(high_risk_indices) > 0 else 0

    plt.figure(figsize=(9, 5.5), dpi=180)
    shap.plots.waterfall(shap_values[sample_idx], max_display=9, show=False)
    plt.title(f"SHAP Waterfall: Giai Thich Du Doan Cho Sinh Vien #{sample_idx+1}", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    chart_sample_path = os.path.join(figures_dir, "shap_student_sample.png")
    plt.savefig(chart_sample_path)
    plt.close()
    print(f"Da luu bieu do SHAP Waterfall mau: {chart_sample_path}")

    # Luu explainer vao models
    joblib.dump(lgb_model, os.path.join(models_dir, "explainer_model.joblib"))

def explain_single_student(student_dict: dict, base_dir: str):
    """
    Ham tien ich cho API va Dashboard: Giai thich 3 yeu to rui ro lon nhat cua 1 sinh vien.
    """
    models_dir = os.path.join(base_dir, "models")
    lgb_model = joblib.load(os.path.join(models_dir, "explainer_model.joblib"))
    
    df_raw = pd.DataFrame([student_dict])
    df_feat = engineer_features(df_raw)[FEATURE_COLUMNS]
    
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer(df_feat)
    
    values = shap_vals.values[0]
    feature_names = FEATURE_COLUMNS
    
    # Sap xep theo muc do day nguy co len cao (SHAP duong lon nhat)
    impact_list = []
    for f_name, f_val, s_val in zip(feature_names, df_feat.iloc[0], values):
        impact_list.append({
            "feature": f_name,
            "value": float(np.round(f_val, 2)),
            "shap_value": float(np.round(s_val, 3))
        })
        
    impact_list.sort(key=lambda x: x["shap_value"], reverse=True)
    top_risk_drivers = [item for item in impact_list if item["shap_value"] > 0][:3]
    top_protective = [item for item in reversed(impact_list) if item["shap_value"] < 0][:2]
    
    return {
        "top_risk_drivers": top_risk_drivers,
        "top_protective": top_protective
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_shap_reports(base_dir)
