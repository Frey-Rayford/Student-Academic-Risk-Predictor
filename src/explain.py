import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline

from src.features import FEATURE_COLUMNS, engineer_features


def explain_features(model, background, features):
    if isinstance(model, Pipeline):
        transformer = model[:-1]
        classifier = model[-1]
        values = transformer.transform(features)
        reference = transformer.transform(background)
        result = shap.LinearExplainer(classifier, reference)(values)
        result.data = features.to_numpy()
        result.feature_names = list(features.columns)
        return result, "log_odds"
    result = shap.TreeExplainer(model)(features)
    if result.values.ndim == 3:
        result = result[:, :, 1]
    space = "probability" if hasattr(model, "estimators_") and not hasattr(model, "booster_") else "log_odds"
    return result, space


def load_explanation(base_dir, features):
    root = Path(base_dir)
    model = joblib.load(root / "models/best_model.joblib")
    background = pd.read_csv(root / "data/processed/train.csv")[FEATURE_COLUMNS]
    return explain_features(model, background, features)


def generate_shap_reports(base_dir):
    root = Path(base_dir)
    features = pd.read_csv(root / "data/processed/test.csv")[FEATURE_COLUMNS]
    explanation, space = load_explanation(root, features)
    metadata = json.loads((root / "models/model_metadata.json").read_text(encoding="utf-8"))
    figures = root / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(explanation, features, show=False, max_display=12)
    plt.title(f"{metadata['best_model_name']} SHAP ({space}; synthetic data)")
    plt.tight_layout()
    plt.savefig(figures / "shap_summary.png", dpi=160)
    plt.close()
    shap.plots.waterfall(explanation[0], max_display=9, show=False)
    plt.title(f"Sample 1 — {metadata['best_model_name']} ({space})")
    plt.tight_layout()
    plt.savefig(figures / "shap_student_sample.png", dpi=160)
    plt.close()


def explain_single_student(student_dict, base_dir):
    df_feat = engineer_features(pd.DataFrame([student_dict]))[FEATURE_COLUMNS]
    explanation, space = load_explanation(base_dir, df_feat)
    values = explanation.values[0]
    metadata = json.loads((Path(base_dir) / "models/model_metadata.json").read_text(encoding="utf-8"))
    feature_vn_names = {
        "quiz_2": "Điểm kiểm tra Quiz 2",
        "quiz_1": "Điểm kiểm tra Quiz 1",
        "quiz_average": "Điểm Quiz trung bình",
        "score_trend": "Xu hướng điểm kiểm tra",
        "engagement_penalty": "Điểm phạt vắng & trễ hạn",
        "absences": "Số buổi vắng học",
        "late_submissions": "Số bài tập nộp trễ hạn",
        "academic_burden": "Áp lực học vụ (tín chỉ + nợ môn)",
        "study_hours_weekly": "Thời gian tự học hàng tuần",
        "previous_gpa": "GPA các học kỳ trước",
        "past_failures": "Số môn từng nợ quá khứ",
        "credits_enrolled": "Số tín chỉ đang học",
        "study_per_credit": "Giờ tự học trên mỗi tín chỉ",
        "lms_access_frequency": "Tương tác hệ thống LMS",
        "commute_minutes": "Thời gian đi lại tới trường",
        "part_time_job": "Đi làm thêm ngoài giờ",
        "lms_intensity": "Mức độ tương tác LMS"
    }

    impacts = [{
        "feature": name,
        "feature_label": feature_vn_names.get(name, name),
        "value": float(df_feat.iloc[0][name]),
        "shap_value": round(float(value), 3),
    } for name, value in zip(FEATURE_COLUMNS, values)]
    impacts.sort(key=lambda row: row["shap_value"], reverse=True)
    return {
        "model_name": metadata["best_model_name"],
        "output_space": space,
        "base_value": float(np.asarray(explanation.base_values).reshape(-1)[0]),
        "top_risk_drivers": [row for row in impacts if row["shap_value"] > 0][:3],
        "top_protective": [row for row in reversed(impacts) if row["shap_value"] < 0][:2],
    }


if __name__ == "__main__":
    generate_shap_reports(Path(__file__).resolve().parents[1])
