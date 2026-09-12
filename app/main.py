import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Thiet lap duong dan import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.features import engineer_features, FEATURE_COLUMNS
from src.explain import explain_single_student

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = FastAPI(title="Student Academic Risk Predictor API", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
if os.path.exists(FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

# Load model & metadata
MODELS_DIR = os.path.join(BASE_DIR, "models")
model_path = os.path.join(MODELS_DIR, "best_model.joblib")
meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

model = joblib.load(model_path) if os.path.exists(model_path) else None
with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

OPTIMAL_THRESHOLD = metadata.get("optimal_threshold", 0.75)

class StudentProfile(BaseModel):
    credits_enrolled: int = Field(16, ge=10, le=30)
    past_failures: int = Field(0, ge=0, le=10)
    previous_gpa: float = Field(2.8, ge=0.0, le=4.0)
    study_hours_weekly: float = Field(12.0, ge=0.0, le=50.0)
    commute_minutes: int = Field(30, ge=5, le=180)
    part_time_job: int = Field(0, ge=0, le=1)
    absences: int = Field(1, ge=0, le=15)
    late_submissions: int = Field(0, ge=0, le=10)
    lms_access_frequency: int = Field(15, ge=0, le=60)
    quiz_1: float = Field(7.0, ge=0.0, le=10.0)
    quiz_2: float = Field(6.5, ge=0.0, le=10.0)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>Dang khoi tao UI...</h3>")

@app.post("/api/predict")
async def predict_risk(profile: StudentProfile):
    if model is None:
        raise HTTPException(status_code=500, detail="Chua tim thay model da huan luyen")
        
    data_dict = profile.model_dump()
    df_raw = pd.DataFrame([data_dict])
    df_feat = engineer_features(df_raw)[FEATURE_COLUMNS]
    
    # Du doan xac suat
    prob_risk = float(model.predict_proba(df_feat)[0, 1])
    is_at_risk = bool(prob_risk >= OPTIMAL_THRESHOLD)
    
    if prob_risk >= 0.70:
        risk_level = "HIGH_RISK"
        risk_label = "Nguy Cơ Cao"
        status_color = "#ef4444"
    elif prob_risk >= 0.35:
        risk_level = "MODERATE_RISK"
        risk_label = "Cảnh Báo Nhẹ"
        status_color = "#f59e0b"
    else:
        risk_level = "SAFE"
        risk_label = "An Toàn"
        status_color = "#10b981"
        
    # Giai thich SHAP
    explanation = explain_single_student(data_dict, BASE_DIR)
    
    # De xuat khuyen nghi hoc vu
    recommendations = []
    if data_dict["absences"] >= 3:
        recommendations.append(f"Điểm danh: Đã vắng {data_dict['absences']} buổi, cần hạn chế tối đa vắng học để tránh bị cấm thi cuối kỳ.")
    if data_dict["quiz_2"] < data_dict["quiz_1"]:
        diff = round(data_dict["quiz_1"] - data_dict["quiz_2"], 1)
        recommendations.append(f"Xu hướng điểm: Bài Quiz 2 bị tụt {diff} điểm so với Quiz 1. Cần rà soát và ôn tập lại phần kiến thức tuần 4 - 6.")
    if data_dict["study_hours_weekly"] < 8.0:
        recommendations.append(f"Thời gian tự học: Hiện tại chỉ đạt {data_dict['study_hours_weekly']} giờ/tuần, quá ít so với khối lượng {data_dict['credits_enrolled']} tín chỉ đang theo học.")
    if data_dict["past_failures"] > 0:
        recommendations.append(f"Lịch sử học vụ: Từng nợ {data_dict['past_failures']} môn trước đó, cần chủ động liên hệ giảng viên hoặc nhóm học tập để được trợ giúp sớm.")
    if not recommendations:
        recommendations.append("Duy trì phong độ học tập tích cực hiện tại và tập trung chuẩn bị tốt cho kỳ thi cuối kỳ.")

    return {
        "risk_probability": round(prob_risk * 100, 1),
        "is_at_risk": is_at_risk,
        "risk_level": risk_level,
        "risk_label": risk_label,
        "status_color": status_color,
        "optimal_threshold_used": OPTIMAL_THRESHOLD,
        "features_derived": {
            "score_trend": round(data_dict["quiz_2"] - data_dict["quiz_1"], 1),
            "quiz_average": round((data_dict["quiz_1"] + data_dict["quiz_2"]) / 2, 1),
            "study_per_credit": round(data_dict["study_hours_weekly"] / max(data_dict["credits_enrolled"], 1), 2)
        },
        "shap_explanation": explanation,
        "recommendations": recommendations
    }

@app.get("/api/sample-profiles")
async def get_sample_profiles():
    return {
        "safe": {
            "label": "Sinh viên gương mẫu (An toàn)",
            "data": {
                "credits_enrolled": 16,
                "past_failures": 0,
                "previous_gpa": 3.4,
                "study_hours_weekly": 16.0,
                "commute_minutes": 20,
                "part_time_job": 0,
                "absences": 0,
                "late_submissions": 0,
                "lms_access_frequency": 24,
                "quiz_1": 8.0,
                "quiz_2": 8.5
            }
        },
        "declining": {
            "label": "Dấu hiệu sa sút giữa kỳ (Cảnh báo)",
            "data": {
                "credits_enrolled": 20,
                "past_failures": 1,
                "previous_gpa": 2.6,
                "study_hours_weekly": 8.0,
                "commute_minutes": 45,
                "part_time_job": 1,
                "absences": 3,
                "late_submissions": 2,
                "lms_access_frequency": 10,
                "quiz_1": 6.5,
                "quiz_2": 4.5
            }
        },
        "critical": {
            "label": "Nguy cơ rớt môn cao (Báo động)",
            "data": {
                "credits_enrolled": 22,
                "past_failures": 2,
                "previous_gpa": 2.1,
                "study_hours_weekly": 5.0,
                "commute_minutes": 60,
                "part_time_job": 1,
                "absences": 5,
                "late_submissions": 4,
                "lms_access_frequency": 6,
                "quiz_1": 4.5,
                "quiz_2": 3.0
            }
        }
    }
