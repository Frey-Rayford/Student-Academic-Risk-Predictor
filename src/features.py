import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "credits_enrolled",
    "past_failures",
    "previous_gpa",
    "study_hours_weekly",
    "commute_minutes",
    "part_time_job",
    "absences",
    "late_submissions",
    "lms_access_frequency",
    "quiz_1",
    "quiz_2",
    "score_trend",
    "quiz_average",
    "study_per_credit",
    "academic_burden",
    "engagement_penalty",
    "lms_intensity"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tao cac dac trung phai sinh mang tinh nghiep vu hoc vu (domain knowledge).
    """
    data = df.copy()
    
    # 1. Xu huong diem qua cac dot kiem tra
    data["score_trend"] = np.round(data["quiz_2"] - data["quiz_1"], 2)
    data["quiz_average"] = np.round((data["quiz_1"] + data["quiz_2"]) / 2, 2)
    
    # 2. Ty le gio tu hoc tren moi tin chi dang ky
    data["study_per_credit"] = np.round(data["study_hours_weekly"] / np.maximum(data["credits_enrolled"], 1), 3)
    
    # 3. Ap luc hoc vu: so tin chi ket hop so mon no qua khu
    data["academic_burden"] = data["credits_enrolled"] * (data["past_failures"] + 1)
    
    # 4. Diem phat thieu tuong tac (vang hoc + nop bai muon)
    data["engagement_penalty"] = data["absences"] * 2.0 + data["late_submissions"] * 1.5
    
    # 5. Tan suat hoat dong LMS tuong doi voi gio tu hoc
    data["lms_intensity"] = np.round(data["lms_access_frequency"] / (data["study_hours_weekly"] + 1.0), 2)
    
    return data

def get_feature_matrix(df: pd.DataFrame):
    """
    Tach ma tran dac trung X va bien muc tieu y.
    """
    data = engineer_features(df)
    X = data[FEATURE_COLUMNS]
    y = data["at_risk"] if "at_risk" in data.columns else None
    return X, y
