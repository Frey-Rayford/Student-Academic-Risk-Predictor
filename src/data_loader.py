import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_student_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Sinh bo du lieu thuc nghiem mo phong hanh vi hoc tap cua sinh vien
    tinh den giai doan giua ky (tuan 6-8).
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Thong tin hoc vu va boi canh
    credits_enrolled = rng.integers(12, 25, size=n_samples)
    past_failures = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.72, 0.18, 0.07, 0.03])
    
    # GPA cac ky truoc (phan phoi quanh 2.7, lech theo past_failures)
    base_gpa = rng.normal(loc=2.85, scale=0.55, size=n_samples) - past_failures * 0.4
    previous_gpa = np.clip(np.round(base_gpa, 2), 1.0, 4.0)
    
    study_hours_weekly = np.clip(np.round(rng.normal(loc=12, scale=5, size=n_samples), 1), 1.0, 30.0)
    commute_minutes = rng.choice([15, 30, 45, 60, 90], size=n_samples, p=[0.35, 0.30, 0.18, 0.12, 0.05])
    part_time_job = rng.choice([0, 1], size=n_samples, p=[0.58, 0.42])
    
    # 2. Hanh vi chuyen can va tuong tac LMS (tuan 1-7)
    # Di lam them + di xa thuong co xu huong vang nhieu hon mot chut
    absence_prob = 1.8 + part_time_job * 0.8 + (commute_minutes >= 60) * 0.6
    absences = np.clip(rng.poisson(lam=absence_prob, size=n_samples), 0, 10)
    
    late_submissions = np.clip(rng.poisson(lam=0.9 + part_time_job * 0.5, size=n_samples), 0, 6)
    lms_access_frequency = np.clip(rng.poisson(lam=study_hours_weekly * 1.1 + 4, size=n_samples), 2, 40)
    
    # 3. Diem kiem tra tien trinh (Quiz 1 va Quiz 2 tren thang 10)
    # Quiz 1: Nen tang kien thuc ban dau
    q1_mean = 6.8 + (previous_gpa - 2.5) * 1.2 - past_failures * 0.5
    quiz_1 = np.clip(np.round(rng.normal(loc=q1_mean, scale=1.3), 1), 1.0, 10.0)
    
    # Quiz 2: Chuyen bien theo do chuyen can va gio hoc
    effort_effect = (study_hours_weekly - 10) * 0.15 - absences * 0.35 - late_submissions * 0.2
    q2_mean = quiz_1 + effort_effect + rng.normal(0, 0.8, size=n_samples)
    quiz_2 = np.clip(np.round(q2_mean, 1), 1.0, 10.0)
    
    # 4. Tinh toan nhan nguy co rot mon (at_risk)
    # Log-odds cua viec rot mon dua tren su ket hop phi tuyen
    risk_score = (
        - 0.9 * (quiz_2 - 5.0)
        - 0.6 * (quiz_1 - 5.0)
        + 0.55 * absences
        + 0.65 * past_failures
        + 0.35 * late_submissions
        - 0.8 * (previous_gpa - 2.5)
        - 0.12 * (study_hours_weekly - 10)
        + 0.08 * (credits_enrolled - 16)
        + 0.35 * part_time_job
        + rng.normal(0, 0.75, size=n_samples)
    )
    
    # Chuyen thanh xac suat bang sigmoid
    prob_risk = 1 / (1 + np.exp(-risk_score))
    at_risk = (prob_risk >= 0.52).astype(int)
    
    df = pd.DataFrame({
        "student_id": [f"STU_{i+1:04d}" for i in range(n_samples)],
        "credits_enrolled": credits_enrolled,
        "past_failures": past_failures,
        "previous_gpa": previous_gpa,
        "study_hours_weekly": study_hours_weekly,
        "commute_minutes": commute_minutes,
        "part_time_job": part_time_job,
        "absences": absences,
        "late_submissions": late_submissions,
        "lms_access_frequency": lms_access_frequency,
        "quiz_1": quiz_1,
        "quiz_2": quiz_2,
        "at_risk": at_risk
    })
    
    return df

def load_or_create_data(data_dir: str = "data") -> pd.DataFrame:
    raw_path = os.path.join(data_dir, "raw", "student_data.csv")
    if os.path.exists(raw_path):
        print(f"Doc du lieu tu: {raw_path}")
        return pd.read_csv(raw_path)
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df = generate_student_dataset(n_samples=1200, random_state=42)
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Da tao va luu bo du lieu mau ({len(df)} sinh vien) tai: {raw_path}")
    print(f"Ty le sinh vien co nguy co rot mon: {df['at_risk'].mean():.2%}")
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_or_create_data(os.path.join(base_dir, "data"))
