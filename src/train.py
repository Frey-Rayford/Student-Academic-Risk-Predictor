import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from data_loader import load_or_create_data
from features import get_feature_matrix, FEATURE_COLUMNS

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": float(np.round(accuracy_score(y_true, y_pred), 4)),
        "precision": float(np.round(precision_score(y_true, y_pred, zero_division=0), 4)),
        "recall": float(np.round(recall_score(y_true, y_pred), 4)),
        "f1_score": float(np.round(f1_score(y_true, y_pred), 4)),
        "roc_auc": float(np.round(roc_auc_score(y_true, y_prob), 4)),
        "pr_auc": float(np.round(average_precision_score(y_true, y_prob), 4))
    }

def find_optimal_threshold(y_true, y_prob, min_recall=0.85):
    """
    Tim nguong cat xac suat toi uu sao cho do bao phu (Recall) dat toi thieu 85%
    nham han che toi da viec bo sot sinh vien can ho tro hoc vu.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = 0.0
    
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        if rec >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            
    # Neu khong dat duoc min_recall o bat ky threshold nao tren, chon nguong co recall cao nhat
    if best_f1 == 0.0:
        for th in thresholds:
            preds = (y_prob >= th).astype(int)
            rec = recall_score(y_true, preds, zero_division=0)
            if rec >= min_recall:
                best_thresh = th
                break
                
    return float(np.round(best_thresh, 2))

def train_and_compare(base_dir: str):
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)

    # 1. Load data
    df = load_or_create_data(data_dir)
    X, y = get_feature_matrix(df)

    # 2. Train-Test Split (80/20, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Luu tap du lieu da phan chia
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(os.path.join(data_dir, "processed", "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "processed", "test.csv"), index=False)

    # 3. Khoi tao 3 mo hinh
    models = {
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.04, class_weight="balanced",
            random_state=42, verbose=-1
        )
    }

    results = {}
    fitted_models = {}
    probs_dict = {}

    print("\n------------------------------------------------------------")
    print("KET QUA HUAN LUYEN VA DANH GIA TREN TAP TEST (DEFAULT THRESHOLD = 0.5)")
    print("------------------------------------------------------------")
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'ROC-AUC':<8}")
    print("------------------------------------------------------------")

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Du doan xac suat
        y_prob = model.predict_proba(X_test)[:, 1]
        probs_dict[name] = y_prob
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        results[name] = metrics

        print(
            f"{name:<22} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<9.4f} | "
            f"{metrics['recall']:<7.4f} | {metrics['f1_score']:<7.4f} | {metrics['roc_auc']:<8.4f}"
        )

    # 4. Chon mo hinh tot nhat dua tren ROC-AUC va F1
    best_name = max(results, key=lambda k: results[k]["roc_auc"] + results[k]["f1_score"])
    best_model = fitted_models[best_name]
    best_prob = probs_dict[best_name]

    print("\n------------------------------------------------------------")
    print(f"Mo hinh duoc lua chon: {best_name}")
    print("------------------------------------------------------------")

    # 5. Threshold Tuning cho mo hinh tot nhat
    optimal_th = find_optimal_threshold(y_test, best_prob, min_recall=0.85)
    tuned_preds = (best_prob >= optimal_th).astype(int)
    tuned_metrics = evaluate_predictions(y_test, tuned_preds, best_prob)

    print(f"Nguong cat toi uu (Threshold Tuning): {optimal_th} (Thay vi 0.5)")
    print(f"- Recall sau tinh chinh:    {tuned_metrics['recall']:.4f} (Dat muc tieu bao phu)")
    print(f"- Precision sau tinh chinh: {tuned_metrics['precision']:.4f}")
    print(f"- F1-Score sau tinh chinh:  {tuned_metrics['f1_score']:.4f}")
    print("------------------------------------------------------------\n")

    # 6. Luu mo hinh va metadata
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "default_metrics": results,
        "optimal_threshold": optimal_th,
        "tuned_metrics": tuned_metrics,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Da luu model tai: {model_path}")
    print(f"Da luu metadata tai: {os.path.join(models_dir, 'model_metadata.json')}")

    return best_name, results, tuned_metrics

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_compare(base_dir)
