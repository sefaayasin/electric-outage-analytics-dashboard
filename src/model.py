import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =============================================================================
# DOSYA YOLLARI
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

OUTAGES_PATH = os.path.join(DATA_DIR, "outages.csv")

DURATION_MODEL_PATH = os.path.join(MODELS_DIR, "duration_model.pkl")
HIGH_IMPACT_MODEL_PATH = os.path.join(MODELS_DIR, "high_impact_model.pkl")

MODEL_METRICS_PATH = os.path.join(PROCESSED_DIR, "model_metrics.csv")
DURATION_FEATURE_IMPORTANCE_PATH = os.path.join(PROCESSED_DIR, "feature_importance_duration.csv")
HIGH_IMPACT_FEATURE_IMPORTANCE_PATH = os.path.join(PROCESSED_DIR, "feature_importance_high_impact.csv")
HIGH_IMPACT_REPORT_PATH = os.path.join(PROCESSED_DIR, "high_impact_classification_report.csv")
HIGH_IMPACT_CONFUSION_MATRIX_PATH = os.path.join(PROCESSED_DIR, "high_impact_confusion_matrix.csv")


# =============================================================================
# VERİ OKUMA
# =============================================================================

def load_data():
    if not os.path.exists(OUTAGES_PATH):
        raise FileNotFoundError(
            f"{OUTAGES_PATH} bulunamadı. Önce python src/generate_data.py çalıştırmalısın."
        )

    df = pd.read_csv(OUTAGES_PATH)

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])

    df["year"] = df["started_at"].dt.year
    df["month"] = df["started_at"].dt.month
    df["day"] = df["started_at"].dt.day
    df["hour"] = df["started_at"].dt.hour
    df["dayofweek"] = df["started_at"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    return df


# =============================================================================
# ÖZELLİK HAZIRLAMA
# =============================================================================

def get_feature_columns():
    categorical_features = [
        "city",
        "district",
        "network_element_type",
        "outage_type",
        "source",
        "cause"
    ]

    numerical_features = [
        "affected_customer_count",
        "is_planned",
        "is_force_majeure",
        "storm_flag",
        "wind_speed",
        "precipitation_mm",
        "year",
        "month",
        "day",
        "hour",
        "dayofweek",
        "is_weekend"
    ]

    return categorical_features, numerical_features


def create_preprocessor(categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    return preprocessor


def prepare_features(df):
    categorical_features, numerical_features = get_feature_columns()

    feature_columns = categorical_features + numerical_features

    X = df[feature_columns].copy()

    return X, categorical_features, numerical_features


# =============================================================================
# FEATURE IMPORTANCE ÇIKARMA
# =============================================================================

def get_feature_names_from_pipeline(pipeline, categorical_features, numerical_features):
    preprocessor = pipeline.named_steps["preprocessor"]

    ohe = preprocessor.named_transformers_["categorical"]
    categorical_names = ohe.get_feature_names_out(categorical_features).tolist()

    feature_names = categorical_names + numerical_features

    return feature_names


def save_feature_importance(pipeline, categorical_features, numerical_features, output_path):
    model = pipeline.named_steps["model"]

    feature_names = get_feature_names_from_pipeline(
        pipeline=pipeline,
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        "importance",
        ascending=False
    )

    feature_importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return feature_importance_df


# =============================================================================
# 1. MODEL: KESİNTİ SÜRESİ TAHMİNİ
# =============================================================================

def train_duration_model(df):
    print("\nKesinti süresi tahmin modeli eğitiliyor...")

    X, categorical_features, numerical_features = prepare_features(df)
    y = df["duration_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = create_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    joblib.dump(pipeline, DURATION_MODEL_PATH)

    feature_importance_df = save_feature_importance(
        pipeline=pipeline,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        output_path=DURATION_FEATURE_IMPORTANCE_PATH
    )

    print("Kesinti süresi modeli kaydedildi:")
    print(DURATION_MODEL_PATH)

    print("\nKesinti Süresi Model Metrikleri")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    print("\nKesinti Süresi Modeli En Önemli 10 Özellik:")
    print(feature_importance_df.head(10))

    metrics = {
        "model_name": "duration_prediction",
        "target": "duration_min",
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None
    }

    return pipeline, metrics


# =============================================================================
# 2. MODEL: YÜKSEK ETKİLİ KESİNTİ TAHMİNİ
# =============================================================================

def train_high_impact_model(df):
    print("\nYüksek etkili kesinti sınıflandırma modeli eğitiliyor...")

    X, categorical_features, numerical_features = prepare_features(df)
    y = df["high_impact"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = create_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(HIGH_IMPACT_REPORT_PATH, encoding="utf-8-sig")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        columns=["predicted_0", "predicted_1"],
        index=["actual_0", "actual_1"]
    )
    cm_df.to_csv(HIGH_IMPACT_CONFUSION_MATRIX_PATH, encoding="utf-8-sig")

    joblib.dump(pipeline, HIGH_IMPACT_MODEL_PATH)

    feature_importance_df = save_feature_importance(
        pipeline=pipeline,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        output_path=HIGH_IMPACT_FEATURE_IMPORTANCE_PATH
    )

    print("Yüksek etkili kesinti modeli kaydedildi:")
    print(HIGH_IMPACT_MODEL_PATH)

    print("\nYüksek Etkili Kesinti Model Metrikleri")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nYüksek Etkili Kesinti Modeli En Önemli 10 Özellik:")
    print(feature_importance_df.head(10))

    metrics = {
        "model_name": "high_impact_classification",
        "target": "high_impact",
        "mae": None,
        "rmse": None,
        "r2": None,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

    return pipeline, metrics


# =============================================================================
# ÖRNEK TAHMİN
# =============================================================================

def make_sample_predictions(duration_model, high_impact_model, df):
    sample_df = df.sample(5, random_state=42).copy()

    X_sample, _, _ = prepare_features(sample_df)

    duration_predictions = duration_model.predict(X_sample)
    high_impact_predictions = high_impact_model.predict(X_sample)

    if hasattr(high_impact_model.named_steps["model"], "predict_proba"):
        high_impact_probabilities = high_impact_model.predict_proba(X_sample)[:, 1]
    else:
        high_impact_probabilities = [None] * len(sample_df)

    result_df = sample_df[
        [
            "outage_id",
            "city",
            "district",
            "network_element_type",
            "outage_type",
            "source",
            "cause",
            "affected_customer_count",
            "energy_not_supplied_kwh",
            "duration_min",
            "high_impact"
        ]
    ].copy()

    result_df["predicted_duration_min"] = np.round(duration_predictions, 2)
    result_df["predicted_high_impact"] = high_impact_predictions
    result_df["high_impact_probability"] = np.round(high_impact_probabilities, 4)

    sample_prediction_path = os.path.join(PROCESSED_DIR, "sample_predictions.csv")
    result_df.to_csv(sample_prediction_path, index=False, encoding="utf-8-sig")

    print("\nÖrnek Tahminler:")
    print(result_df)

    print(f"\nÖrnek tahmin dosyası kaydedildi: {sample_prediction_path}")

    return result_df


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    print("Makine öğrenmesi süreci başladı...")

    df = load_data()

    duration_model, duration_metrics = train_duration_model(df)
    high_impact_model, high_impact_metrics = train_high_impact_model(df)

    metrics_df = pd.DataFrame([
        duration_metrics,
        high_impact_metrics
    ])

    metrics_df.to_csv(MODEL_METRICS_PATH, index=False, encoding="utf-8-sig")

    make_sample_predictions(
        duration_model=duration_model,
        high_impact_model=high_impact_model,
        df=df
    )

    print("\nModel metrikleri kaydedildi:")
    print(MODEL_METRICS_PATH)

    print("\nMakine öğrenmesi süreci tamamlandı.")


if __name__ == "__main__":
    main()