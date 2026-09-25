import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix


# ============================================================
# Diagnosis Mapping
# ============================================================

def map_diagnosis(code):
    code = str(code)

    if code.startswith("V"):
        return "Supplementary"

    if code.startswith("E"):
        return "External_Cause"

    try:
        code_num = float(code)
    except (TypeError, ValueError):
        return "Unknown"

    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"

    elif 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"

    elif 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"

    elif 250 <= code_num < 251:
        return "Diabetes"

    elif 800 <= code_num <= 999:
        return "Injury_Poisoning"

    elif 710 <= code_num <= 739:
        return "Musculoskeletal"

    elif 580 <= code_num <= 629:
        return "Genitourinary"

    elif 630 <= code_num <= 679:
        return "Pregnancy"

    elif 140 <= code_num <= 239:
        return "Neoplasms"

    elif 280 <= code_num <= 289:
        return "Blood"

    elif 290 <= code_num <= 319:
        return "Mental"

    elif 320 <= code_num <= 359:
        return "Nervous"

    elif 360 <= code_num <= 389:
        return "Sense_Organs"

    elif 680 <= code_num <= 709:
        return "Skin"

    elif 240 <= code_num <= 249:
        return "Endocrine"

    return "Other"


# ============================================================
# Load Final Ensemble Bundle
# ============================================================

def load_bundle(path="final_readmission_ensemble.joblib"):
    return joblib.load(path)


# ============================================================
# Additional Feature Engineering
# Same features used during model training
# ============================================================

def add_engineered_features(data):
    data = data.copy()

    data["total_previous_visits"] = (
        data["number_outpatient"]
        + data["number_emergency"]
        + data["number_inpatient"]
    )

    data["hospital_utilization"] = (
        data["number_emergency"]
        + data["number_inpatient"]
    )

    data["total_clinical_activity"] = (
        data["num_lab_procedures"]
        + data["num_procedures"]
        + data["num_medications"]
    )

    data["medications_per_day"] = (
        data["num_medications"]
        / (data["time_in_hospital"] + 1)
    )

    data["labs_per_day"] = (
        data["num_lab_procedures"]
        / (data["time_in_hospital"] + 1)
    )

    data["procedures_per_day"] = (
        data["num_procedures"]
        / (data["time_in_hospital"] + 1)
    )

    data["diagnosis_burden"] = (
        data["number_diagnoses"]
        / (data["time_in_hospital"] + 1)
    )

    data["has_inpatient_history"] = (
        data["number_inpatient"] > 0
    ).astype(int)

    data["has_emergency_history"] = (
        data["number_emergency"] > 0
    ).astype(int)

    data["has_outpatient_history"] = (
        data["number_outpatient"] > 0
    ).astype(int)

    return data


# ============================================================
# Common Input Cleaning
# ============================================================

def prepare_input(row, bundle):

    data = pd.DataFrame([row])

    # Create has_weight if it is not explicitly supplied
    if "has_weight" not in data.columns:
        weight_value = row.get("weight")

        data["has_weight"] = int(
            pd.notna(weight_value)
            and weight_value != "?"
        )

    required_features = bundle["original_input_columns"]

    missing = set(required_features) - set(data.columns)

    if missing:
        raise ValueError(
            "Missing fields: "
            + ", ".join(sorted(missing))
        )

    # Keep exactly the columns used during training
    data = data[required_features].copy()

    # Replace original missing marker
    data = data.replace("?", "Unknown")

    # Gender cleaning
    if "gender" in data.columns:
        data["gender"] = data["gender"].replace(
            "Unknown/Invalid",
            "Unknown"
        )

    # Laboratory result cleaning
    for col in ["max_glu_serum", "A1Cresult"]:
        if col in data.columns:
            data[col] = (
                data[col]
                .replace("Unknown", "Not_Measured")
                .fillna("Not_Measured")
            )

    # IDs treated as categorical variables
    for col in [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id"
    ]:
        if col in data.columns:
            data[col] = data[col].astype(str)

    # Diagnosis grouping
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in data.columns:
            data[col] = data[col].map(map_diagnosis)

    # Convert numerical columns to numeric values
    for col in bundle["numeric_features_group"]:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="raise"
            )

    return data


# ============================================================
# Original preprocessing for Tuned LightGBM
# ============================================================

def preprocess_for_lightgbm(data, bundle):

    encoder = bundle["encoder_group"]
    scaler = bundle["scaler_group"]

    categorical_features = bundle[
        "categorical_features_group"
    ]

    numeric_features = bundle[
        "numeric_features_group"
    ]

    categorical_matrix = encoder.transform(
        data[categorical_features]
    )

    numeric_matrix = scaler.transform(
        data[numeric_features]
    )

    matrix = hstack([
        categorical_matrix,
        csr_matrix(numeric_matrix)
    ]).tocsr()

    return matrix


# ============================================================
# Engineered preprocessing for XGBoost
# ============================================================

def preprocess_for_xgboost(data, bundle):

    engineered_data = add_engineered_features(data)

    preprocessor = bundle[
        "engineered_preprocessor"
    ]

    matrix = preprocessor.transform(
        engineered_data
    )

    return matrix


# ============================================================
# Final Prediction
# ============================================================

def predict(row, bundle):

    # -----------------------------
    # Clean raw patient input
    # -----------------------------
    data = prepare_input(
        row,
        bundle
    )

    # -----------------------------
    # LightGBM pipeline
    # -----------------------------
    X_lgbm = preprocess_for_lightgbm(
        data,
        bundle
    )

    lightgbm_model = bundle[
        "lightgbm_model"
    ]

    probability_lgbm = float(
        lightgbm_model.predict_proba(
            X_lgbm
        )[0, 1]
    )

    # -----------------------------
    # XGBoost + engineered features
    # -----------------------------
    X_xgb = preprocess_for_xgboost(
        data,
        bundle
    )

    xgboost_model = bundle[
        "xgboost_fe_model"
    ]

    probability_xgb = float(
        xgboost_model.predict_proba(
            X_xgb
        )[0, 1]
    )

    # -----------------------------
    # Weighted Ensemble
    # -----------------------------
    lightgbm_weight = float(
        bundle["lightgbm_weight"]
    )

    xgboost_weight = float(
        bundle["xgboost_weight"]
    )

    probability = (
        lightgbm_weight * probability_lgbm
        +
        xgboost_weight * probability_xgb
    )

    threshold = float(
        bundle["threshold"]
    )

    prediction = int(
        probability >= threshold
    )

    # -----------------------------
    # Final response
    # -----------------------------
    return {
        "probability": probability,
        "prediction": prediction,
        "threshold": threshold,

        # Useful for debugging / model transparency
        "lightgbm_probability": probability_lgbm,
        "xgboost_probability": probability_xgb,

        "lightgbm_weight": lightgbm_weight,
        "xgboost_weight": xgboost_weight
    }