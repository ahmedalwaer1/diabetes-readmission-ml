import json
import streamlit as st

from inference import load_bundle, predict


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="30-Day Readmission Prediction",
    page_icon="🏥",
    layout="centered"
)


# ============================================================
# Load Final Ensemble Once
# ============================================================

@st.cache_resource
def get_bundle():
    return load_bundle(
        "final_readmission_ensemble.joblib"
    )


# ============================================================
# Header
# ============================================================

st.title("🏥 30-Day Hospital Readmission Prediction")

st.write(
    """
    This application predicts the probability of hospital readmission
    within 30 days using the final weighted ensemble model.
    """
)

st.info(
    """
    Final Model:
    60% Tuned LightGBM + 40% XGBoost with Engineered Features

    Classification Threshold: 0.13
    """
)


# ============================================================
# Load Model
# ============================================================

try:
    bundle = get_bundle()

    st.success(
        "Final ensemble model loaded successfully."
    )

except Exception as exc:
    st.error(
        f"Failed to load model: {exc}"
    )

    st.stop()


# ============================================================
# Input Section
# ============================================================

st.subheader("Patient Encounter Input")

st.write(
    """
    Paste one patient encounter as JSON using the original dataset
    fields used during training.

    IDs and the target variable should not be included.
    """
)

default_encounter = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "admission_type_id": "1",
    "discharge_disposition_id": "1",
    "admission_source_id": "7",
    "time_in_hospital": 4,
    "payer_code": "MC",
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "diag_1": "428",
    "diag_2": "250.02",
    "diag_3": "401",
    "number_diagnoses": 8,
    "max_glu_serum": "Not_Measured",
    "A1Cresult": "Not_Measured",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "insulin": "Steady",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "Yes"
}

raw = st.text_area(
    "Encounter JSON",
    value=json.dumps(default_encounter, indent=2),
    height=500
)


# ============================================================
# Prediction
# ============================================================

if st.button(
    "Predict Readmission Risk",
    type="primary"
):

    try:
        encounter = json.loads(raw)

        outcome = predict(
            encounter,
            bundle
        )

        probability = outcome["probability"]
        prediction = outcome["prediction"]
        threshold = outcome["threshold"]

        st.divider()

        st.subheader("Prediction Result")

        # ----------------------------------------
        # Main prediction
        # ----------------------------------------

        if prediction == 1:
            st.warning(
                "⚠️ Higher Risk of Readmission Within 30 Days"
            )

        else:
            st.success(
                "✅ Lower Predicted Risk of Readmission Within 30 Days"
            )

        # ----------------------------------------
        # Probability display
        # ----------------------------------------

        st.metric(
            label="Predicted Readmission Probability",
            value=f"{probability:.2%}"
        )

        st.progress(
            min(max(float(probability), 0.0), 1.0)
        )

        st.write(
            f"Decision Threshold: **{threshold:.2f}**"
        )

        # ----------------------------------------
        # Ensemble model details
        # ----------------------------------------

        with st.expander(
            "Model Details"
        ):

            st.write(
                "### Ensemble Components"
            )

            st.write(
                f"""
                **Tuned LightGBM Probability:**  
                {outcome['lightgbm_probability']:.4f}

                **XGBoost Probability:**  
                {outcome['xgboost_probability']:.4f}

                **LightGBM Weight:**  
                {outcome['lightgbm_weight']:.2f}

                **XGBoost Weight:**  
                {outcome['xgboost_weight']:.2f}
                """
            )

            st.write(
                "### Raw Prediction Output"
            )

            st.json(outcome)

    except json.JSONDecodeError:
        st.error(
            "Invalid JSON. Please check the input format."
        )

    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        TypeError
    ) as exc:

        st.error(
            str(exc)
        )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    """
    Educational machine learning project.
    Predictions should not be used as a substitute for clinical judgment.
    """
)