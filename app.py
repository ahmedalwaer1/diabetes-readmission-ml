import streamlit as st

from inference import load_bundle, predict


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="30-Day Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# Load Final Ensemble Model
# ============================================================

@st.cache_resource
def get_bundle():
    return load_bundle("final_readmission_ensemble.joblib")


try:
    bundle = get_bundle()
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()


# ============================================================
# Header
# ============================================================

st.title("🏥 30-Day Hospital Readmission Prediction")

st.write(
    """
    Enter the patient's hospital encounter information below.
    The model estimates the risk of hospital readmission within 30 days.
    """
)

st.info(
    """
    Final Model: 60% Tuned LightGBM + 40% XGBoost with Engineered Features  
    Classification Threshold: 0.13
    """
)


# ============================================================
# Input Form
# ============================================================

with st.form("patient_form"):

    # --------------------------------------------------------
    # Patient Information
    # --------------------------------------------------------

    st.subheader("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        race = st.selectbox(
            "Race",
            [
                "Caucasian",
                "AfricanAmerican",
                "Asian",
                "Hispanic",
                "Other",
                "Unknown"
            ]
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
                "Unknown"
            ]
        )

    with col3:
        age = st.selectbox(
            "Age Group",
            [
                "[0-10)",
                "[10-20)",
                "[20-30)",
                "[30-40)",
                "[40-50)",
                "[50-60)",
                "[60-70)",
                "[70-80)",
                "[80-90)",
                "[90-100)"
            ],
            index=6
        )


    # --------------------------------------------------------
    # Admission Information
    # --------------------------------------------------------

    st.divider()
    st.subheader("🏨 Hospital Admission Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        admission_type_id = st.number_input(
            "Admission Type ID",
            min_value=1,
            max_value=20,
            value=1,
            step=1
        )

    with col2:
        discharge_disposition_id = st.number_input(
            "Discharge Disposition ID",
            min_value=1,
            max_value=40,
            value=1,
            step=1
        )

    with col3:
        admission_source_id = st.number_input(
            "Admission Source ID",
            min_value=1,
            max_value=30,
            value=7,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        time_in_hospital = st.number_input(
            "Time in Hospital (Days)",
            min_value=1,
            max_value=30,
            value=4,
            step=1
        )

    with col2:
        payer_code = st.text_input(
            "Payer Code",
            value="MC"
        )

    with col3:
        medical_specialty = st.text_input(
            "Medical Specialty",
            value="InternalMedicine"
        )


    # --------------------------------------------------------
    # Clinical Activity
    # --------------------------------------------------------

    st.divider()
    st.subheader("🧪 Clinical Activity")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_lab_procedures = st.number_input(
            "Number of Lab Procedures",
            min_value=0,
            max_value=200,
            value=45,
            step=1
        )

    with col2:
        num_procedures = st.number_input(
            "Number of Procedures",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with col3:
        num_medications = st.number_input(
            "Number of Medications",
            min_value=0,
            max_value=100,
            value=15,
            step=1
        )

    number_diagnoses = st.number_input(
        "Number of Diagnoses",
        min_value=1,
        max_value=30,
        value=8,
        step=1
    )


    # --------------------------------------------------------
    # Previous Hospital Utilization
    # --------------------------------------------------------

    st.divider()
    st.subheader("🔁 Previous Hospital Visits")

    col1, col2, col3 = st.columns(3)

    with col1:
        number_outpatient = st.number_input(
            "Previous Outpatient Visits",
            min_value=0,
            max_value=50,
            value=0,
            step=1
        )

    with col2:
        number_emergency = st.number_input(
            "Previous Emergency Visits",
            min_value=0,
            max_value=50,
            value=1,
            step=1
        )

    with col3:
        number_inpatient = st.number_input(
            "Previous Inpatient Visits",
            min_value=0,
            max_value=50,
            value=2,
            step=1
        )


    # --------------------------------------------------------
    # Diagnoses
    # --------------------------------------------------------

    st.divider()
    st.subheader("🩺 Diagnoses")

    st.caption(
        "Enter ICD diagnosis codes, for example: 428, 250.02, 401"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        diag_1 = st.text_input(
            "Primary Diagnosis (diag_1)",
            value="428"
        )

    with col2:
        diag_2 = st.text_input(
            "Secondary Diagnosis (diag_2)",
            value="250.02"
        )

    with col3:
        diag_3 = st.text_input(
            "Third Diagnosis (diag_3)",
            value="401"
        )


    # --------------------------------------------------------
    # Laboratory Results
    # --------------------------------------------------------

    st.divider()
    st.subheader("🧬 Laboratory Results")

    col1, col2 = st.columns(2)

    with col1:
        max_glu_serum = st.selectbox(
            "Max Glucose Serum Result",
            [
                "Not_Measured",
                "Norm",
                ">200",
                ">300"
            ]
        )

    with col2:
        A1Cresult = st.selectbox(
            "A1C Result",
            [
                "Not_Measured",
                "Norm",
                ">7",
                ">8"
            ]
        )


    # --------------------------------------------------------
    # Diabetes Medications
    # --------------------------------------------------------

    st.divider()
    st.subheader("💊 Diabetes Medications")

    medication_options = [
        "No",
        "Steady",
        "Up",
        "Down"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        metformin = st.selectbox(
            "Metformin",
            medication_options
        )

        repaglinide = st.selectbox(
            "Repaglinide",
            medication_options
        )

        nateglinide = st.selectbox(
            "Nateglinide",
            medication_options
        )

        chlorpropamide = st.selectbox(
            "Chlorpropamide",
            medication_options
        )

        glimepiride = st.selectbox(
            "Glimepiride",
            medication_options
        )

        acetohexamide = st.selectbox(
            "Acetohexamide",
            medication_options
        )

    with col2:
        glipizide = st.selectbox(
            "Glipizide",
            medication_options
        )

        glyburide = st.selectbox(
            "Glyburide",
            medication_options
        )

        tolbutamide = st.selectbox(
            "Tolbutamide",
            medication_options
        )

        pioglitazone = st.selectbox(
            "Pioglitazone",
            medication_options
        )

        rosiglitazone = st.selectbox(
            "Rosiglitazone",
            medication_options
        )

        acarbose = st.selectbox(
            "Acarbose",
            medication_options
        )

    with col3:
        miglitol = st.selectbox(
            "Miglitol",
            medication_options
        )

        troglitazone = st.selectbox(
            "Troglitazone",
            medication_options
        )

        tolazamide = st.selectbox(
            "Tolazamide",
            medication_options
        )

        insulin = st.selectbox(
            "Insulin",
            medication_options,
            index=1
        )

        glyburide_metformin = st.selectbox(
            "Glyburide-Metformin",
            medication_options
        )

        glipizide_metformin = st.selectbox(
            "Glipizide-Metformin",
            medication_options
        )


    st.subheader("Additional Medication Combinations")

    col1, col2, col3 = st.columns(3)

    with col1:
        glimepiride_pioglitazone = st.selectbox(
            "Glimepiride-Pioglitazone",
            medication_options
        )

    with col2:
        metformin_rosiglitazone = st.selectbox(
            "Metformin-Rosiglitazone",
            medication_options
        )

    with col3:
        metformin_pioglitazone = st.selectbox(
            "Metformin-Pioglitazone",
            medication_options
        )


    # --------------------------------------------------------
    # Medication Status
    # --------------------------------------------------------

    st.divider()
    st.subheader("📋 Medication Status")

    col1, col2 = st.columns(2)

    with col1:
        change = st.selectbox(
            "Diabetes Medication Changed?",
            ["No", "Ch"]
        )

    with col2:
        diabetesMed = st.selectbox(
            "Currently on Diabetes Medication?",
            ["Yes", "No"]
        )


    # --------------------------------------------------------
    # Submit Button
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "🔍 Predict Readmission Risk",
        type="primary",
        use_container_width=True
    )


# ============================================================
# Prediction
# ============================================================

if submitted:

    encounter = {
        "race": race,
        "gender": gender,
        "age": age,

        "admission_type_id": str(admission_type_id),
        "discharge_disposition_id": str(discharge_disposition_id),
        "admission_source_id": str(admission_source_id),

        "time_in_hospital": int(time_in_hospital),

        "payer_code": payer_code,
        "medical_specialty": medical_specialty,

        "num_lab_procedures": int(num_lab_procedures),
        "num_procedures": int(num_procedures),
        "num_medications": int(num_medications),

        "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency),
        "number_inpatient": int(number_inpatient),

        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,

        "number_diagnoses": int(number_diagnoses),

        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,

        "metformin": metformin,
        "repaglinide": repaglinide,
        "nateglinide": nateglinide,
        "chlorpropamide": chlorpropamide,
        "glimepiride": glimepiride,
        "acetohexamide": acetohexamide,

        "glipizide": glipizide,
        "glyburide": glyburide,
        "tolbutamide": tolbutamide,
        "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone,
        "acarbose": acarbose,

        "miglitol": miglitol,
        "troglitazone": troglitazone,
        "tolazamide": tolazamide,

        "insulin": insulin,

        "glyburide-metformin": glyburide_metformin,
        "glipizide-metformin": glipizide_metformin,

        "glimepiride-pioglitazone":
            glimepiride_pioglitazone,

        "metformin-rosiglitazone":
            metformin_rosiglitazone,

        "metformin-pioglitazone":
            metformin_pioglitazone,

        "change": change,
        "diabetesMed": diabetesMed
    }

    try:

        result = predict(
            encounter,
            bundle
        )

        probability = result["probability"]
        prediction = result["prediction"]
        threshold = result["threshold"]

        st.divider()

        st.header("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Higher Predicted Risk of "
                "Readmission Within 30 Days"
            )

        else:

            st.success(
                "✅ Lower Predicted Risk of "
                "Readmission Within 30 Days"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Readmission Probability",
                f"{probability:.2%}"
            )

        with col2:

            st.metric(
                "Decision Threshold",
                f"{threshold:.2%}"
            )

        st.progress(
            min(
                max(
                    float(probability),
                    0.0
                ),
                1.0
            )
        )

        st.caption(
            "The classification is based on the final "
            "validation-selected decision threshold."
        )


        # ----------------------------------------------------
        # Model Details
        # ----------------------------------------------------

        with st.expander("🔬 Model Details"):

            st.write(
                f"""
                **Tuned LightGBM Probability:**  
                {result['lightgbm_probability']:.4f}

                **XGBoost Probability:**  
                {result['xgboost_probability']:.4f}

                **LightGBM Weight:**  
                {result['lightgbm_weight']:.2f}

                **XGBoost Weight:**  
                {result['xgboost_weight']:.2f}
                """
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
    Educational machine learning project only.
    This prediction should not be used as a substitute
    for professional clinical judgment.
    """
)