from fastapi import FastAPI, HTTPException
from inference import load_bundle, predict


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="30-Day Hospital Readmission Prediction API",
    description=(
        "Predicts the risk of hospital readmission within 30 days "
        "using a weighted LightGBM + XGBoost ensemble."
    ),
    version="2.0"
)


# ============================================================
# Load Final Ensemble Once
# ============================================================

try:
    bundle = load_bundle(
        "final_readmission_ensemble.joblib"
    )

    bundle_loaded = True
    bundle_error = None

except Exception as exc:
    bundle = None
    bundle_loaded = False
    bundle_error = str(exc)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "30-Day Hospital Readmission Prediction API",
        "model": "Weighted LightGBM + XGBoost Ensemble",
        "status": "ready" if bundle_loaded else "model_not_loaded"
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    if bundle_loaded:
        return {
            "status": "healthy",
            "model_loaded": True,
            "lightgbm_weight": bundle["lightgbm_weight"],
            "xgboost_weight": bundle["xgboost_weight"],
            "threshold": bundle["threshold"]
        }

    return {
        "status": "unhealthy",
        "model_loaded": False,
        "error": bundle_error
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict_endpoint(encounter: dict):

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Final ensemble model is not available. "
                f"Error: {bundle_error}"
            )
        )

    try:
        result = predict(
            encounter,
            bundle
        )

        return {
            "status": "success",
            "prediction": result["prediction"],
            "probability": result["probability"],
            "threshold": result["threshold"],

            "model_details": {
                "lightgbm_probability":
                    result["lightgbm_probability"],

                "xgboost_probability":
                    result["xgboost_probability"],

                "lightgbm_weight":
                    result["lightgbm_weight"],

                "xgboost_weight":
                    result["xgboost_weight"]
            }
        }

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc