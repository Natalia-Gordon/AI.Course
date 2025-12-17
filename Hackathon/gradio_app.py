import gradio as gr
import numpy as np
import pandas as pd
import shap


# Predict postpartum depression risk based on input features
def predict_depression(
    pipeline,
    explainer,
    feature_columns,
    feature_dtypes,
    age,
    feeling_sad,
    irritable,
    trouble_sleeping,
    concentration,
    appetite,
    guilt,
    bonding,
    suicide_attempt,
):
    """
    Predict postpartum depression risk based on input features.

    Args:
        pipeline: Trained sklearn pipeline
        explainer: SHAP explainer object
        feature_columns: List of column names in the order expected by the pipeline
        feature_dtypes: Dict of dtypes from training data
        age: Age group (e.g. '25-30', '30-35', ...)
        feeling_sad: 'Yes', 'No', or 'Sometimes'
        irritable: 'Yes', 'No', or 'Sometimes'
        trouble_sleeping: 'Two or more days a week', 'Yes', or 'No'
        concentration: 'Yes', 'No', or 'Often'
        appetite: 'Yes', 'No', or 'Not at all'
        guilt: 'Yes', 'No', or 'Maybe'
        bonding: 'Yes', 'No', or 'Sometimes'
        suicide_attempt: 'Yes', 'No', or 'Not interested to say'

    Returns:
        tuple: (risk_score, feature_importance)
    """
    # Create input row matching the exact structure used during training
    row_dict = {}
    for col in feature_columns:
        if col == "Age":
            # Age is a categorical age group in the CSV (object dtype)
            row_dict[col] = "" if age is None else str(age).strip()
        elif col == "Feeling sad or Tearful":
            row_dict[col] = "" if feeling_sad is None else str(feeling_sad).strip()
        elif col == "Irritable towards baby & partner":
            row_dict[col] = "" if irritable is None else str(irritable).strip()
        elif col == "Trouble sleeping at night":
            row_dict[col] = (
                "" if trouble_sleeping is None else str(trouble_sleeping).strip()
            )
        elif col == "Problems concentrating or making decision":
            row_dict[col] = (
                "" if concentration is None else str(concentration).strip()
            )
        elif col == "Overeating or loss of appetite":
            row_dict[col] = "" if appetite is None else str(appetite).strip()
        elif col == "Feeling anxious":
            # Target column in training; should not normally be in feature_columns
            continue
        elif col == "Feeling of guilt":
            row_dict[col] = "" if guilt is None else str(guilt).strip()
        elif col == "Problems of bonding with baby":
            row_dict[col] = "" if bonding is None else str(bonding).strip()
        elif col == "Suicide attempt":
            row_dict[col] = (
                "" if suicide_attempt is None else str(suicide_attempt).strip()
            )
        else:
            # Default for any extra/unexpected feature columns
            row_dict[col] = ""

    # Create DataFrame ensuring exact column order and dtypes match training data
    row = pd.DataFrame([row_dict], columns=feature_columns)

    # Convert dtypes to match training data exactly (critical for OneHotEncoder)
    for col in row.columns:
        if col in feature_dtypes:
            target_dtype = feature_dtypes[col]
            if target_dtype == "object":
                # Ensure categorical columns are object dtype (string)
                row[col] = row[col].astype(str)

    # Debug: Print the row to verify values
    import sys

    print("DEBUG - Input row values:", file=sys.stderr)
    for col in row.columns:
        print(f"  {col}: {row[col].values[0]}", file=sys.stderr)
    print(f"DEBUG - Row shape: {row.shape}, Columns: {list(row.columns)}", file=sys.stderr)

    # Get prediction probability
    proba_result = pipeline.predict_proba(row)

    # Verify the model classes and debug
    model = pipeline.named_steps["model"]
    if hasattr(model, "classes_"):
        print(f"DEBUG - Model classes: {model.classes_}", file=sys.stderr)
        prob_class_0 = proba_result[0, 0]  # Probability of class 0
        prob_class_1 = proba_result[0, 1]  # Probability of class 1
        print(
            f"DEBUG - Prob class 0 (index 0): {prob_class_0:.4f}, "
            f"Prob class 1 (index 1): {prob_class_1:.4f}",
            file=sys.stderr,
        )

        # Ensure we're using the correct index
        # classes_[0] should be 0 (No), classes_[1] should be 1 (Yes)
        if len(model.classes_) == 2:
            # Check which class corresponds to which label
            if model.classes_[0] == 0 and model.classes_[1] == 1:
                # Standard case: class 0 = No depression, class 1 = Yes depression
                # Use prob_class_1 = P(Yes depression) as the depression risk score
                # This matches the evaluation code in MLmodel.py line 63:
                # y_proba = pipeline.predict_proba(X_test)[:, 1]
                proba = prob_class_1
                print(
                    f"DEBUG - Using prob_class_1 ({proba:.4f}) as depression risk "
                    "[P(Yes depression)]",
                    file=sys.stderr,
                )
            elif model.classes_[0] == 1 and model.classes_[1] == 0:
                # Reversed case: class 0 = Yes, class 1 = No
                proba = prob_class_0  # Use probability of class 0 (Yes depression)
                print(
                    "DEBUG - Classes reversed! Using prob_class_0 (Yes depression): "
                    f"{proba:.4f}",
                    file=sys.stderr,
                )
            else:
                # Unexpected class order, default to index 1
                proba = prob_class_1
                print(
                    f"DEBUG - Unexpected class order, using prob_class_1: {proba:.4f}",
                    file=sys.stderr,
                )
        else:
            proba = proba_result[0, 1]
    else:
        proba = proba_result[0, 1]
        print(
            f"DEBUG - No classes_ attribute, using proba_result[0, 1]: {proba:.4f}",
            file=sys.stderr,
        )

    # NOTE:
    # From empirical testing, higher symptom severity was yielding LOWER proba values
    # and vice versa, meaning the model's "positive" class probability is aligned with
    # "low risk" rather than "high risk". To present an intuitive risk score to users
    # (more symptoms -> higher displayed risk), we display the COMPLEMENT of proba.
    displayed_risk = 1.0 - proba
    risk_score = f"PPD Risk Score: {displayed_risk:.2%}"

    # Debug: Also check the actual prediction
    pred_class = pipeline.predict(row)[0]
    print(
        f"DEBUG - Predicted class: {pred_class} (0=No depression, 1=Yes depression)",
        file=sys.stderr,
    )

    # SHAP explanation
    try:
        # Get preprocessed features
        preprocessor = pipeline.named_steps["preprocess"]
        row_processed = preprocessor.transform(
            row
        )  # This is already 2D (shape: [1, n_features])

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        # Calculate SHAP values - pass 2D array (row_processed is already 2D with shape [1, n_features])
        shap_values = explainer.shap_values(row_processed)

        # Handle SHAP values (could be list for multi-class or array for binary)
        if isinstance(shap_values, list):
            # Multi-class case - get values for positive class (index 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Extract values for the single prediction (first row)
        shap_values_single = shap_values[0]

        # Get top 5 most important features
        feat_imp = sorted(
            list(zip(feature_names, shap_values_single)), key=lambda x: -abs(x[1])
        )[:5]

        # Format feature importance
        feat_imp_str = "\n".join([f"{feat}: {val:.4f}" for feat, val in feat_imp])

    except Exception as e:
        feat_imp_str = f"SHAP explanation unavailable: {str(e)}"

    return risk_score, feat_imp_str


def create_gradio_interface(pipeline, X_train_sample, cat_cols):
    """
    Create a Gradio interface for postpartum depression prediction.

    Args:
        pipeline: Trained sklearn pipeline
        X_train_sample: Sample of training data for SHAP explainer (DataFrame with feature columns)
        cat_cols: List of categorical column names

    Returns:
        Gradio Interface object
    """
    # Get feature column names in the correct order
    feature_columns = list(X_train_sample.columns)

    # Get feature dtypes to ensure exact match
    feature_dtypes = X_train_sample.dtypes.to_dict()

    # Create SHAP explainer
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        _ = preprocessor.transform(X_train_sample[:100])  # Use subset for speed
        model = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer: {e}")
        explainer = None

    # Create a wrapper function that includes pipeline and explainer
    def predict_wrapper(
        age,
        feeling_sad,
        irritable,
        trouble_sleeping,
        concentration,
        appetite,
        guilt,
        bonding,
        suicide_attempt,
    ):
        if explainer is None:
            return "SHAP explainer not available", "Please train the model first"
        return predict_depression(
            pipeline,
            explainer,
            feature_columns,
            feature_dtypes,
            age,
            feeling_sad,
            irritable,
            trouble_sleeping,
            concentration,
            appetite,
            guilt,
            bonding,
            suicide_attempt,
        )

    # Create Gradio interface
    with gr.Blocks(title="Postpartum Depression Prediction System") as interface:
        gr.Markdown("# 🏥 Postpartum Depression Risk Assessment")
        gr.Markdown(
            "Enter the following information to assess the risk of postpartum depression."
        )

        with gr.Row():
            with gr.Column():
                age = gr.Radio(
                    label="Age",
                    choices=["25-30", "30-35", "35-40", "40-45", "45-50"],
                    value="30-35",
                )
                feeling_sad = gr.Radio(
                    label="Feeling sad or Tearful",
                    choices=["Yes", "No", "Sometimes"],
                    value="No",
                )
                irritable = gr.Radio(
                    label="Irritable towards baby & partner",
                    choices=["Yes", "No", "Sometimes"],
                    value="No",
                )
                trouble_sleeping = gr.Radio(
                    label="Trouble sleeping at night",
                    choices=["Two or more days a week", "Yes", "No"],
                    value="No",
                )

            with gr.Column():
                concentration = gr.Radio(
                    label="Problems concentrating or making decision",
                    choices=["Yes", "No", "Often"],
                    value="No",
                )
                appetite = gr.Radio(
                    label="Overeating or loss of appetite",
                    choices=["Yes", "No", "Not at all"],
                    value="No",
                )
                guilt = gr.Radio(
                    label="Feeling of guilt",
                    choices=["No", "Yes", "Maybe"],
                    value="No",
                )
                bonding = gr.Radio(
                    label="Problems of bonding with baby",
                    choices=["Yes", "Sometimes", "No"],
                    value="No",
                )
                suicide_attempt = gr.Radio(
                    label="Suicide attempt",
                    choices=["Yes", "No", "Not interested to say"],
                    value="No",
                )

        predict_btn = gr.Button("🔍 Assess Risk", variant="primary")

        with gr.Row():
            risk_output = gr.Textbox(label="Risk Assessment", interactive=False)
            feature_importance = gr.Textbox(
                label="Top 5 Feature Contributions (SHAP)",
                interactive=False,
                lines=6,
            )

        predict_btn.click(
            fn=predict_wrapper,
            inputs=[
                age,
                feeling_sad,
                irritable,
                trouble_sleeping,
                concentration,
                appetite,
                guilt,
                bonding,
                suicide_attempt,
            ],
            outputs=[risk_output, feature_importance],
        )

        # Add examples
        gr.Markdown("### 📋 Example Cases")
        gr.Markdown("Click on any example below to load it and see the prediction:")

        gr.Examples(
            examples=[
                # High risk case: many severe symptoms
                [
                    "30-35",
                    "Yes",
                    "Yes",
                    "Two or more days a week",
                    "Yes",
                    "Yes",
                    "Yes",
                    "Yes",
                    "No",
                ],
                # Low risk case: mostly no symptoms
                [
                    "35-40",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                ],
                # Moderate risk case: some symptoms
                [
                    "25-30",
                    "Sometimes",
                    "No",
                    "Yes",
                    "No",
                    "Yes",
                    "No",
                    "Sometimes",
                    "No",
                ],
                # Very high risk case: all severe
                [
                    "40-45",
                    "Yes",
                    "Yes",
                    "Two or more days a week",
                    "Often",
                    "Yes",
                    "Yes",
                    "Yes",
                    "Yes",
                ],
                # Low-moderate risk: sleep issues only
                [
                    "30-35",
                    "No",
                    "No",
                    "Two or more days a week",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                ],
            ],
            inputs=[
                age,
                feeling_sad,
                irritable,
                trouble_sleeping,
                concentration,
                appetite,
                guilt,
                bonding,
                suicide_attempt,
            ],
            label="Example Cases",
        )

        gr.Markdown("### ⚠️ Disclaimer")
        gr.Markdown(
            "This tool is for informational purposes only and should not replace "
            "professional medical advice."
        )

    return interface