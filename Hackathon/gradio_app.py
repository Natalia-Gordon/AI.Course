import gradio as gr
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64


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
    feeling_anxious,
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
        feeling_anxious: 'Yes' or 'No'
        guilt: 'Yes', 'No', or 'Maybe'
        bonding: 'Yes', 'No', or 'Sometimes'
        suicide_attempt: 'Yes', 'No', or 'Not interested to say'

    Returns:
        tuple: (risk_score, feature_importance)
    """
    # Filter out composite target column if present (should not be in feature_columns)
    feature_columns_filtered = [col for col in feature_columns if col != "PPD_Composite"]
    
    # Create input row matching the exact structure used during training
    row_dict = {}
    for col in feature_columns_filtered:
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
            row_dict[col] = "" if feeling_anxious is None else str(feeling_anxious).strip()
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

    # Validate that all required columns are present in row_dict
    missing_cols = set(feature_columns_filtered) - set(row_dict.keys())
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns: {missing_cols}. "
            "This indicates a bug in the feature mapping logic."
        )

    # Create DataFrame ensuring exact column order and dtypes match training data
    # Use feature_columns_filtered to avoid including target column
    row = pd.DataFrame([row_dict], columns=feature_columns_filtered)

    # Validate for NaN values before type conversion
    if row.isna().any().any():
        nan_cols = row.columns[row.isna().any()].tolist()
        raise ValueError(
            f"Found NaN values in feature columns: {nan_cols}. "
            "All feature values must be provided."
        )

    # Convert dtypes to match training data exactly (critical for OneHotEncoder)
    for col in row.columns:
        if col in feature_dtypes:
            target_dtype = feature_dtypes[col]
            if target_dtype == "object":
                # Ensure categorical columns are object dtype (string)
                # Note: Empty strings are acceptable for categorical features
                # Fill any NaN values with empty string before conversion (defensive check)
                row[col] = row[col].fillna("").astype(str)
                # Replace any "nan" strings that might have been created from NaN conversion
                row[col] = row[col].replace("nan", "", regex=False)

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
    #displayed_risk = 1.0 - proba
    risk_score = f"PPD Risk Score: {proba:.2%}"

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

        # Format feature importance in user-friendly way
        feat_imp_lines = []
        for i, (feat, val) in enumerate(feat_imp, 1):
            # Clean up feature name (remove one-hot encoding prefixes)
            clean_feat = feat.split('__')[-1] if '__' in feat else feat
            direction = "increases" if val > 0 else "decreases"
            impact = "high" if abs(val) > 0.1 else "moderate" if abs(val) > 0.05 else "low"
            feat_imp_lines.append(
                f"{i}. {clean_feat}\n   Impact: {impact} ({direction} risk by {abs(val):.3f})"
            )
        feat_imp_str = "\n\n".join(feat_imp_lines)

        # Generate personalized explanation
        personalized_explanation = generate_personalized_explanation(
            proba, pred_class, feat_imp, row, feature_names, shap_values_single
        )

        # Create user-friendly SHAP summary plot
        plot_html = create_shap_summary_plot(feat_imp, shap_values_single, feature_names)

    except Exception as e:
        feat_imp_str = f"SHAP explanation unavailable: {str(e)}"
        personalized_explanation = "Unable to generate personalized explanation."
        plot_html = ""

    return risk_score, feat_imp_str, personalized_explanation, plot_html


def generate_personalized_explanation(proba, pred_class, top_features, row, feature_names, shap_values_single):
    """
    Generate a personalized, user-friendly explanation based on SHAP values and predictions.
    """
    # Determine risk level
    if proba >= 0.7:
        risk_level = "high"
        risk_phrase = "identifies a high risk"
    elif proba >= 0.4:
        risk_level = "moderate"
        risk_phrase = "identifies a moderate risk"
    else:
        risk_level = "low"
        risk_phrase = "identifies a low risk"
    
    # Get top contributing features (positive SHAP = increases risk)
    risk_increasing = [(feat, val) for feat, val in top_features if val > 0]
    
    # Clean feature names and map to user-friendly descriptions
    feature_descriptions = {
        'Feeling sad or Tearful': 'feeling sad or tearful',
        'Irritable towards baby & partner': 'irritability towards baby and partner',
        'Trouble sleeping at night': 'trouble sleeping',
        'Problems concentrating or making decision': 'concentration problems',
        'Overeating or loss of appetite': 'appetite changes',
        'Feeling anxious': 'anxiety',
        'Feeling of guilt': 'feelings of guilt',
        'Problems of bonding with baby': 'bonding difficulties with baby',
        'Suicide attempt': 'suicidal thoughts or attempts',
        'Age': 'age'
    }
    
    # Build explanation
    explanation_parts = [f"The model {risk_phrase} in you"]
    
    if risk_increasing:
        # Get the actual input values for these features
        contributing_factors = []
        for feat, shap_val in risk_increasing[:3]:  # Top 3 risk-increasing factors
            clean_feat = feat.split('__')[-1] if '__' in feat else feat
            # Check if this is a one-hot encoded feature
            if '__' in feat:
                # Extract the original feature name and value
                parts = feat.split('__')
                if len(parts) >= 2:
                    orig_feat = parts[0]
                    feat_value = parts[-1]
                    # Get the actual input value
                    if orig_feat in row.columns:
                        input_val = str(row[orig_feat].values[0])
                        if input_val == feat_value or (feat_value in input_val):
                            desc = feature_descriptions.get(orig_feat, orig_feat.lower())
                            contributing_factors.append(desc)
            else:
                desc = feature_descriptions.get(clean_feat, clean_feat.lower())
                contributing_factors.append(desc)
        
        if contributing_factors:
            if len(contributing_factors) == 1:
                explanation_parts.append(f"mainly due to {contributing_factors[0]}")
            elif len(contributing_factors) == 2:
                explanation_parts.append(f"mainly due to the combination of {contributing_factors[0]} and {contributing_factors[1]}")
            else:
                explanation_parts.append(
                    f"mainly due to the combination of {', '.join(contributing_factors[:-1])}, and {contributing_factors[-1]}"
                )
    
    explanation = ". ".join(explanation_parts) + "."
    
    # Add risk score context
    if risk_level == "high":
        explanation += " Please consider consulting with a healthcare professional."
    elif risk_level == "moderate":
        explanation += " It may be helpful to monitor your symptoms and seek support if needed."
    
    return explanation


def create_shap_summary_plot(top_features, shap_values_single, feature_names):
    """
    Create a user-friendly SHAP summary plot as HTML.
    """
    try:
        # Extract feature names and values
        feat_names = [feat.split('__')[-1] if '__' in feat else feat for feat, _ in top_features]
        shap_vals = [val for _, val in top_features]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if val > 0 else '#2ecc71' for val in shap_vals]
        y_pos = np.arange(len(feat_names))
        
        bars = ax.barh(y_pos, shap_vals, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names, fontsize=10)
        ax.set_xlabel('SHAP Value (Impact on Risk)', fontsize=12, fontweight='bold')
        ax.set_title('Top 5 Feature Contributions to PPD Risk', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shap_vals)):
            width = bar.get_width()
            label_x = width + (0.01 if width > 0 else -0.01)
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left' if width > 0 else 'right', 
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64 HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'
    except Exception as e:
        return f"<p>Unable to generate plot: {str(e)}</p>"


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
        feeling_anxious,
        guilt,
        bonding,
        suicide_attempt,
    ):
        if explainer is None:
            return (
                "SHAP explainer not available",
                "Please train the model first",
                "Unable to generate explanation.",
                ""
            )
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
            feeling_anxious,
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
                concentration = gr.Radio(
                    label="Problems concentrating or making decision",
                    choices=["Yes", "No", "Often"],
                    value="No",
                )

            with gr.Column():
                appetite = gr.Radio(
                    label="Overeating or loss of appetite",
                    choices=["Yes", "No", "Not at all"],
                    value="No",
                )
                feeling_anxious = gr.Radio(
                    label="Feeling anxious",
                    choices=["Yes", "No"],
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
            personalized_explanation = gr.Textbox(
                label="Personalized Explanation",
                interactive=False,
                lines=4,
            )

        with gr.Row():
            feature_importance = gr.Textbox(
                label="Top 5 Feature Contributions (SHAP)",
                interactive=False,
                lines=8,
            )
            shap_plot = gr.HTML(label="SHAP Summary Plot")

        predict_btn.click(
            fn=predict_wrapper,
            inputs=[
                age,
                feeling_sad,
                irritable,
                trouble_sleeping,
                concentration,
                appetite,
                feeling_anxious,
                guilt,
                bonding,
                suicide_attempt,
            ],
            outputs=[risk_output, feature_importance, personalized_explanation, shap_plot],
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
                feeling_anxious,
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