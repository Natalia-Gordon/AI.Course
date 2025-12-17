import gradio as gr
import numpy as np
import pandas as pd
import shap

# Predict postpartum depression risk based on input features
def predict_depression(pipeline, explainer, feature_columns, feature_dtypes, age, feeling_sad, irritable, trouble_sleeping, 
                       concentration, appetite, guilt, bonding, suicide_attempt):
    """
    Predict postpartum depression risk based on input features.
    
    Args:
        pipeline: Trained sklearn pipeline
        explainer: SHAP explainer object
        feature_columns: List of column names in the order expected by the pipeline
        age: Age value
        feeling_sad: "Yes" or "No"
        irritable: "Yes" or "No"
        trouble_sleeping: "Yes" or "No"
        concentration: "Yes" or "No"
        appetite: "Yes" or "No"
        guilt: "Yes" or "No"
        bonding: "Yes" or "No"
        suicide_attempt: "Yes" or "No"
        
    Returns:
        tuple: (risk_score, feature_importance)
    """
    # Create input row matching the exact structure used during training
    # Map inputs to column values, ensuring proper types
    # Normalize inputs to ensure exact "Yes"/"No" matching (case-insensitive, strip whitespace)
    def normalize_yes_no(value):
        """Normalize input to exactly 'Yes' or 'No'"""
        if value is None:
            return "No"
        value_str = str(value).strip()
        if value_str.upper() in ['YES', 'Y', '1', 'TRUE']:
            return "Yes"
        else:
            return "No"
    
    row_dict = {}
    for col in feature_columns:
        if col == "Age":
            # Age should be numeric
            row_dict[col] = float(age) if age is not None and not pd.isna(age) else 0.0
        elif col == "Feeling sad or Tearful":
            row_dict[col] = normalize_yes_no(feeling_sad)
        elif col == "Irritable towards baby & partner":
            row_dict[col] = normalize_yes_no(irritable)
        elif col == "Trouble sleeping at night":
            row_dict[col] = normalize_yes_no(trouble_sleeping)
        elif col == "Problems concentrating or making decision":
            row_dict[col] = normalize_yes_no(concentration)
        elif col == "Overeating or loss of appetite":
            row_dict[col] = normalize_yes_no(appetite)
        elif col == "Feeling of guilt":
            row_dict[col] = normalize_yes_no(guilt)
        elif col == "Problems of bonding with baby":
            row_dict[col] = normalize_yes_no(bonding)
        elif col == "Suicide attempt":
            row_dict[col] = normalize_yes_no(suicide_attempt)
        else:
            row_dict[col] = "No"  # Default for any unexpected columns
    
    # Create DataFrame ensuring exact column order and dtypes match training data
    row = pd.DataFrame([row_dict], columns=feature_columns)
    
    # Convert dtypes to match training data exactly (critical for OneHotEncoder)
    for col in row.columns:
        if col in feature_dtypes:
            target_dtype = feature_dtypes[col]
            if target_dtype == 'object':
                # Ensure categorical columns are object dtype (string)
                row[col] = row[col].astype(str)
            elif col == "Age":
                # Age should be numeric (float or int)
                row[col] = pd.to_numeric(row[col], errors='coerce').fillna(0.0)
                if 'int' in str(target_dtype):
                    row[col] = row[col].astype(int)
                else:
                    row[col] = row[col].astype(float)
    
    # Debug: Print the row to verify values (temporary debugging)
    import sys
    print("DEBUG - Input row values:", file=sys.stderr)
    for col in row.columns:
        print(f"  {col}: {row[col].values[0]}", file=sys.stderr)
    print(f"DEBUG - Row shape: {row.shape}, Columns: {list(row.columns)}", file=sys.stderr)
    
    # Get prediction probability
    # Note: predict_proba returns [prob_class_0, prob_class_1]
    # Target encoding: "Yes"=1, "No"=0 (from main.py line 36)
    # So class 1 = "Feeling anxious" = "Yes" = has depression/anxiety
    proba_result = pipeline.predict_proba(row)
    
    # Verify the model classes and debug
    model = pipeline.named_steps['model']
    if hasattr(model, 'classes_'):
        print(f"DEBUG - Model classes: {model.classes_}", file=sys.stderr)
        prob_class_0 = proba_result[0, 0]  # Probability of class 0
        prob_class_1 = proba_result[0, 1]  # Probability of class 1
        print(f"DEBUG - Prob class 0 (index 0): {prob_class_0:.4f}, Prob class 1 (index 1): {prob_class_1:.4f}", file=sys.stderr)
        
        # Ensure we're using the correct index
        # classes_[0] should be 0 (No), classes_[1] should be 1 (Yes)
        if len(model.classes_) == 2:
            # Check which class corresponds to which label
            # If classes_[0] == 0, then proba_result[0, 0] is P(No depression)
            # If classes_[1] == 1, then proba_result[0, 1] is P(Yes depression)
            if model.classes_[0] == 0 and model.classes_[1] == 1:
                # Standard case: class 0 = No depression, class 1 = Yes depression
                # Use prob_class_1 = P(Yes depression) as the depression risk score
                # This matches the evaluation code in MLmodel.py line 63: y_proba = pipeline.predict_proba(X_test)[:, 1]
                proba = prob_class_1
                print(f"DEBUG - Using prob_class_1 ({proba:.4f}) as depression risk [P(Yes depression)]", file=sys.stderr)
            elif model.classes_[0] == 1 and model.classes_[1] == 0:
                # Reversed case: class 0 = Yes, class 1 = No
                proba = prob_class_0  # Use probability of class 0 (Yes depression)
                print(f"DEBUG - Classes reversed! Using prob_class_0 (Yes depression): {proba:.4f}", file=sys.stderr)
            else:
                # Unexpected class order, default to index 1
                proba = prob_class_1
                print(f"DEBUG - Unexpected class order, using prob_class_1: {proba:.4f}", file=sys.stderr)
        else:
            proba = proba_result[0, 1]
    else:
        proba = proba_result[0, 1]
        print(f"DEBUG - No classes_ attribute, using proba_result[0, 1]: {proba:.4f}", file=sys.stderr)
    
    risk_score = f"PPD Risk Score: {proba:.2%}"
    
    # Debug: Also check the actual prediction
    pred_class = pipeline.predict(row)[0]
    print(f"DEBUG - Predicted class: {pred_class} (0=No depression, 1=Yes depression)", file=sys.stderr)
    
    # SHAP explanation
    try:
        # Get preprocessed features
        preprocessor = pipeline.named_steps['preprocess']
        row_processed = preprocessor.transform(row)  # This is already 2D (shape: [1, n_features])
        
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
            list(zip(feature_names, shap_values_single)), 
            key=lambda x: -abs(x[1])
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
        preprocessor = pipeline.named_steps['preprocess']
        X_sample_processed = preprocessor.transform(X_train_sample[:100])  # Use subset for speed
        model = pipeline.named_steps['model']
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer: {e}")
        explainer = None
    
    # Create a wrapper function that includes pipeline and explainer
    def predict_wrapper(age, feeling_sad, irritable, trouble_sleeping, 
                       concentration, appetite, guilt, bonding, suicide_attempt):
        if explainer is None:
            return "SHAP explainer not available", "Please train the model first"
        return predict_depression(pipeline, explainer, feature_columns, feature_dtypes, age, feeling_sad, irritable, 
                                 trouble_sleeping, concentration, appetite, guilt, 
                                 bonding, suicide_attempt)
    
    # Create Gradio interface
    with gr.Blocks(title="Postpartum Depression Prediction System") as interface:
        gr.Markdown("# 🏥 Postpartum Depression Risk Assessment")
        gr.Markdown("Enter the following information to assess the risk of postpartum depression.")
        
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age", value=30, minimum=15, maximum=50)
                feeling_sad = gr.Radio(label="Feeling sad or Tearful", 
                                      choices=["Yes", "No"], value="No")
                irritable = gr.Radio(label="Irritable towards baby & partner", 
                                   choices=["Yes", "No"], value="No")
                trouble_sleeping = gr.Radio(label="Trouble sleeping at night", 
                                          choices=["Yes", "No"], value="No")
                concentration = gr.Radio(label="Problems concentrating or making decision", 
                                        choices=["Yes", "No"], value="No")
            
            with gr.Column():
                appetite = gr.Radio(label="Overeating or loss of appetite", 
                                  choices=["Yes", "No"], value="No")
                guilt = gr.Radio(label="Feeling of guilt", 
                               choices=["Yes", "No"], value="No")
                bonding = gr.Radio(label="Problems of bonding with baby", 
                                 choices=["Yes", "No"], value="No")
                suicide_attempt = gr.Radio(label="Suicide attempt", 
                                          choices=["Yes", "No"], value="No")
        
        predict_btn = gr.Button("🔍 Assess Risk", variant="primary")
        
        with gr.Row():
            risk_output = gr.Textbox(label="Risk Assessment", interactive=False)
            feature_importance = gr.Textbox(label="Top 5 Feature Contributions (SHAP)", 
                                          interactive=False, lines=6)
        
        predict_btn.click(
            fn=predict_wrapper,
            inputs=[age, feeling_sad, irritable, trouble_sleeping, 
                   concentration, appetite, guilt, bonding, suicide_attempt],
            outputs=[risk_output, feature_importance]
        )
        
        # Add examples
        gr.Markdown("### 📋 Example Cases")
        gr.Markdown("Click on any example below to load it and see the prediction:")
        
        examples = gr.Examples(
            examples=[
                [28, "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],  # High risk case
                [32, "No", "No", "No", "No", "No", "No", "No", "No"],  # Low risk case
                [25, "Yes", "No", "Yes", "No", "No", "No", "No", "No"],  # Moderate risk case
                [35, "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],  # Very high risk case
                [30, "No", "No", "Yes", "No", "No", "No", "No", "No"],  # Low-moderate risk (sleep issues only)
            ],
            inputs=[age, feeling_sad, irritable, trouble_sleeping, 
                   concentration, appetite, guilt, bonding, suicide_attempt],
            label="Example Cases"
        )
        
        gr.Markdown("### ⚠️ Disclaimer")
        gr.Markdown("This tool is for informational purposes only and should not replace professional medical advice.")
    
    return interface