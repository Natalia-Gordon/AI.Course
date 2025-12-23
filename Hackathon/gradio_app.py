import gradio as gr
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import io
import base64
import os
from MLmodel import create_XGBoost_pipeline, create_rf_pipeline, train_and_evaluate, optimize_XGBoost_hyperparameters, optimize_rf_hyperparameters
from sklearn.model_selection import train_test_split


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

    risk_score = f"PPD Risk Score: {proba:.2%}"

    # Debug: Also check the actual prediction
    pred_class = pipeline.predict(row)[0]
    print(
        f"DEBUG - Predicted class: {pred_class} (0=No depression, 1=Yes depression)",
        file=sys.stderr,
    )

    # SHAP explanation (works for both XGBoost and Random Forest)
    try:
        if explainer is None:
            raise ValueError("SHAP explainer not initialized. Please train the model first.")
        
        # Get preprocessed features
        preprocessor = pipeline.named_steps["preprocess"]
        row_processed = preprocessor.transform(
            row
        )  # This is already 2D (shape: [1, n_features])

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        # Calculate SHAP values - pass 2D array (row_processed is already 2D with shape [1, n_features])
        # TreeExplainer works for both XGBoost and Random Forest
        shap_values = explainer.shap_values(row_processed)

        # Handle SHAP values (could be list for multi-class or array for binary)
        # For binary classification, Random Forest may return a list with [class_0_values, class_1_values]
        # or a single array for the positive class
        if isinstance(shap_values, list):
            # Multi-class or binary with list format - get values for positive class (index 1)
            if len(shap_values) > 1:
                shap_values = shap_values[1]  # Use positive class (depression = 1)
            else:
                shap_values = shap_values[0]

        # Convert to numpy array if not already
        shap_values = np.array(shap_values)
        
        # Extract values for the single prediction (first row)
        # Handle different array shapes: [1, n_features] or [n_features]
        if len(shap_values.shape) > 1:
            shap_values_single = shap_values[0]  # First row if 2D
        else:
            shap_values_single = shap_values  # Already 1D
        
        # Ensure it's a 1D array and convert to list for easier handling
        shap_values_single = np.array(shap_values_single).flatten()

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
        # Determine save path based on current model (if available)
        save_path_shap = None
        try:
            model_type = type(pipeline.named_steps['model']).__name__
            algo_name = "XGBoost" if "XGB" in model_type.upper() else "RandomForest"
            save_path_shap = os.path.join("output", "plots", algo_name)
        except:
            pass
        plot_html = create_shap_summary_plot(feat_imp, shap_values_single, feature_names, save_path=save_path_shap)
        
        # Create detailed SHAP explanation
        shap_explanation_lines = []
        for i, (feat, val) in enumerate(feat_imp, 1):
            clean_feat = feat.split('__')[-1] if '__' in feat else feat
            abs_val = abs(val)
            impact = "high" if abs_val > 0.1 else "moderate" if abs_val > 0.05 else "low"
            direction = "increases" if val > 0 else "decreases"
            shap_explanation_lines.append(
                f"**{clean_feat}**: SHAP value = {val:.4f}\n"
                f"  • This feature {direction} the PPD risk prediction by {abs_val:.4f}\n"
                f"  • Impact level: {impact.upper()}\n"
                f"  • {'Positive SHAP value means this feature pushes the prediction toward higher risk.' if val > 0 else 'Negative SHAP value means this feature pushes the prediction toward lower risk.'}"
            )
        
        shap_explanation = f"""## SHAP (SHapley Additive exPlanations) Analysis

SHAP values explain how each feature contributes to the final prediction.
- **Positive SHAP values** push the prediction toward higher PPD risk
- **Negative SHAP values** push the prediction toward lower PPD risk
- **Magnitude** indicates the strength of the contribution

### Feature Contributions:

{chr(10).join(shap_explanation_lines) if shap_explanation_lines else 'No SHAP values available'}

### How to Interpret:
- **High impact** (|SHAP| > 0.1): This feature significantly influences the prediction
- **Moderate impact** (0.05 < |SHAP| ≤ 0.1): This feature moderately influences the prediction
- **Low impact** (|SHAP| ≤ 0.05): This feature has a minor influence on the prediction

The sum of all SHAP values equals the difference between the model's prediction and the baseline (average prediction).
"""

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"SHAP Error: {error_details}", file=sys.stderr)
        feat_imp_str = f"SHAP explanation unavailable: {str(e)}\n\nPlease ensure the model is trained and the SHAP explainer is initialized."
        personalized_explanation = f"Unable to generate personalized explanation. Error: {str(e)}"
        shap_explanation = f"## SHAP Explanation Unavailable\n\nError: {str(e)}\n\nPlease ensure the model is trained and the SHAP explainer is initialized."
        plot_html = ""

    return risk_score, feat_imp_str, personalized_explanation, shap_explanation, plot_html


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


def create_shap_summary_plot(top_features, shap_values_single, feature_names, save_path=None):
    """
    Create a user-friendly SHAP summary plot as HTML.
    
    Args:
        top_features: List of tuples (feature_name, shap_value)
        shap_values_single: Array of SHAP values
        feature_names: List of feature names
        save_path: Optional directory path to save the plot
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
        
        # Save to file if path provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, "shap_summary_bar.png")
            plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
            print(f"   💾 Saved: {filepath}")
        
        # Convert to base64 HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'
    except Exception as e:
        return f"<p>Unable to generate plot: {str(e)}</p>"


def create_enhanced_shap_plot(top_features, shap_values_single, feature_names, base_value=0.5, save_path=None):
    """
    Create an enhanced SHAP plot with waterfall-style visualization showing cumulative effect.
    
    Args:
        top_features: List of tuples (feature_name, shap_value)
        shap_values_single: Array of SHAP values
        feature_names: List of feature names
        base_value: Base prediction value (default: 0.5)
        save_path: Optional directory path to save the plot
    """
    try:
        # Extract feature names and values
        feat_names = [feat.split('__')[-1] if '__' in feat else feat for feat, _ in top_features]
        shap_vals = [val for _, val in top_features]
        
        # Sort by absolute value for better visualization
        sorted_data = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        feat_names = [x[0] for x in sorted_data]
        shap_vals = [x[1] for x in sorted_data]
        
        # Create figure with two subplots: bar plot and waterfall
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.3)
        
        # Subplot 1: Horizontal bar plot
        ax1 = fig.add_subplot(gs[0])
        colors = ['#e74c3c' if val > 0 else '#2ecc71' for val in shap_vals]
        y_pos = np.arange(len(feat_names))
        
        bars = ax1.barh(y_pos, shap_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feat_names, fontsize=10)
        ax1.set_xlabel('SHAP Value (Impact on Risk)', fontsize=11, fontweight='bold')
        ax1.set_title('Feature Contributions to PPD Risk Prediction', fontsize=13, fontweight='bold', pad=15)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(axis='x', alpha=0.3, linestyle=':')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shap_vals)):
            width = bar.get_width()
            label_x = width + (0.02 if width > 0 else -0.02)
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{val:+.3f}', ha='left' if width > 0 else 'right', 
                    va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Subplot 2: Waterfall-style cumulative plot
        ax2 = fig.add_subplot(gs[1])
        
        # Calculate cumulative values
        cumulative = [base_value]
        for val in shap_vals:
            cumulative.append(cumulative[-1] + val)
        
        # Create waterfall bars
        x_pos = np.arange(len(feat_names) + 1)
        waterfall_colors = ['#3498db'] + colors  # Base value in blue
        
        for i in range(len(feat_names)):
            # Bar showing the contribution
            bar_height = shap_vals[i]
            bar_bottom = cumulative[i]
            color = '#e74c3c' if bar_height > 0 else '#2ecc71'
            
            ax2.bar(i + 1, bar_height, bottom=bar_bottom, color=color, 
                   alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add value label
            label_y = bar_bottom + bar_height/2
            ax2.text(i + 1, label_y, f'{bar_height:+.3f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white' if abs(bar_height) > 0.05 else 'black')
        
        # Show base value
        ax2.bar(0, base_value, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
        ax2.text(0, base_value/2, f'Base\n{base_value:.2f}', 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Show final prediction
        final_pred = cumulative[-1]
        ax2.bar(len(feat_names) + 1, final_pred, color='#9b59b6', alpha=0.7, 
               edgecolor='black', linewidth=2)
        ax2.text(len(feat_names) + 1, final_pred/2, f'Final\n{final_pred:.2f}', 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Set labels
        ax2.set_xticks(range(len(feat_names) + 2))
        ax2.set_xticklabels(['Base'] + feat_names + ['Final'], rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Cumulative Prediction Value', fontsize=11, fontweight='bold')
        ax2.set_title('Waterfall Plot: How Features Build Up to Final Prediction', 
                     fontsize=13, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle=':')
        ax2.set_ylim([0, max(cumulative) * 1.1])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.7, label='Increases Risk'),
            Patch(facecolor='#2ecc71', alpha=0.7, label='Decreases Risk'),
            Patch(facecolor='#3498db', alpha=0.7, label='Base Value'),
            Patch(facecolor='#9b59b6', alpha=0.7, label='Final Prediction')
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # Use subplots_adjust() instead of tight_layout() for GridSpec compatibility
        # GridSpec axes are not fully compatible with tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
        
        # Save to file if path provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, "shap_enhanced_waterfall.png")
            plt.savefig(filepath, format='png', dpi=120, bbox_inches='tight')
            print(f"   💾 Saved: {filepath}")
        
        # Convert to base64 HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'
    except Exception as e:
        import traceback
        print(f"Error creating enhanced SHAP plot: {e}")
        print(traceback.format_exc())
        # Fallback to simple plot
        return create_shap_summary_plot(top_features, shap_values_single, feature_names)


def create_shap_summary_plot_class1(pipeline, X_test, max_display=15, return_image=True, title=None, save_path=None):
    """
    Create a SHAP summary plot for class 1 (Yes Depression) using test data.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features (DataFrame)
        max_display: Maximum number of features to display
        return_image: If True, return base64 HTML image, else show plot
        title: Optional custom title for the plot
        save_path: Optional directory path to save the plot
        
    Returns:
        HTML string with embedded image or None
    """
    try:
        # Get the preprocessed features
        preprocessor = pipeline.named_steps['preprocess']
        # Use a sample of test data for SHAP calculation (faster)
        X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42) if len(X_test) > 100 else X_test
        X_test_processed = preprocessor.transform(X_test_sample)
        
        # Convert to numpy array if it's a sparse matrix (from OneHotEncoder)
        if hasattr(X_test_processed, 'toarray'):
            X_test_processed = X_test_processed.toarray()
        X_test_processed = np.array(X_test_processed)
        
        # Get feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()
        
        # Get the model
        model = pipeline.named_steps['model']
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        # For Random Forest, ensure data is in the right format
        shap_values_output = explainer.shap_values(X_test_processed)
        
        # Debug: Print information about the model and SHAP output
        model_type = type(model).__name__
        print(f"DEBUG SHAP: Model type: {model_type}")
        print(f"DEBUG SHAP: SHAP output type: {type(shap_values_output)}")
        if isinstance(shap_values_output, list):
            print(f"DEBUG SHAP: SHAP output list length: {len(shap_values_output)}")
            for i, item in enumerate(shap_values_output):
                if hasattr(item, 'shape'):
                    print(f"DEBUG SHAP: SHAP output[{i}] shape: {item.shape}")
        elif hasattr(shap_values_output, 'shape'):
            print(f"DEBUG SHAP: SHAP output shape: {shap_values_output.shape}")
        
        # Handle different return types for both XGBoost and Random Forest
        # Random Forest often returns a list with [class_0, class_1] for binary classification
        # XGBoost might return a single array or a list
        if isinstance(shap_values_output, list):
            if len(shap_values_output) == 2:
                # List format with both classes: [shap_values_class_0, shap_values_class_1]
                # For binary classification, use class 1 (positive class)
                shap_values_class_1 = np.array(shap_values_output[1])
            elif len(shap_values_output) == 1:
                # Single element list - use it as class 1
                shap_values_class_1 = np.array(shap_values_output[0])
            else:
                # Multiple classes - use the last one (typically class 1 for binary)
                shap_values_class_1 = np.array(shap_values_output[-1])
        else:
            # Single array format: typically for positive class (class 1)
            shap_values_class_1 = np.array(shap_values_output)
        
        # Ensure shap_values_class_1 is 2D: [n_samples, n_features]
        if len(shap_values_class_1.shape) == 1:
            # If 1D, reshape to 2D (assuming it's for a single sample)
            shap_values_class_1 = shap_values_class_1.reshape(1, -1)
        elif len(shap_values_class_1.shape) == 3:
            # If 3D, take the last dimension (for class 1)
            shap_values_class_1 = shap_values_class_1[:, :, -1] if shap_values_class_1.shape[2] > 1 else shap_values_class_1[:, :, 0]
        
        # Ensure X_test_processed and shap_values_class_1 have matching shapes
        if X_test_processed.shape[0] != shap_values_class_1.shape[0]:
            # Adjust to match the number of samples
            min_samples = min(X_test_processed.shape[0], shap_values_class_1.shape[0])
            X_test_processed = X_test_processed[:min_samples]
            shap_values_class_1 = shap_values_class_1[:min_samples]
        
        # Determine number of features
        n_features = shap_values_class_1.shape[1]
        n_samples = shap_values_class_1.shape[0]
        
        # Ensure feature names match the number of features
        if len(feature_names) > n_features:
            feature_names_to_use = feature_names[:n_features]
        elif len(feature_names) < n_features:
            # If feature names are fewer, create generic names
            feature_names_to_use = [f"Feature_{i}" for i in range(n_features)]
        else:
            feature_names_to_use = feature_names
        
        # Clean feature names: remove "cat__" prefix and other preprocessing prefixes
        # Remove "cat__" prefix (can appear at start or after "__")
        feature_names_to_use = [name.replace('cat__', '').replace('num__', '') if 'cat__' in name or 'num__' in name else name for name in feature_names_to_use]
        # Also clean up any remaining "__" separators that might be left
        feature_names_to_use = [name.split('__')[-1] if '__' in name else name for name in feature_names_to_use]
        
        # Ensure X_test_processed has the same number of features
        if X_test_processed.shape[1] != n_features:
            # Adjust to match the number of features
            min_features = min(X_test_processed.shape[1], n_features)
            X_test_processed = X_test_processed[:, :min_features]
            shap_values_class_1 = shap_values_class_1[:, :min_features]
            feature_names_to_use = feature_names_to_use[:min_features]
        
        # Debug: Print final shapes
        print(f"DEBUG SHAP: Final shapes - X_test: {X_test_processed.shape}, SHAP: {shap_values_class_1.shape}")
        print(f"DEBUG SHAP: Number of features: {len(feature_names_to_use)}")
        
        # Create SHAP summary plot for Class 1 (Yes Depression)
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(shap_values_class_1, X_test_processed, 
                             feature_names=feature_names_to_use,
                             max_display=max_display,
                             show=False)
        except Exception as plot_error:
            # If summary_plot fails, try without feature names
            print(f"DEBUG SHAP: Error with feature names, trying without: {plot_error}")
            try:
                shap.summary_plot(shap_values_class_1, X_test_processed, 
                                 max_display=max_display,
                                 show=False)
            except Exception as plot_error2:
                # If that also fails, use a bar plot instead
                print(f"DEBUG SHAP: Error with summary_plot, using bar plot: {plot_error2}")
                shap.summary_plot(shap_values_class_1, X_test_processed, 
                                 plot_type="bar",
                                 max_display=max_display,
                                 show=False)
        plot_title = title if title is not None else "SHAP Summary Plot - Class 1 (Yes Depression)"
        plt.title(plot_title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if return_image:
            # Convert to base64 HTML
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'
        else:
            try:
                plt.show()
            except Exception:
                pass
            plt.close()
            return None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p>Error generating SHAP summary plot: {str(e)}</p>"


def create_gradio_interface(pipeline, X_train_sample, cat_cols, df=None, X_test=None, 
                           y_test=None, y_pred=None, y_proba=None, roc_auc=None, target=None,
                           X_train=None, y_train=None, ppd_agent=None):
    """
    Create a Gradio interface for postpartum depression prediction.

    Args:
        pipeline: Trained sklearn pipeline
        X_train_sample: Sample of training data for SHAP explainer (DataFrame with feature columns)
        cat_cols: List of categorical column names
        df: Full dataset (for visualizations)
        X_test: Test features (for visualizations)
        y_test: Test labels (for visualizations)
        y_pred: Predicted labels (for visualizations)
        y_proba: Predicted probabilities (for visualizations)
        roc_auc: ROC AUC score (for visualizations)
        target: Target variable name (for visualizations)
        X_train: Full training features (for model training, optional)
        y_train: Full training labels (for model training, optional)
        ppd_agent: PPD Agent instance (optional, if provided will use agent.predict() method)

    Returns:
        Gradio Interface object
    """
    # Get feature column names in the correct order
    feature_columns = list(X_train_sample.columns)

    # Get feature dtypes to ensure exact match
    feature_dtypes = X_train_sample.dtypes.to_dict()

    # Use mutable containers to store current pipeline and explainer
    current_pipeline = [pipeline]  # Use list to allow mutation
    current_explainer = [None]
    current_X_train = [X_train_sample]
    current_agent = [ppd_agent]  # Store agent if provided
    
    # Store current predictions for visualizations (will be updated when model is trained)
    current_y_pred = [y_pred if y_pred is not None else None]
    current_y_proba = [y_proba if y_proba is not None else None]
    current_roc_auc = [roc_auc if roc_auc is not None else None]

    # Create SHAP explainer
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        _ = preprocessor.transform(X_train_sample[:100])  # Use subset for speed
        model = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        current_explainer[0] = explainer
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer: {e}")
        current_explainer[0] = None

    # Training function
    def train_model_wrapper(model_algorithm, use_optimization):
        """Train model based on selected algorithm."""
        if X_train is None or y_train is None:
            return "❌ Training data not available. Please provide X_train and y_train when creating the interface."
        
        try:
            # Split data if needed
            # Always use a consistent split for fair comparison between models
            if X_test is None or y_test is None:
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
                )
            else:
                # Use provided test data, but still split training data for consistency
                # This ensures we're always evaluating on the same test set
                X_train_split = X_train
                X_test_split = X_test
                y_train_split = y_train
                y_test_split = y_test
            
            # Initialize optimization_info and algorithm_name
            optimization_info = ""
            algorithm_name = "XGBoost" if model_algorithm == "Training XGBoost Model" else "Random Forest"
            
            # Use PPD Agent tool for training instead of direct algorithm calls
            if model_algorithm == "Training XGBoost Model":
                # Use agent's train_xgboost method
                if current_agent[0] is None:
                    # Create a temporary agent if none exists
                    from ppd_agent import create_agent_from_training
                    # Create a dummy agent first (will be updated by training)
                    temp_pipeline = create_XGBoost_pipeline(cat_cols)
                    temp_agent = create_agent_from_training(temp_pipeline, X_train_split, cat_cols, list(X_train_split.columns))
                    current_agent[0] = temp_agent
                
                print("Using PPD Agent tool to train XGBoost model...")
                training_result = current_agent[0].train_xgboost(
                    X_train=X_train_split,
                    y_train=y_train_split,
                    X_test=X_test_split,
                    y_test=y_test_split,
                    use_optimization=use_optimization,
                    n_iter=30,  # Reduced for faster training in UI
                    cv=3,       # Reduced for faster training in UI
                    scoring='roc_auc',
                    random_state=42,
                    n_jobs=-1
                )
                
                if not training_result["success"]:
                    error_msg = f"❌ Agent training failed: {training_result.get('message', 'Unknown error')}"
                    if df is not None and X_test is not None and y_test is not None:
                        return (error_msg, "", "", "", "")
                    else:
                        return error_msg
                
                # Extract results from agent training
                new_pipeline = training_result["pipeline"]
                y_proba_new = training_result["y_proba"]
                y_pred_new = training_result["y_pred"]
                roc_auc_new = training_result["roc_auc"]
                best_params = training_result.get("parameters", {})
                
                # Format optimization info
                if use_optimization and best_params:
                    optimization_info = f"\n\n🔍 Hyperparameter Optimization Applied:\n"
                    for param, value in best_params.items():
                        param_name = param.replace('model__', '')
                        optimization_info += f"   • {param_name}: {value}\n"
                else:
                    optimization_info = "\n\nℹ️ Using default hyperparameters. Enable optimization for better performance."
            elif model_algorithm == "Training Random Forest Model":
                # Use agent's train_random_forest method
                if current_agent[0] is None:
                    # Create a temporary agent if none exists
                    from ppd_agent import create_agent_from_training
                    from MLmodel import create_rf_pipeline
                    # Create a dummy agent first (will be updated by training)
                    temp_pipeline = create_rf_pipeline(cat_cols)
                    temp_agent = create_agent_from_training(temp_pipeline, X_train_split, cat_cols, list(X_train_split.columns))
                    current_agent[0] = temp_agent
                
                print("Using PPD Agent tool to train Random Forest model...")
                training_result = current_agent[0].train_random_forest(
                    X_train=X_train_split,
                    y_train=y_train_split,
                    X_test=X_test_split,
                    y_test=y_test_split,
                    random_state=42,
                    n_jobs=-1
                )
                
                if not training_result["success"]:
                    error_msg = f"❌ Agent training failed: {training_result.get('message', 'Unknown error')}"
                    if df is not None and X_test is not None and y_test is not None:
                        return (error_msg, "", "", "", "")
                    else:
                        return error_msg
                
                # Extract results from agent training (all returned by agent)
                new_pipeline = training_result["pipeline"]
                roc_auc_new = training_result["roc_auc"]
                best_params = training_result.get("parameters", {})
                y_proba_new = training_result["y_proba"]
                y_pred_new = training_result["y_pred"]
                
                # Format optimization info (Random Forest training doesn't support optimization in agent yet)
                if use_optimization:
                    optimization_info = "\n\n⚠️ Hyperparameter optimization for Random Forest is not yet available through the agent tool. Using default parameters."
                else:
                    optimization_info = "\n\nℹ️ Using default hyperparameters."
            else:
                return f"❌ Unknown algorithm: {model_algorithm}"
            
            # Note: Training and evaluation is now done by the agent tool above
            # y_proba_new, y_pred_new, roc_auc_new are already set from agent training results
            
            # Debug: Print prediction statistics
            print(f"DEBUG: {algorithm_name} predictions - Unique values: {np.unique(y_pred_new, return_counts=True)}")
            print(f"DEBUG: {algorithm_name} predictions - Sum: {np.sum(y_pred_new)}, Total: {len(y_pred_new)}")
            
            # Update current pipeline and predictions
            current_pipeline[0] = new_pipeline
            current_X_train[0] = X_train_split
            current_y_pred[0] = y_pred_new
            current_y_proba[0] = y_proba_new
            current_roc_auc[0] = roc_auc_new
            
            # Reinitialize SHAP explainer
            try:
                model = new_pipeline.named_steps["model"]
                # For Random Forest, TreeExplainer works well
                # For better accuracy, we could pass background data, but TreeExplainer works without it
                current_explainer[0] = shap.TreeExplainer(model)
                explainer_status = "✅ SHAP explainer initialized (supports Top 5 Feature Contributions and Personalized Explanation)"
            except Exception as e:
                current_explainer[0] = None
                explainer_status = f"⚠️ SHAP explainer failed: {str(e)}"
            
            # Update agent with new pipeline if agent exists
            if current_agent[0] is not None:
                try:
                    from ppd_agent import create_agent_from_training
                    current_agent[0] = create_agent_from_training(
                        new_pipeline, X_train_split, cat_cols, list(X_train_split.columns)
                    )
                    explainer_status += "\n✅ PPD Agent updated with new model"
                except Exception as e:
                    print(f"Warning: Could not update agent: {e}")
            
            # Update visualizations if test data is available
            confusion_html = ""
            roc_html = ""
            prediction_html = ""
            
            # Initialize visualization HTML variables
            confusion_html = ""
            roc_html = ""
            prediction_html = ""
            shap_html = ""
            
            # Update visualizations using the test data and new predictions
            # Always use y_test_split and y_pred_new to ensure we're showing the current model's performance
            if y_test_split is not None and y_pred_new is not None:
                try:
                    from visualization import (
                        plot_confusion_matrix, plot_roc_curve,
                        plot_prediction_distribution
                    )
                    # Use the test data and predictions from the newly trained model
                    # Debug: Print to verify we're using new predictions
                    print(f"DEBUG: Creating confusion matrix - Model: {algorithm_name}")
                    print(f"DEBUG: y_test_split shape: {len(y_test_split)}, y_pred_new shape: {len(y_pred_new)}")
                    print(f"DEBUG: y_pred_new unique values: {np.unique(y_pred_new)}")
                    print(f"DEBUG: y_test_split unique values: {np.unique(y_test_split)}")
                    print(f"DEBUG: Confusion matrix will show predictions from {algorithm_name} model")
                    
                    # Create confusion matrix with model name in title
                    # Calculate confusion matrix values for debugging
                    from sklearn.metrics import confusion_matrix as cm_func
                    cm_values = cm_func(y_test_split, y_pred_new)
                    print(f"DEBUG: {algorithm_name} Confusion Matrix values:\n{cm_values}")
                    print(f"DEBUG: True Positives: {cm_values[1,1]}, True Negatives: {cm_values[0,0]}")
                    print(f"DEBUG: False Positives: {cm_values[0,1]}, False Negatives: {cm_values[1,0]}")
                    
                    # Create save path for plots
                    save_path = os.path.join("output", "plots", algorithm_name.replace(" ", "_"))
                    os.makedirs(save_path, exist_ok=True)
                    
                    confusion_html = plot_confusion_matrix(
                        y_test_split, y_pred_new, 
                        title=f"Confusion Matrix - {algorithm_name}",
                        return_image=True,
                        save_path=save_path
                    )
                    roc_html = plot_roc_curve(
                        y_test_split, y_proba_new, roc_auc_new,
                        title=f"ROC Curve - {algorithm_name}",
                        return_image=True,
                        save_path=save_path
                    )
                    prediction_html = plot_prediction_distribution(
                        y_proba_new, 
                        title=f"Prediction Probability Distribution - {algorithm_name}",
                        return_image=True,
                        save_path=save_path
                    )
                    # Update SHAP summary plot for Class 1 with the newly trained model
                    shap_html = create_shap_summary_plot_class1(
                        new_pipeline, X_test_split, max_display=15, return_image=True,
                        title=f"SHAP Summary Plot - Class 1 (Yes Depression) - {algorithm_name}",
                        save_path=save_path
                    )
                except Exception as e:
                    import traceback
                    print(f"Warning: Could not update visualizations: {e}")
                    print(traceback.format_exc())
                    confusion_html = f"<p>Error updating confusion matrix: {str(e)}</p>"
                    roc_html = f"<p>Error updating ROC curve: {str(e)}</p>"
                    prediction_html = f"<p>Error updating prediction distribution: {str(e)}</p>"
                    shap_html = f"<p>Error updating SHAP summary plot: {str(e)}</p>"
            
            status_message = f"""✅ {algorithm_name} model trained successfully!

📊 Model Performance:
   • ROC AUC Score: {roc_auc_new:.4f}
   • Test Set Size: {len(X_test_split)} samples
   • Training Set Size: {len(X_train_split)} samples

{explainer_status}{optimization_info}

The model is now ready for predictions!"""
            
            # Return based on whether test data is available for visualizations
            if df is not None and X_test is not None and y_test is not None:
                return (status_message, confusion_html, roc_html, prediction_html, shap_html)
            else:
                return status_message
            
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            if df is not None and X_test is not None and y_test is not None:
                return (error_msg, "", "", "", "")
            else:
                return error_msg

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
        # Use agent if available (Standalone Python usage - Example 1)
        if current_agent[0] is not None:
            try:
                # Use agent's predict method (Standalone Python usage pattern)
                result = current_agent[0].predict(
                    age=age,
                    feeling_sad=feeling_sad,
                    irritable=irritable,
                    trouble_sleeping=trouble_sleeping,
                    concentration=concentration,
                    appetite=appetite,
                    feeling_anxious=feeling_anxious,
                    guilt=guilt,
                    bonding=bonding,
                    suicide_attempt=suicide_attempt
                )
                
                # Format output to match Gradio interface expectations
                risk_score = f"PPD Risk Score: {result['risk_percentage']:.2f}%"
                
                # Format feature importance with detailed SHAP explanation
                feat_imp_lines = []
                shap_explanation_lines = []
                
                for i, feature in enumerate(result['feature_importance'][:5], 1):
                    feat_name = feature['feature'].split('__')[-1] if '__' in feature['feature'] else feature['feature']
                    shap_val = feature['shap_value']
                    abs_shap = abs(shap_val)
                    impact = "high" if abs_shap > 0.1 else "moderate" if abs_shap > 0.05 else "low"
                    direction = "increases" if shap_val > 0 else "decreases"
                    
                    # Format for feature importance display
                    feat_imp_lines.append(
                        f"{i}. {feat_name}\n   Impact: {impact} ({direction} risk by {abs_shap:.3f})"
                    )
                    
                    # Detailed SHAP explanation
                    shap_explanation_lines.append(
                        f"**{feat_name}**: SHAP value = {shap_val:.4f}\n"
                        f"  • This feature {direction} the PPD risk prediction by {abs_shap:.4f}\n"
                        f"  • Impact level: {impact.upper()}\n"
                        f"  • {'Positive SHAP value means this feature pushes the prediction toward higher risk.' if shap_val > 0 else 'Negative SHAP value means this feature pushes the prediction toward lower risk.'}"
                    )
                
                feat_imp_str = "\n\n".join(feat_imp_lines) if feat_imp_lines else "Feature importance not available"
                
                # Create comprehensive SHAP explanation
                shap_explanation = f"""## SHAP (SHapley Additive exPlanations) Analysis

SHAP values explain how each feature contributes to the final prediction. 
- **Positive SHAP values** push the prediction toward higher PPD risk
- **Negative SHAP values** push the prediction toward lower PPD risk
- **Magnitude** indicates the strength of the contribution

### Feature Contributions:

{chr(10).join(shap_explanation_lines) if shap_explanation_lines else 'No SHAP values available'}

### How to Interpret:
- **High impact** (|SHAP| > 0.1): This feature significantly influences the prediction
- **Moderate impact** (0.05 < |SHAP| ≤ 0.1): This feature moderately influences the prediction
- **Low impact** (|SHAP| ≤ 0.05): This feature has a minor influence on the prediction

The sum of all SHAP values equals the difference between the model's prediction and the baseline (average prediction).
"""
                
                # Use agent's explanation
                personalized_explanation = result['explanation']
                
                # Create enhanced SHAP plot with waterfall-style visualization
                if result['feature_importance']:
                    top_features = result['feature_importance'][:5]
                    feat_names = [f['feature'].split('__')[-1] if '__' in f['feature'] else f['feature'] for f in top_features]
                    shap_vals = np.array([f['shap_value'] for f in top_features])
                    
                    # Create both bar plot and waterfall-style plot
                    # Use 0.5 as base value (average prediction) for waterfall visualization
                    # Determine save path based on current model
                    model_type = type(current_agent[0].pipeline.named_steps['model']).__name__
                    algo_name = "XGBoost" if "XGB" in model_type.upper() else "RandomForest"
                    save_path_plot = os.path.join("output", "plots", algo_name)
                    os.makedirs(save_path_plot, exist_ok=True)
                    
                    plot_html = create_enhanced_shap_plot(
                        list(zip(feat_names, shap_vals)),
                        shap_vals,
                        feat_names,
                        base_value=0.5,
                        save_path=save_path_plot
                    )
                else:
                    plot_html = ""
                
                return risk_score, feat_imp_str, personalized_explanation, shap_explanation, plot_html
                
            except Exception as e:
                import traceback
                error_msg = f"Agent prediction error: {str(e)}\n{traceback.format_exc()}"
                return (
                    "Error during prediction",
                    error_msg,
                    "Unable to generate explanation.",
                    "SHAP explanation unavailable due to error.",
                    ""
                )
        
        # Fallback to original predict_depression if agent not available
        if current_explainer[0] is None:
            return (
                "SHAP explainer not available",
                "Please train the model first",
                "Unable to generate explanation.",
                "SHAP explanation unavailable. Please train the model first.",
                ""
            )
        
        # Get results from original predict_depression
        risk_score, feat_imp_str, personalized_explanation, plot_html = predict_depression(
            current_pipeline[0],
            current_explainer[0],
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
        
        # Create SHAP explanation for non-agent path
        shap_explanation = f"""## SHAP (SHapley Additive exPlanations) Analysis

SHAP values explain how each feature contributes to the final prediction.
- **Positive SHAP values** push the prediction toward higher PPD risk
- **Negative SHAP values** push the prediction toward lower PPD risk
- **Magnitude** indicates the strength of the contribution

### Feature Contributions:
{feat_imp_str}

### How to Interpret:
- **High impact** (|SHAP| > 0.1): This feature significantly influences the prediction
- **Moderate impact** (0.05 < |SHAP| ≤ 0.1): This feature moderately influences the prediction
- **Low impact** (|SHAP| ≤ 0.05): This feature has a minor influence on the prediction

The sum of all SHAP values equals the difference between the model's prediction and the baseline (average prediction).
"""
        
        return risk_score, feat_imp_str, personalized_explanation, shap_explanation, plot_html

    # Create Gradio interface
    with gr.Blocks(title="Postpartum Depression Prediction Agent Tool") as interface:
        gr.Markdown("# 🏥 Postpartum Depression Risk Assessment Model Training Agent")
        
        # Model selection and training section
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Select Model Training Algorithm**")
                model_algorithm = gr.Dropdown(
                    label=None,
                    choices=["Training XGBoost Model", "Training Random Forest Model"],
                    value="Training XGBoost Model",
                    info="Choose the algorithm to train the model"
                )
                use_optimization = gr.Checkbox(
                    label="Use RandomizedSearchCV optimization (slower but better performance)",
                    value=False,
                    info="Enable hyperparameter optimization for the selected model. This will take longer but may improve model performance."
                )
                train_btn = gr.Button("🚀 Train Model", variant="secondary")
                training_status = gr.Textbox(
                    label="Training Status",
                    interactive=False,
                    lines=1,
                    value="Ready to train model. Select an algorithm and click 'Train Model'."
                )
        
        # Add visualization tabs
        if df is not None and X_test is not None and y_test is not None:
            with gr.Tabs() as viz_tabs:
                with gr.Tab("1. Target Distribution"):
                    target_dist_plot = gr.HTML(label="Target Distribution")
                
                with gr.Tab("2. Feature Distributions"):
                    feature_dist_plot = gr.HTML(label="Feature Distributions by Target")
                
                with gr.Tab("3. Confusion Matrix"):
                    confusion_matrix_plot = gr.HTML(label="Confusion Matrix")
                
                with gr.Tab("4. ROC Curve"):
                    roc_curve_plot = gr.HTML(label="ROC Curve")
                
                with gr.Tab("5. Prediction Distribution"):
                    prediction_dist_plot = gr.HTML(label="Prediction Probability Distribution")
                
                with gr.Tab("6. Correlation Heatmap"):
                    correlation_heatmap_plot = gr.HTML(label="Correlation Heatmap")
                
                with gr.Tab("7. SHAP Summary (Class 1 - Yes Depression)"):
                    shap_summary_class1_plot = gr.HTML(label="SHAP Summary Plot - Class 1 (Yes Depression)")
            
            # Load visualizations when interface loads
            def load_visualizations():
                from visualization import (
                    plot_target_distribution, plot_feature_distributions,
                    plot_confusion_matrix, plot_roc_curve,
                    plot_prediction_distribution, plot_correlation_heatmap
                )
                
                # Determine algorithm name from pipeline
                try:
                    model_type = type(pipeline.named_steps['model']).__name__
                    algorithm_name = "XGBoost" if "XGB" in model_type.upper() else "RandomForest"
                    save_path_viz = os.path.join("output", "plots", algorithm_name)
                    os.makedirs(save_path_viz, exist_ok=True)
                except:
                    save_path_viz = None
                
                plots = {}
                try:
                    plots['target'] = plot_target_distribution(df[target], return_image=True, save_path=save_path_viz)
                except Exception as e:
                    plots['target'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    plots['features'] = plot_feature_distributions(df, cat_cols, target, return_image=True, save_path=save_path_viz)
                except Exception as e:
                    plots['features'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    plots['confusion'] = plot_confusion_matrix(y_test, y_pred, return_image=True, save_path=save_path_viz)
                except Exception as e:
                    plots['confusion'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    plots['roc'] = plot_roc_curve(y_test, y_proba, roc_auc, return_image=True, save_path=save_path_viz)
                except Exception as e:
                    plots['roc'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    plots['prediction'] = plot_prediction_distribution(y_proba, return_image=True, save_path=save_path_viz)
                except Exception as e:
                    plots['prediction'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    plots['correlation'] = plot_correlation_heatmap(df, target, return_image=True, save_path=save_path_viz)
                except Exception as e:
                    plots['correlation'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    plots['shap_class1'] = create_shap_summary_plot_class1(
                        pipeline, X_test, max_display=15, return_image=True, save_path=save_path_viz
                    )
                except Exception as e:
                    plots['shap_class1'] = f"<p>Error: {str(e)}</p>"
                
                return (
                    plots['target'],
                    plots['features'],
                    plots['confusion'],
                    plots['roc'],
                    plots['prediction'],
                    plots['correlation'],
                    plots['shap_class1']
                )
            
            # Load visualizations on interface load
            interface.load(fn=load_visualizations, outputs=[
                target_dist_plot, feature_dist_plot, confusion_matrix_plot,
                roc_curve_plot, prediction_dist_plot, correlation_heatmap_plot,
                shap_summary_class1_plot
            ])
        
        gr.Markdown("---")
        gr.Markdown(
            "**Enter the following information to assess the risk of postpartum depression.**"
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
            shap_explanation = gr.Markdown(
                label="Detailed SHAP Explanation",
                value="SHAP explanation will appear here after prediction."
            )
        
        with gr.Row():
            shap_plot = gr.HTML(label="SHAP Visualization (Bar Plot & Waterfall)")

        # Connect training button
        # If visualizations are available, update them when training
        if df is not None and X_test is not None and y_test is not None:
            train_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, confusion_matrix_plot, roc_curve_plot, prediction_dist_plot, shap_summary_class1_plot],
            )
        else:
            # If no test data, only update training status
            train_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status],
            )
        
        # Connect prediction button
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
            outputs=[risk_output, feature_importance, personalized_explanation, shap_explanation, shap_plot],
        )

        gr.Markdown("### ⚠️ Disclaimer")
        gr.Markdown(
            "This tool is for informational purposes only and should not replace "
            "professional medical advice."
        )

    return interface