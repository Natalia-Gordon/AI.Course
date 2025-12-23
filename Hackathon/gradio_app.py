"""
Gradio interface for Postpartum Depression Prediction Agent Tool.

This module has been refactored to use separate modules for:
- gradio_predictions: Prediction functions with type hints
- gradio_visualizations: Visualization functions with type hints
- exceptions: Custom exception classes for better error handling
"""
from typing import Optional, Dict, Any, List, Tuple
import gradio as gr
import numpy as np
import pandas as pd
import os
import io
import base64
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import from refactored modules
from gradio_predictions import predict_depression, generate_personalized_explanation
from gradio_visualizations import (
    create_shap_summary_plot,
    create_enhanced_shap_plot,
    create_shap_summary_plot_class1,
    generate_detailed_shap_explanation
)
from gradio_helpers import (
    get_algorithm_name,
    clean_feature_name,
    calculate_impact_level,
    format_feature_importance_line,
    generate_shap_explanation_markdown,
    get_save_path_for_algorithm
)
from exceptions import (
    PredictionError,
    SHAPExplanationError,
    VisualizationError
)
from MLmodel import (
    create_XGBoost_pipeline,
    create_rf_pipeline,
    optimize_XGBoost_hyperparameters,
    optimize_rf_hyperparameters
)


# All prediction and visualization functions are imported from:
# - gradio_predictions: predict_depression, generate_personalized_explanation
# - gradio_visualizations: create_shap_summary_plot, create_enhanced_shap_plot, create_shap_summary_plot_class1
# - gradio_helpers: Helper functions for common operations

def create_gradio_interface(
    pipeline: Pipeline,
    X_train_sample: pd.DataFrame,
    cat_cols: List[str],
    df: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    roc_auc: Optional[float] = None,
    target: Optional[str] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    ppd_agent: Optional[Any] = None
) -> gr.Blocks:
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

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Get feature column names in the correct order
    feature_columns = list(X_train_sample.columns)

    # Get feature dtypes to ensure exact match
    feature_dtypes = X_train_sample.dtypes.to_dict()

    # Try to load existing trained agent if available (try both XGBoost and Random Forest)
    agent_loaded = False
    loaded_algorithm = None
    if ppd_agent is None:
        import os
        # Try to load XGBoost first, then Random Forest
        agent_paths = [
            "output/agents/ppd_agent_xgboost.pkl",
            "output/agents/ppd_agent_rf.pkl"
        ]
        
        for agent_path in agent_paths:
            if os.path.exists(agent_path):
                try:
                    from ppd_agent import PPDAgent
                    loaded_agent = PPDAgent.load(agent_path)
                    ppd_agent = loaded_agent
                    # Update pipeline from loaded agent
                    if hasattr(loaded_agent, 'pipeline') and loaded_agent.pipeline is not None:
                        pipeline = loaded_agent.pipeline
                        agent_loaded = True
                        # Determine algorithm type
                        model_type = type(pipeline.named_steps.get("model", None)).__name__
                        if "XGB" in model_type.upper():
                            loaded_algorithm = "XGBoost"
                        elif "RandomForest" in model_type or "Random" in model_type:
                            loaded_algorithm = "RandomForest"
                        print(f"✅ Loaded existing trained agent ({loaded_algorithm}) from {agent_path}")
                        break
                except Exception as e:
                    print(f"⚠️ Could not load agent from {agent_path}: {e}")
                    continue
    
    # Use mutable containers to store current pipeline and explainer
    current_pipeline = [pipeline]  # Use list to allow mutation
    current_explainer = [None]
    current_X_train = [X_train_sample]
    current_agent = [ppd_agent]  # Store agent if provided
    
    # Store current predictions for visualizations (will be updated when model is trained)
    current_y_pred = [y_pred if y_pred is not None else None]
    current_y_proba = [y_proba if y_proba is not None else None]
    current_roc_auc = [roc_auc if roc_auc is not None else None]

    # Create SHAP explainer (only if pipeline is trained or agent is loaded)
    model_trained = False
    try:
        # Check if model is trained by trying to get feature importances
        model = pipeline.named_steps.get("model")
        if model is not None and hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            preprocessor = pipeline.named_steps["preprocess"]
            _ = preprocessor.transform(X_train_sample[:100])  # Use subset for speed
            explainer = shap.TreeExplainer(model)
            current_explainer[0] = explainer
            model_trained = True
        else:
            # Model not trained yet, explainer will be created after training
            current_explainer[0] = None
            model_trained = False
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer: {e}")
        current_explainer[0] = None
        model_trained = False
    
    # If agent was loaded, mark model as trained
    if agent_loaded:
        model_trained = True

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
                    # Create a new agent instance without training (will train via train_xgboost)
                    from ppd_agent import PPDAgent
                    # Create an untrained pipeline for the agent
                    temp_pipeline = create_XGBoost_pipeline(cat_cols)
                    # Create agent with untrained pipeline (SHAP explainer will be None initially)
                    temp_agent = PPDAgent(temp_pipeline, X_train_split, cat_cols, list(X_train_split.columns))
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
                        return (error_msg, "", "", "", "", gr.update(interactive=False), gr.update(visible=agent_loaded))
                    else:
                        return (error_msg, gr.update(interactive=False), gr.update(visible=agent_loaded))
                
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
                    # Create a new agent instance without training (will train via train_random_forest)
                    from ppd_agent import PPDAgent
                    from MLmodel import create_rf_pipeline
                    # Create an untrained pipeline for the agent
                    temp_pipeline = create_rf_pipeline(cat_cols)
                    # Create agent with untrained pipeline (SHAP explainer will be None initially)
                    temp_agent = PPDAgent(temp_pipeline, X_train_split, cat_cols, list(X_train_split.columns))
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
                        return (error_msg, "", "", "", "", gr.update(interactive=False), gr.update(visible=agent_loaded))
                    else:
                        return (error_msg, gr.update(interactive=False), gr.update(visible=agent_loaded))
                
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
                    updated_agent = create_agent_from_training(
                        new_pipeline, X_train_split, cat_cols, list(X_train_split.columns)
                    )
                    current_agent[0] = updated_agent
                    explainer_status += "\n✅ PPD Agent updated with new model"
                    
                    # Save the updated agent with algorithm-specific filename
                    import os
                    os.makedirs("output/agents", exist_ok=True)
                    # Determine algorithm type from pipeline
                    model_type = type(new_pipeline.named_steps.get("model", None)).__name__
                    if "XGB" in model_type.upper():
                        algo_suffix = "xgboost"
                    elif "RandomForest" in model_type or "Random" in model_type:
                        algo_suffix = "rf"
                    else:
                        algo_suffix = "unknown"
                    
                    agent_path = f"output/agents/ppd_agent_{algo_suffix}.pkl"
                    updated_agent.save(agent_path)
                    print(f"✅ Agent saved to {agent_path} (Algorithm: {algorithm_name})")
                except Exception as e:
                    print(f"Warning: Could not update/save agent: {e}")
            
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
            # Also enable the "Assess Risk" button and show "Retrain Model" button after successful training
            if df is not None and X_test is not None and y_test is not None:
                return (status_message, confusion_html, roc_html, prediction_html, shap_html, gr.update(interactive=True), gr.update(visible=True))
            else:
                return (status_message, gr.update(interactive=True), gr.update(visible=True))
            
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            if df is not None and X_test is not None and y_test is not None:
                return (error_msg, "", "", "", "", gr.update(interactive=False), gr.update(visible=agent_loaded))
            else:
                return (error_msg, gr.update(interactive=False), gr.update(visible=agent_loaded))

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
        # Check if model has been trained
        if current_agent[0] is None and (current_pipeline[0] is None or 
            not hasattr(current_pipeline[0].named_steps.get("model", None), 'feature_importances_')):
            return (
                "⚠️ Model not trained yet. Please click 'Start Train Model' to train the model first.",
                "Model not trained yet.",
                "Model not trained yet.",
                "Model not trained yet.",
                "Model not trained yet."
            )
        
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
                    shap_val = feature['shap_value']
                    
                    # Format for feature importance display using helper function
                    feat_imp_lines.append(
                        format_feature_importance_line(feature['feature'], shap_val, i)
                    )
                
                feat_imp_str = "\n\n".join(feat_imp_lines) if feat_imp_lines else "Feature importance not available"
                
                # Create comprehensive SHAP explanation using helper function
                shap_explanation = generate_shap_explanation_markdown(
                    result['feature_importance'],
                    risk_percentage=result.get('risk_percentage')
                )
                
                # Use agent's explanation
                personalized_explanation = result['explanation']
                
                # Create enhanced SHAP plot with waterfall-style visualization
                if result['feature_importance']:
                    top_features = result['feature_importance'][:5]
                    feat_names = [clean_feature_name(f['feature']) for f in top_features]
                    shap_vals = np.array([f['shap_value'] for f in top_features])
                    
                    # Create both bar plot and waterfall-style plot
                    # Use 0.5 as base value (average prediction) for waterfall visualization
                    # Determine save path based on current model
                    save_path_plot = get_save_path_for_algorithm(pipeline=current_agent[0].pipeline)
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
        
        # Get results from original predict_depression (returns 5 values)
        risk_score, feat_imp_str, personalized_explanation, shap_explanation, plot_html = predict_depression(
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
                train_btn = gr.Button("🚀 Start Train Model", variant="secondary")
                retrain_btn = gr.Button("🔄 Retrain Model", variant="secondary", visible=agent_loaded)
                training_status = gr.Textbox(
                    label="Training Status",
                    interactive=False,
                    lines=1,
                    value=f"✅ {loaded_algorithm} model loaded from saved agent. Ready for predictions!" if agent_loaded and loaded_algorithm else "Ready to train model. Select an algorithm and click 'Start Train Model'. ⚠️ Note: You must train the model before making predictions."
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
                
                # Use current pipeline (may be from loaded agent)
                viz_pipeline = current_pipeline[0] if current_pipeline[0] is not None else pipeline
                
                # Determine algorithm name from pipeline using helper function
                try:
                    save_path_viz = get_save_path_for_algorithm(pipeline=viz_pipeline)
                    if save_path_viz:
                        os.makedirs(save_path_viz, exist_ok=True)
                except:
                    save_path_viz = None
                
                # If y_pred or y_proba are None but agent is loaded, generate predictions
                viz_y_pred = current_y_pred[0] if current_y_pred[0] is not None else y_pred
                viz_y_proba = current_y_proba[0] if current_y_proba[0] is not None else y_proba
                viz_roc_auc = current_roc_auc[0] if current_roc_auc[0] is not None else roc_auc
                
                # If predictions are still None but we have a trained pipeline, generate them
                if (viz_y_pred is None or viz_y_proba is None) and viz_pipeline is not None:
                    try:
                        model = viz_pipeline.named_steps.get("model")
                        if model is not None and hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                            # Generate predictions from loaded agent
                            if X_test is not None and y_test is not None:
                                viz_y_proba = viz_pipeline.predict_proba(X_test)[:, 1]
                                viz_y_pred = viz_pipeline.predict(X_test)
                                from sklearn.metrics import roc_auc_score
                                viz_roc_auc = roc_auc_score(y_test, viz_y_proba)
                                # Update current predictions
                                current_y_pred[0] = viz_y_pred
                                current_y_proba[0] = viz_y_proba
                                current_roc_auc[0] = viz_roc_auc
                                print("✅ Generated predictions from loaded agent for visualizations")
                    except Exception as e:
                        print(f"⚠️ Could not generate predictions for visualizations: {e}")
                
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
                    if viz_y_pred is not None and y_test is not None:
                        plots['confusion'] = plot_confusion_matrix(y_test, viz_y_pred, return_image=True, save_path=save_path_viz)
                    else:
                        plots['confusion'] = "<p>Confusion matrix not available (model not trained or predictions not available)</p>"
                except Exception as e:
                    plots['confusion'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    if viz_y_proba is not None and y_test is not None and viz_roc_auc is not None:
                        plots['roc'] = plot_roc_curve(y_test, viz_y_proba, roc_auc=viz_roc_auc, return_image=True, save_path=save_path_viz)
                    else:
                        plots['roc'] = "<p>ROC curve not available (model not trained or predictions not available)</p>"
                except Exception as e:
                    plots['roc'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    if viz_y_proba is not None:
                        plots['prediction'] = plot_prediction_distribution(viz_y_proba, return_image=True, save_path=save_path_viz)
                    else:
                        plots['prediction'] = "<p>Prediction distribution not available (model not trained or predictions not available)</p>"
                except Exception as e:
                    plots['prediction'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    plots['correlation'] = plot_correlation_heatmap(df, target, return_image=True, save_path=save_path_viz)
                except Exception as e:
                    plots['correlation'] = f"<p>Error: {str(e)}</p>"
                
                try:
                    if viz_pipeline is not None and X_test is not None:
                        plots['shap_class1'] = create_shap_summary_plot_class1(
                            viz_pipeline, X_test, max_display=15, return_image=True, save_path=save_path_viz
                        )
                    else:
                        plots['shap_class1'] = "<p>SHAP summary plot not available (model not trained or test data not available)</p>"
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
        gr.Markdown(
            "⚠️ **Important:** You must train the model first by clicking 'Start Train Model' above before you can make predictions. "
            "The 'Assess Risk' button will be enabled automatically after successful training.",
            elem_classes=["warning-message"]
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

        predict_btn = gr.Button("🔍 Assess Risk", variant="primary", interactive=model_trained)
        gr.Markdown(
            "ℹ️ **Note:** This button is disabled until the model is trained. Please train the model first using the 'Start Train Model' button above.",
            visible=True
        )

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
                outputs=[training_status, confusion_matrix_plot, roc_curve_plot, prediction_dist_plot, shap_summary_class1_plot, predict_btn, retrain_btn],
            )
            # Retrain button uses the same function
            retrain_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, confusion_matrix_plot, roc_curve_plot, prediction_dist_plot, shap_summary_class1_plot, predict_btn, retrain_btn],
            )
        else:
            # If no test data, only update training status
            train_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, predict_btn, retrain_btn],
            )
            # Retrain button uses the same function
            retrain_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, predict_btn, retrain_btn],
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