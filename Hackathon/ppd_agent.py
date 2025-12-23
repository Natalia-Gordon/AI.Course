"""
Postpartum Depression (PPD) Prediction Agent Tool

This module provides an agent interface for the PPD prediction system,
making it easy to integrate with agent frameworks, APIs, and other tools.
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from MLmodel import create_rf_pipeline, create_XGBoost_pipeline, train_and_evaluate, optimize_XGBoost_hyperparameters


class PPDAgent:
    """
    Agent class for Postpartum Depression risk prediction.
    
    This agent can be used as a standalone tool or integrated with
    agent frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self, pipeline, X_train, cat_cols, feature_columns=None):
        """
        Initialize the PPD Agent.
        
        Args:
            pipeline: Trained sklearn pipeline
            X_train: Training data (for SHAP explainer)
            cat_cols: List of categorical column names
            feature_columns: Optional list of feature column names in order
        """
        self.pipeline = pipeline
        self.X_train = X_train
        self.cat_cols = cat_cols
        self.feature_columns = feature_columns if feature_columns is not None else list(X_train.columns)
        
        # Filter out target column if present
        self.feature_columns = [col for col in self.feature_columns if col != "PPD_Composite"]
        
        # Get feature dtypes from training data
        self.feature_dtypes = {col: str(X_train[col].dtype) for col in self.feature_columns}
        
        # Initialize SHAP explainer (only if model is trained)
        try:
            model = self.pipeline.named_steps.get("model")
            if model is not None and hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                print("Initializing SHAP explainer...")
                self.explainer = shap.TreeExplainer(model)
                print("SHAP explainer ready!")
            else:
                # Model not trained yet, explainer will be created after training
                self.explainer = None
                print("Model not trained yet. SHAP explainer will be created after training.")
        except Exception as e:
            # Model not trained or explainer creation failed
            self.explainer = None
            print(f"SHAP explainer not initialized (model may not be trained yet): {e}")
    
    def predict(self, 
                age: str,
                feeling_sad: str,
                irritable: str,
                trouble_sleeping: str,
                concentration: str,
                appetite: str,
                feeling_anxious: str,
                guilt: str,
                bonding: str,
                suicide_attempt: str) -> Dict[str, Any]:
        """
        Predict PPD risk based on input features.
        
        Args:
            age: Age group (e.g., '25-30', '30-35', etc.)
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
            Dictionary with prediction results including:
            - risk_score: PPD risk probability (0-1)
            - risk_percentage: PPD risk as percentage
            - risk_level: 'Low', 'Moderate', 'High', or 'Very High'
            - prediction: Binary prediction (0 or 1)
            - feature_importance: Top 5 feature contributions
            - explanation: Personalized explanation
        """
        # Normalize input values
        def normalize_yes_no(val):
            if val is None:
                return "No"
            val_str = str(val).strip().lower()
            if val_str in ["yes", "y", "true", "1"]:
                return "Yes"
            elif val_str in ["no", "n", "false", "0"]:
                return "No"
            return str(val)  # Keep original if not Yes/No
        
        # Create input row
        row_dict = {
            "Age": str(age) if age else "",
            "Feeling sad or Tearful": normalize_yes_no(feeling_sad),
            "Irritable towards baby & partner": normalize_yes_no(irritable),
            "Trouble sleeping at night": normalize_yes_no(trouble_sleeping),
            "Problems concentrating or making decision": normalize_yes_no(concentration),
            "Overeating or loss of appetite": normalize_yes_no(appetite),
            "Feeling anxious": normalize_yes_no(feeling_anxious),
            "Feeling of guilt": normalize_yes_no(guilt),
            "Problems of bonding with baby": normalize_yes_no(bonding),
            "Suicide attempt": normalize_yes_no(suicide_attempt)
        }
        
        # Create DataFrame with correct column order
        row = pd.DataFrame([row_dict], columns=self.feature_columns)
        
        # Convert dtypes to match training data
        for col in row.columns:
            if col in self.feature_dtypes:
                if self.feature_dtypes[col] == "object":
                    row[col] = row[col].fillna("").astype(str)
                    row[col] = row[col].replace("nan", "", regex=False)
        
        # Get prediction
        proba_result = self.pipeline.predict_proba(row)
        prob_class_0 = float(proba_result[0][0])
        prob_class_1 = float(proba_result[0][1])
        
        # Use prob_class_1 as risk score
        risk_score = prob_class_1
        risk_percentage = risk_score * 100
        prediction = int(prob_class_1 > 0.5)
        
        # Determine risk level
        if risk_percentage < 25:
            risk_level = "Low"
        elif risk_percentage < 50:
            risk_level = "Moderate"
        elif risk_percentage < 75:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Get SHAP values
        try:
            # Preprocess the row
            preprocessor = self.pipeline.named_steps["preprocess"]
            row_processed = preprocessor.transform(row)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(row_processed)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Get feature names after preprocessing
            feature_names = preprocessor.get_feature_names_out(self.feature_columns)
            
            # Get top 5 features
            shap_abs = np.abs(shap_values[0])
            top_indices = np.argsort(shap_abs)[-5:][::-1]
            
            feature_importance = []
            for idx in top_indices:
                feature_name = feature_names[idx]
                shap_value = float(shap_values[0][idx])
                impact = "increases" if shap_value > 0 else "decreases"
                feature_importance.append({
                    "feature": feature_name,
                    "shap_value": shap_value,
                    "impact": impact,
                    "abs_contribution": float(shap_abs[idx])
                })
            
            # Generate personalized explanation
            explanation = self._generate_explanation(risk_level, risk_percentage, feature_importance)
            
        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            feature_importance = []
            explanation = f"Risk assessment: {risk_level} risk ({risk_percentage:.2f}%)"
        
        return {
            "risk_score": risk_score,
            "risk_percentage": round(risk_percentage, 2),
            "risk_level": risk_level,
            "prediction": prediction,
            "feature_importance": feature_importance,
            "explanation": explanation,
            "probabilities": {
                "no_depression": round(prob_class_0 * 100, 2),
                "depression": round(prob_class_1 * 100, 2)
            }
        }
    
    def _generate_explanation(self, risk_level: str, risk_percentage: float, 
                             feature_importance: List[Dict]) -> str:
        """Generate a personalized explanation based on risk and features."""
        explanation = f"The model identifies a {risk_level.lower()} risk ({risk_percentage:.2f}%)"
        
        if feature_importance:
            top_features = feature_importance[:3]
            feature_names = [f["feature"].split("_")[-1] if "_" in f["feature"] else f["feature"] 
                           for f in top_features]
            impacts = [f["impact"] for f in top_features]
            
            if len(top_features) > 0:
                explanation += ", mainly due to "
                if len(top_features) == 1:
                    explanation += f"{feature_names[0]} which {impacts[0]} the risk"
                elif len(top_features) == 2:
                    explanation += f"the combination between {feature_names[0]} and {feature_names[1]}"
                else:
                    explanation += f"the combination between {', '.join(feature_names[:-1])}, and {feature_names[-1]}"
        
        explanation += "."
        return explanation
    
    def predict_from_dict(self, input_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Predict from a dictionary of inputs.
        
        Args:
            input_dict: Dictionary with feature names as keys
        
        Returns:
            Prediction results dictionary
        """
        return self.predict(
            age=input_dict.get("Age", ""),
            feeling_sad=input_dict.get("Feeling sad or Tearful", "No"),
            irritable=input_dict.get("Irritable towards baby & partner", "No"),
            trouble_sleeping=input_dict.get("Trouble sleeping at night", "No"),
            concentration=input_dict.get("Problems concentrating or making decision", "No"),
            appetite=input_dict.get("Overeating or loss of appetite", "No"),
            feeling_anxious=input_dict.get("Feeling anxious", "No"),
            guilt=input_dict.get("Feeling of guilt", "No"),
            bonding=input_dict.get("Problems of bonding with baby", "No"),
            suicide_attempt=input_dict.get("Suicide attempt", "No")
        )
    
    def batch_predict(self, input_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Predict for multiple inputs at once.
        
        Args:
            input_list: List of dictionaries with feature names as keys
        
        Returns:
            List of prediction results
        """
        return [self.predict_from_dict(input_dict) for input_dict in input_list]
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI Function Calling schema for prediction tool.
        
        Returns:
            Dictionary with function schema for predict_ppd_risk
        """
        return {
            "name": "predict_ppd_risk",
            "description": "Predicts postpartum depression (PPD) risk based on patient symptoms and demographics. Returns risk score, level, and explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "string",
                        "description": "Age group (e.g., '25-30', '30-35', '35-40', etc.)"
                    },
                    "feeling_sad": {
                        "type": "string",
                        "enum": ["Yes", "No", "Sometimes"],
                        "description": "Feeling sad or tearful"
                    },
                    "irritable": {
                        "type": "string",
                        "enum": ["Yes", "No", "Sometimes"],
                        "description": "Irritable towards baby & partner"
                    },
                    "trouble_sleeping": {
                        "type": "string",
                        "enum": ["Yes", "No", "Two or more days a week"],
                        "description": "Trouble sleeping at night"
                    },
                    "concentration": {
                        "type": "string",
                        "enum": ["Yes", "No", "Often"],
                        "description": "Problems concentrating or making decision"
                    },
                    "appetite": {
                        "type": "string",
                        "enum": ["Yes", "No", "Not at all"],
                        "description": "Overeating or loss of appetite"
                    },
                    "feeling_anxious": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": "Feeling anxious"
                    },
                    "guilt": {
                        "type": "string",
                        "enum": ["Yes", "No", "Maybe"],
                        "description": "Feeling of guilt"
                    },
                    "bonding": {
                        "type": "string",
                        "enum": ["Yes", "No", "Sometimes"],
                        "description": "Problems of bonding with baby"
                    },
                    "suicide_attempt": {
                        "type": "string",
                        "enum": ["Yes", "No", "Not interested to say"],
                        "description": "Suicide attempt"
                    }
                },
                "required": ["age", "feeling_sad", "irritable", "trouble_sleeping", 
                           "concentration", "appetite", "feeling_anxious", "guilt", 
                           "bonding", "suicide_attempt"]
            }
        }
    
    def get_training_tool_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI Function Calling schema for Random Forest training tool.
        
        Returns:
            Dictionary with function schema for train_random_forest
        """
        return {
            "name": "train_random_forest",
            "description": "Trains a Random Forest model for postpartum depression prediction. Requires training data (X_train, y_train) and optional test data. Updates the agent's pipeline with the trained model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_estimators": {
                        "type": "integer",
                        "description": "Number of trees in the forest (default: 100)",
                        "default": 100
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth of trees (default: 10)",
                        "default": 10
                    },
                    "min_samples_split": {
                        "type": "integer",
                        "description": "Minimum samples required to split a node (default: 2)",
                        "default": 2
                    },
                    "min_samples_leaf": {
                        "type": "integer",
                        "description": "Minimum samples required in a leaf node (default: 1)",
                        "default": 1
                    },
                    "max_features": {
                        "type": "string",
                        "enum": ["sqrt", "log2", "auto", "None"],
                        "description": "Number of features to consider for best split (default: 'sqrt')",
                        "default": "sqrt"
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Proportion of dataset to include in test split (default: 0.2)",
                        "default": 0.2,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (default: 42)",
                        "default": 42
                    }
                },
                "required": []
            }
        }
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tool schemas (both prediction and training).
        
        Returns:
            List of tool schema dictionaries
        """
        return [
            self.get_tool_schema(),
            self.get_training_tool_schema()
        ]
    
    def save(self, filepath: str):
        """Save the agent to a file."""
        agent_data = {
            "pipeline": self.pipeline,
            "X_train": self.X_train,
            "cat_cols": self.cat_cols,
            "feature_columns": self.feature_columns,
            "feature_dtypes": self.feature_dtypes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        print(f"Agent saved to {filepath}")
    
    def train_random_forest(self, 
                           X_train: pd.DataFrame = None,
                           y_train: pd.Series = None,
                           X_test: pd.DataFrame = None,
                           y_test: pd.Series = None,
                           test_size: float = 0.2,
                           random_state: int = 42,
                           n_estimators: int = 100,
                           max_depth: int = 10,
                           min_samples_split: int = 2,
                           min_samples_leaf: int = 1,
                           max_features: str = "sqrt",
                           n_jobs: int = -1) -> Dict[str, Any]:
        """
        Train a Random Forest model and update the agent's pipeline.
        
        Args:
            X_train: Training features (if None, uses agent's X_train)
            y_train: Training labels (required if X_train provided)
            X_test: Test features (optional, will split if not provided)
            y_test: Test labels (optional, will split if not provided)
            test_size: Test set size if splitting (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of trees (default: 10)
            min_samples_split: Minimum samples to split a node (default: 2)
            min_samples_leaf: Minimum samples in a leaf (default: 1)
            max_features: Number of features to consider for best split (default: "sqrt")
            n_jobs: Number of parallel jobs (default: -1)
        
        Returns:
            Dictionary with training results including:
            - success: Whether training was successful
            - roc_auc: ROC AUC score on test set
            - classification_report: Classification report text
            - message: Status message
        """
        try:
            # Use agent's training data if not provided
            if X_train is None:
                # Need to get y_train from somewhere - this is a limitation
                # For now, we'll require both X_train and y_train
                raise ValueError("X_train and y_train must be provided for training")
            
            if y_train is None:
                raise ValueError("y_train must be provided for training")
            
            # Split data if test set not provided
            if X_test is None or y_test is None:
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                    X_train, y_train, test_size=test_size, 
                    stratify=y_train, random_state=random_state
                )
            else:
                X_train_split = X_train
                X_test_split = X_test
                y_train_split = y_train
                y_test_split = y_test
            
            # Create Random Forest pipeline
            print("Creating Random Forest pipeline...")
            # Handle max_features: convert "None" string to None, "auto" to "sqrt"
            max_features_param = None if max_features == "None" else ("sqrt" if max_features == "auto" else max_features)
            
            rf_pipeline = create_rf_pipeline(
                cat_cols=self.cat_cols,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features_param,
                n_jobs=n_jobs,
                random_state=random_state
            )
            
            # Train and evaluate
            print("Training Random Forest model...")
            y_proba, y_pred, roc_auc = train_and_evaluate(
                rf_pipeline, X_train_split, y_train_split, X_test_split, y_test_split
            )
            
            # Update agent's pipeline and training data
            self.pipeline = rf_pipeline
            self.X_train = X_train_split
            
            # Reinitialize SHAP explainer with new model
            print("Reinitializing SHAP explainer with Random Forest model...")
            self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])
            print("SHAP explainer ready!")
            
            return {
                "success": True,
                "roc_auc": float(roc_auc),
                "message": f"Random Forest model trained successfully! ROC AUC: {roc_auc:.4f}",
                "model_type": "RandomForest",
                "parameters": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features
                },
                "pipeline": rf_pipeline,
                "X_train": X_train_split,
                "X_test": X_test_split,
                "y_train": y_train_split,
                "y_test": y_test_split,
                "y_proba": y_proba,
                "y_pred": y_pred
            }
            
        except Exception as e:
            return {
                "success": False,
                "roc_auc": None,
                "message": f"Training failed: {str(e)}",
                "error": str(e)
            }
    
    def train_xgboost(self,
                     X_train: pd.DataFrame = None,
                     y_train: pd.Series = None,
                     X_test: pd.DataFrame = None,
                     y_test: pd.Series = None,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     use_optimization: bool = False,
                     n_iter: int = 30,
                     cv: int = 3,
                     scoring: str = 'roc_auc',
                     n_jobs: int = -1) -> Dict[str, Any]:
        """
        Train an XGBoost model and update the agent's pipeline.
        
        Args:
            X_train: Training features (required)
            y_train: Training labels (required)
            X_test: Test features (optional, will split if not provided)
            y_test: Test labels (optional, will split if not provided)
            test_size: Test set size if splitting (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            use_optimization: Whether to use RandomizedSearchCV for hyperparameter optimization (default: False)
            n_iter: Number of iterations for optimization (default: 30)
            cv: Number of cross-validation folds (default: 3)
            scoring: Scoring metric for optimization (default: 'roc_auc')
            n_jobs: Number of parallel jobs (default: -1)
        
        Returns:
            Dictionary with training results including:
            - success: Whether training was successful
            - roc_auc: ROC AUC score on test set
            - message: Status message
            - model_type: "XGBoost"
            - parameters: Best hyperparameters if optimization used
        """
        try:
            if X_train is None:
                raise ValueError("X_train must be provided for training")
            
            if y_train is None:
                raise ValueError("y_train must be provided for training")
            
            # Split data if test set not provided
            if X_test is None or y_test is None:
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                    X_train, y_train, test_size=test_size,
                    stratify=y_train, random_state=random_state
                )
            else:
                X_train_split = X_train
                X_test_split = X_test
                y_train_split = y_train
                y_test_split = y_test
            
            # Create and train XGBoost pipeline
            if use_optimization:
                print("Creating XGBoost pipeline with hyperparameter optimization...")
                print("⏱ This may take several minutes...")
                xgb_pipeline, best_params, cv_results = optimize_XGBoost_hyperparameters(
                    X_train_split, y_train_split, self.cat_cols,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    random_state=random_state,
                    n_jobs=n_jobs
                )
                optimization_info = "Hyperparameter optimization applied"
            else:
                print("Creating XGBoost pipeline with default parameters...")
                xgb_pipeline = create_XGBoost_pipeline(self.cat_cols)
                best_params = {}
                optimization_info = "Using default hyperparameters"
            
            # Train and evaluate
            print("Training XGBoost model...")
            y_proba, y_pred, roc_auc = train_and_evaluate(
                xgb_pipeline, X_train_split, y_train_split, X_test_split, y_test_split
            )
            
            # Update agent's pipeline and training data
            self.pipeline = xgb_pipeline
            self.X_train = X_train_split
            
            # Reinitialize SHAP explainer with new model
            print("Reinitializing SHAP explainer with XGBoost model...")
            self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])
            print("SHAP explainer ready!")
            
            return {
                "success": True,
                "roc_auc": float(roc_auc),
                "message": f"XGBoost model trained successfully! ROC AUC: {roc_auc:.4f}",
                "model_type": "XGBoost",
                "optimization_info": optimization_info,
                "parameters": best_params,
                "pipeline": xgb_pipeline,
                "X_train": X_train_split,
                "X_test": X_test_split,
                "y_train": y_train_split,
                "y_test": y_test_split,
                "y_proba": y_proba,
                "y_pred": y_pred
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "roc_auc": None,
                "message": f"Training failed: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    @classmethod
    def load(cls, filepath: str):
        """Load an agent from a file."""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        agent = cls(
            pipeline=agent_data["pipeline"],
            X_train=agent_data["X_train"],
            cat_cols=agent_data["cat_cols"],
            feature_columns=agent_data.get("feature_columns")
        )
        print(f"Agent loaded from {filepath}")
        return agent


def create_agent_from_training(pipeline, X_train, cat_cols, feature_columns=None):
    """
    Factory function to create a PPD Agent from training results.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_train: Training data
        cat_cols: List of categorical column names
        feature_columns: Optional list of feature column names
    
    Returns:
        PPDAgent instance
    """
    return PPDAgent(pipeline, X_train, cat_cols, feature_columns)

