# 📌 ML libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier


def create_pipeline(cat_cols):
    """
    Create a preprocessing and modeling pipeline for XGBoost classifier.
    
    Args:
        cat_cols: List of categorical column names
        
    Returns:
        sklearn Pipeline object
    """
    # 📌 OneHotEncode categorical columns
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocess = ColumnTransformer(
        transformers=[("cat", categorical_transformer, cat_cols)]
    )
    
    # 📌 XGBoost classifier
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    # 📌 Pipeline
    pipeline = Pipeline(steps=[("preprocess", preprocess),
                               ("model", xgb_model)])
    
    return pipeline


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    """
    Train the pipeline and evaluate on test set.
    
    Args:
        pipeline: sklearn Pipeline object
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (y_proba, y_pred, roc_auc)
    """
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # 💡 Predict probabilities and classes
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    # 📊 Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", roc_auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return y_proba, y_pred, roc_auc