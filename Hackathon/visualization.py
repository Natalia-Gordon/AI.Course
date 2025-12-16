# 📌 Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap


def plot_target_distribution(y, title="Target Distribution"):
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(8, 6))
    counts = y.value_counts()
    labels = ['No Depression', 'Depression'] if 0 in counts.index else ['Depression', 'No Depression']
    colors = ['#2ecc71', '#e74c3c']
    plt.pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, cat_cols, target, n_cols=3):
    """Plot distributions of categorical features by target."""
    n_features = len(cat_cols)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(cat_cols[:n_features]):
        ax = axes[idx]
        crosstab = pd.crosstab(df[col], df[target])
        crosstab.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_title(f'{col} by {target}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.legend(['No', 'Yes'])
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Depression', 'Depression'],
                yticklabels=['No Depression', 'Depression'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_test, y_proba, roc_auc, title="ROC Curve"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498db', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_shap_summary(pipeline, X_test, cat_cols, max_display=10):
    """Plot SHAP summary for feature importance."""
    try:
        # Get the preprocessed features
        preprocessor = pipeline.named_steps['preprocess']
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()
        
        # Get the model
        model = pipeline.named_steps['model']
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_processed[:100])  # Use subset for speed
        
        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_processed[:100], 
                         feature_names=feature_names[:len(shap_values[0])],
                         max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
        
        # Plot bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_processed[:100], 
                         feature_names=feature_names[:len(shap_values[0])],
                         plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"SHAP visualization error: {e}")
        print("Skipping SHAP plots...")


def plot_correlation_heatmap(df, target):
    """Plot correlation heatmap for numerical features."""
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def plot_prediction_distribution(y_proba, title="Prediction Probability Distribution"):
    """Plot distribution of prediction probabilities."""
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='#e74c3c', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_all_visualizations(df, X_test, y_test, y_pred, y_proba, roc_auc, 
                            pipeline, cat_cols, target):
    """Create all visualizations for the model."""
    print("\n" + "="*50)
    print("Creating Visualizations...")
    print("="*50)
    
    # 1. Target distribution
    print("\n1. Plotting target distribution...")
    plot_target_distribution(df[target])
    
    # 2. Feature distributions
    if len(cat_cols) > 0:
        print("\n2. Plotting feature distributions...")
        plot_feature_distributions(df, cat_cols, target)
    
    # 3. Confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    # 4. ROC curve
    print("\n4. Plotting ROC curve...")
    plot_roc_curve(y_test, y_proba, roc_auc)
    
    # 5. Prediction distribution
    print("\n5. Plotting prediction probability distribution...")
    plot_prediction_distribution(y_proba)
    
    # 6. Correlation heatmap
    print("\n6. Plotting correlation heatmap...")
    plot_correlation_heatmap(df, target)
    
    # 7. SHAP plots
    print("\n7. Creating SHAP visualizations...")
    plot_shap_summary(pipeline, X_test, cat_cols)
    
    print("\n✅ All visualizations completed!")