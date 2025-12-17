# Target Variable Influence Analysis

## Analyzing Target Variable Influence on SHAP Values and PPD Risk Score

---

## 1. Target Variable Analysis

**Target variable:** `'Feeling anxious'`

### Original Distribution

| Value | Count | Proportion |
|-------|-------|------------|
| Yes   | 968   | 64.9%      |
| No    | 523   | 35.1%      |

### After Encoding

- **Encoding:** `'Yes' = 1` (depression/anxiety), `'No' = 0` (no depression/anxiety)
- **Class 0 (No):** 523 cases (35.1%)
- **Class 1 (Yes):** 968 cases (64.9%)

---

## 2. How Target Affects Model Predictions

The model learns to predict **P(class 1) = P('Feeling anxious' = 'Yes')**

This means:
- **prob_class_0** = P(No anxiety/depression)
- **prob_class_1** = P(Yes anxiety/depression) = PPD Risk Score

> **⚠️ IMPORTANT:** The target is `'Feeling anxious'`, not `'Postpartum Depression'`.  
> The model predicts anxiety, which is used as a proxy for PPD risk.

---

## 3. SHAP Values Explanation

SHAP values explain how each feature contributes to the prediction.

### Current Implementation:
- SHAP values are calculated for **class 1** (Yes anxiety/depression)
- **Positive SHAP value** → increases probability of class 1
- **Negative SHAP value** → decreases probability of class 1

### Code Reference:
```python
shap_values = explainer.shap_values(row_processed)
shap_values = shap_values[1]  # Uses class 1 (Yes depression)
```

---

## 4. Target Variable Impact on Predictions

### Feature Correlation with Target (Feeling anxious)

The correlation between each symptom feature and the target variable shows how strongly each symptom is associated with anxiety:

| Feature | Correlation |
|---------|-------------|
| Irritable towards baby & partner | 0.182 |
| Problems of bonding with baby | 0.108 |
| Suicide attempt | 0.078 |
| Feeling sad or Tearful | -0.016 |
| Problems concentrating or making decision | -0.009 |
| Trouble sleeping at night | -0.026 |
| Overeating or loss of appetite | -0.192 |
| Feeling of guilt | -0.520 |

**Key Insights:**
- **Highest positive correlation:** Irritable towards baby & partner (0.182)
- **Highest negative correlation:** Feeling of guilt (-0.520) - interesting finding!
- Most symptoms show weak or negative correlations with the target

---

## 5. Example: How Target Distribution Affects Predictions

### Symptom Count vs Target (Feeling anxious)

This analysis shows how the number of symptoms relates to the target variable:

| Symptom Count | Cases | % with 'Feeling anxious' = Yes | Model Learning |
|---------------|-------|-------------------------------|----------------|
| 0 symptoms    | 65    | 9.2%                          | Model learns: 0 symptoms → 9.2% PPD risk |
| 1 symptom     | 325   | 77.2%                         | Model learns: 1 symptom → 77.2% PPD risk ⚠️ |
| 2 symptoms    | 340   | 80.9%                         | Model learns: 2 symptoms → 80.9% PPD risk ⚠️ |
| 3 symptoms    | 182   | 60.4%                         | Model learns: 3 symptoms → 60.4% PPD risk |
| 4 symptoms    | 217   | 51.2%                         | Model learns: 4 symptoms → 51.2% PPD risk |
| 5 symptoms    | 39    | 10.3%                         | Model learns: 5 symptoms → 10.3% PPD risk ⚠️ |
| 6 symptoms    | 13    | 46.2%                         | Model learns: 6 symptoms → 46.2% PPD risk |
| 7 symptoms    | 1     | 100.0%                        | Model learns: 7 symptoms → 100% PPD risk |
| 8 symptoms    | 10    | 100.0%                        | Model learns: 8 symptoms → 100% PPD risk |

**⚠️ Critical Findings:**
- **Unusual pattern:** 1-2 symptoms show very high risk (77-81%), which is counterintuitive
- **Very unusual:** 5 symptoms show very low risk (10.3%), which doesn't align with medical intuition
- These patterns suggest data quality issues that affect model learning

---

## 6. Impact on SHAP Values

SHAP values are calculated based on:

1. **The model's learned patterns** from training data
2. **The specific input features** for a prediction
3. **The target variable distribution** (class imbalance)

### Class Distribution Impact:

Since the target has:
- **Class 0 (No):** 35.1%
- **Class 1 (Yes):** 64.9%

**The model is biased toward predicting class 1 (Yes) due to class imbalance.**

This affects:
- **Base SHAP values** (starting point)
- **Feature importance rankings**
- **Individual feature contributions**

---

## 7. Impact on PPD Risk Score

The **PPD Risk Score = prob_class_1 = P('Feeling anxious' = 'Yes')**

### This is affected by:

1. **Target variable distribution (64.9% Yes, 35.1% No)**
   - → Model has baseline ~65% probability for class 1

2. **Unusual data patterns (1-2 symptoms = high risk, 5 symptoms = low risk)**
   - → Model learns counterintuitive relationships

3. **Class imbalance**
   - → Model tends to predict class 1 more often

---

## 8. Recommendations

To improve SHAP values and risk scores:

1. **Address class imbalance**
   - Use `class_weight='balanced'` or SMOTE (Synthetic Minority Oversampling Technique)

2. **Fix data quality issues**
   - Investigate unusual symptom patterns in the dataset

3. **Consider target variable choice**
   - Evaluate if `'Feeling anxious'` is the right target for PPD prediction
   - Consider using a composite target based on multiple symptoms

4. **Use calibrated probabilities**
   - Apply probability calibration techniques (Platt scaling, isotonic regression) for more accurate risk scores

5. **Calculate SHAP for both classes**
   - Calculate and visualize SHAP values for both class 0 and class 1 to see the full picture
   - This provides complete insight into feature contributions

---

## 9. Key Takeaways

1. The current target variable (`'Feeling anxious'`) creates a class imbalance (65% vs 35%)
2. The model learns patterns that may not align with medical intuition due to data quality issues
3. SHAP values are currently only calculated for class 1, limiting interpretability
4. The PPD risk score is heavily influenced by the baseline probability of ~65%
5. Multiple improvements can be made to enhance model performance and interpretability

---

## Analysis Complete

This analysis provides insights into how the target variable structure affects model predictions, SHAP value interpretations, and PPD risk scoring accuracy.
