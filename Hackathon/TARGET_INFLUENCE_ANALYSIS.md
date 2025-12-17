# How Target Variable Influences SHAP Values and PPD Risk Score

## Overview

The target variable **"Feeling anxious"** directly influences both SHAP values and the PPD Risk Score through:
1. **Class distribution** (64.9% Yes, 35.1% No)
2. **Data quality patterns** (unusual symptom-to-risk relationships)
3. **Model learning** (the model learns patterns from the target variable)

---

## 1. Target Variable Encoding

```
"Feeling anxious" → Encoded as:
  - "Yes" = 1 (has anxiety/depression)
  - "No" = 0 (no anxiety/depression)
```

**Important Note:** The model predicts "Feeling anxious" (anxiety), not "Postpartum Depression" directly. It's used as a **proxy** for PPD risk.

---

## 2. Impact on PPD Risk Score

### Current Implementation
```python
PPD Risk Score = prob_class_1 = P('Feeling anxious' = 'Yes')
```

### How Target Distribution Affects Risk Score

**Baseline Probability:**
- Since 64.9% of training data has "Feeling anxious" = "Yes"
- The model has a **baseline ~65% probability** for class 1
- This means even with no symptoms, the model starts with ~65% risk

**Unusual Patterns Learned:**
The model learns counterintuitive patterns from the data:

| Symptom Count | Training Data Shows | Model Learns |
|--------------|---------------------|--------------|
| 0 symptoms   | 9.2% depression     | Low risk (~9%) |
| 1-2 symptoms | 77-81% depression   | **Very high risk** |
| 5 symptoms   | 10.3% depression    | **Very low risk** |
| 7-8 symptoms | 100% depression      | High risk (100%) |

**Result:** The risk score reflects these learned patterns, not medical intuition.

---

## 3. Impact on SHAP Values

### Current SHAP Implementation
```python
# Line 179 in gradio_app.py
shap_values = explainer.shap_values(row_processed)

# Line 184: Uses class 1 (Yes depression)
shap_values = shap_values[1]  # Positive class
```

### How Target Affects SHAP Values

**1. Base SHAP Value (Starting Point)**
- SHAP values are calculated relative to the **expected value** (baseline)
- With 64.9% class 1 in training data, the baseline is high
- This shifts all SHAP values upward

**2. Feature Contributions**
- **Positive SHAP value** → Feature **increases** probability of class 1 (Yes)
- **Negative SHAP value** → Feature **decreases** probability of class 1 (Yes)
- The magnitude shows how much the feature contributes

**3. Class Imbalance Effect**
- Since class 1 (Yes) is the majority (64.9%), the model is biased toward predicting it
- This affects:
  - Which features get positive vs negative SHAP values
  - The magnitude of SHAP contributions
  - Feature importance rankings

**4. Unusual Data Patterns**
- The model learned that 1-2 symptoms = high risk (77-81%)
- The model learned that 5 symptoms = low risk (10.3%)
- SHAP values reflect these learned patterns, not medical reality

---

## 4. Example: Feature Correlations with Target

From the analysis, here are feature correlations with "Feeling anxious":

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| Feeling of guilt | -0.520 | **Strong negative** - "Yes" guilt → Lower anxiety |
| Overeating/loss of appetite | -0.192 | Negative - "Yes" appetite issues → Lower anxiety |
| Irritable towards baby & partner | +0.182 | Positive - "Yes" irritability → Higher anxiety |
| Problems of bonding with baby | +0.108 | Positive - "Yes" bonding issues → Higher anxiety |
| Suicide attempt | +0.078 | Positive - "Yes" suicide attempt → Higher anxiety |

**⚠️ Counterintuitive Finding:**
- "Feeling of guilt" has a **strong negative correlation** (-0.520)
- This means "Yes" guilt is associated with **lower** anxiety in the data
- This is medically counterintuitive and affects SHAP values

---

## 5. How This Affects Predictions

### Example 1: High Risk Case (7 symptoms)
- **Expected:** High PPD risk
- **Model prediction:** `prob_class_1 = 85.48%` ✓ Correct
- **SHAP values:** Features contributing to high risk show positive values

### Example 2: Low Risk Case (0 symptoms)
- **Expected:** Low PPD risk (~9% based on data)
- **Model prediction:** `prob_class_1 = 97.92%` ✗ Wrong
- **SHAP values:** May show incorrect feature contributions due to learned patterns

### Why Example 2 is Wrong:
- The model learned that 0 symptoms → 9.2% risk (from training data)
- But it's predicting 97.92% instead
- This suggests the model's learned patterns are inconsistent

---

## 6. Recommendations

### To Improve SHAP Values:
1. **Address class imbalance:**
   ```python
   # In MLmodel.py, add class_weight
   XGBClassifier(..., class_weight='balanced')
   ```

2. **Use calibrated probabilities:**
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated_model = CalibratedClassifierCV(model, method='isotonic')
   ```

3. **Calculate SHAP for both classes:**
   ```python
   shap_values_class_0 = explainer.shap_values(row_processed)[0]
   shap_values_class_1 = explainer.shap_values(row_processed)[1]
   # Show both to understand full picture
   ```

### To Improve Risk Scores:
1. **Fix data quality issues** (investigate unusual patterns)
2. **Use model calibration** for more accurate probabilities
3. **Consider alternative target** (direct PPD diagnosis vs "Feeling anxious")
4. **Document limitations** due to data quality issues

---

## 7. Summary

**Target Variable Influence:**

1. **PPD Risk Score:**
   - Directly = `prob_class_1` = P('Feeling anxious' = 'Yes')
   - Affected by: class imbalance (65% baseline), unusual data patterns
   - Result: Inconsistent risk scores that don't match medical intuition

2. **SHAP Values:**
   - Calculated for class 1 (Yes anxiety/depression)
   - Affected by: class imbalance, learned patterns, feature correlations
   - Result: SHAP values reflect learned patterns (including counterintuitive ones)

**Key Insight:** The target variable "Feeling anxious" is a proxy for PPD, but the data quality issues and class imbalance cause both the risk scores and SHAP values to be inconsistent with medical expectations.

