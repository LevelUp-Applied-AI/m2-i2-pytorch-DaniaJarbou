[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field
---
## 📝 My Project Documentation (Integration 2)

### 1. What the Model Predicts
The model predicts the **housing price in JOD** (`price_jod`) using the following 5 input features:
* `area_sqm`: Total area in square meters.
* `bedrooms`: Number of bedrooms.
* `floor`: Apartment floor level.
* `age_years`: Building age.
* `distance_to_center_km`: Distance to city center.

### 2. Training Configuration
* **Epochs:** 100
* **Learning Rate:** 0.01
* **Optimizer:** Adam
* **Loss Function:** MSELoss (Mean Squared Error)

### 3. Training Outcome
* **Loss Decrease:** Observed a consistent decrease throughout training.
* **Initial Loss (Epoch 0):** 1,950,608,256
* **Final Loss (Epoch 99):** 1,943,536,256
* **Predictions:** Successfully generated and saved to `predictions.csv`.

### 4. Behavioral Observation
I observed that the loss decreased steadily right from the first epoch. This was due to the **Standardization** step; by scaling features like `area_sqm` and `distance_to_center_km`, the Adam optimizer was able to update weights efficiently without being dominated by large-scale values.
