# Week 3: Evaluation and Implementation Basics

##  Objectives
* Manually segment data into Training, Validation, and Test distributions to prevent data leakage.
* Build a standard Confusion Matrix from raw binary predictions.
* Calculate structural evaluation metrics beyond simple accuracy to account for imbalanced datasets.

##  The Data Split Architecture
A model that perfectly memorizes its training data has **high variance (overfitting)**. To diagnose this, we split the data:

1. **Training Set (80%):** The environment where the model calculates gradients and updates weights.
2. **Validation Set (10%):** Unseen data used during the training loop. If training loss drops but validation loss spikes, the model is memorizing noise.
3. **Test Set (10%):** The final evaluation environment, strictly used only once after all training and hyperparameter tuning is complete.

##  Evaluation Mathematics
For classification, Mean Squared Error (MSE) is ineffective. We evaluate probability distributions using a **Confusion Matrix** and the following derived metrics:

* **Accuracy:** The percentage of correct predictions across all classes.
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

* **Precision:** The model's exactness. When it guesses positive, how often is it right?
  $$\text{Precision} = \frac{TP}{TP + FP}$$

* **Recall (Sensitivity):** The model's completeness. Out of all actual positive cases, how many did it successfully find?
  $$\text{Recall} = \frac{TP}{TP + FN}$$

* **F1-Score:** The harmonic mean of Precision and Recall, heavily penalizing extreme imbalances between the two.
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

##  Usage
To run the manual dataset splitter and generate a randomized confusion matrix evaluation, execute:

```bash
python evaluation.py
