# Kickstarter Success Prediction

Can we predict whether a Kickstarter campaign will succeed **before it launches**?

This project builds and compares 7 machine learning models that predict campaign success using only information an entrepreneur has at decision time: the funding goal, the category, the campaign length and the launch month. The goal is a practical decision-support tool for entrepreneurs, not just a high accuracy score.

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.665** | 0.609 | **0.473** | **0.533** | **0.710** |
| Random Forest | 0.650 | **0.619** | 0.350 | 0.447 | 0.695 |
| Logistic Regression | 0.646 | 0.590 | 0.402 | 0.478 | 0.680 |
| Linear SVM* | 0.638 | 0.589 | 0.340 | 0.431 | 0.671 |

\* Trained on a 5,000-row subsample for runtime reasons. Decision Tree, Naive Bayes and KNN were also evaluated (full table in the notebook).

- **Best model:** XGBoost, with ROC-AUC of 0.710 on the test set.
- **Stable:** 5-fold cross-validation gave a mean ROC-AUC of 0.712 (std 0.0015).
- **No overfitting:** the gap between train and test ROC-AUC is under 0.01.

## Main Insights

- **The funding goal is the strongest predictor.** Higher goals sharply reduce the chance of success. This was confirmed by both permutation importance and SHAP values.
- **Category matters,** and comes close behind the goal in most models.
- **Longer campaigns do slightly worse,** not better.
- **Launch month has almost no effect.**

## Approach

1. **Data:** 378,661 Kickstarter projects ([Kaggle Kickstarter Projects dataset](https://www.kaggle.com/kemical/kickstarter-projects), file `ks-projects-201801.csv`), filtered to 331,675 projects that clearly succeeded or failed.
2. **Avoiding data leakage:** features like number of backers and amount pledged correlate strongly with success, but they only exist *after* the campaign starts. They were deliberately excluded, so the model reflects what an entrepreneur can actually know in advance.
3. **Feature engineering:** log transform of the funding goal (heavily skewed distribution), campaign duration in days, launch month and category as one-hot encoded variables.
4. **Modeling:** stratified 80/20 train/test split, standardization fit on the training set only, and 7 classifiers: Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, Linear SVM and XGBoost.
5. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, cross-validation and an overfitting check.
6. **Explainability:** tree-based feature importance, permutation importance and SHAP values.

## Tech Stack

Python, Pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn

## How to Run

1. Download `ks-projects-201801.csv` from the Kaggle dataset linked above.
2. Place it in the same folder as the notebook (or upload it to Google Colab).
3. Open `Kickstarter_Success_Prediction.ipynb` and run all cells.

> The notebook's explanations are written in Hebrew. The code comments are in English.

---

Built as part of the B.Sc. in Information Systems at The Academic College of Tel Aviv-Yafo.
