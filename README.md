# BITS Pilani M.Tech (AIML) — Machine Learning Assignment 2

## a. Problem statement

Build and evaluate five classification algorithms on one public classification dataset, then provide an interactive Streamlit application that loads saved models and evaluates uploaded test data without retraining.

This project predicts whether a breast tumour is **malignant (1)** or **benign (0)** using 30 diagnostic measurements. This is an educational machine-learning demonstration and is not a medical diagnostic tool.

## b. Dataset description

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), distributed through `sklearn.datasets.load_breast_cancer`.
- **Public origin:** UCI Machine Learning Repository.
- **Observations:** 569.
- **Input features:** 30 numeric measurements derived from digitized fine-needle aspirate images.
- **Target:** two classes. The original scikit-learn encoding is converted so malignant = 1 and benign = 0.
- **Eligibility:** 30 features (minimum 12) and 569 instances (minimum 500).
- **Split:** stratified 80% training / 20% testing; `random_state=42`.
- **Preprocessing:** StandardScaler is included inside the Logistic Regression, KNN, Gaussian Naive Bayes and SVM pipelines. The Decision Tree is trained without scaling.
- **Important:** the test CSV is held out from model fitting. The Streamlit app only loads the saved fitted model and calculates test metrics.

## c. GitHub Repository Link:  https://github.com/kiran1488/bits-aiml-ml-assignment-2

Repository contents:
- `app.py` — Streamlit frontend and evaluation logic.
- `train_models.py` — reproducible model training and evaluation.
- `requirements.txt` — deployment dependencies.
- `README.md` — project documentation and evaluation results.
- `test_data.csv` and `data/test_data.csv` — held-out test data including target.
- `model/` — five fitted model pipelines saved as Joblib files.
- `model_metrics.csv` — metrics produced by the reproducible training run.

## d. Models used and evaluation metrics

All five models use the same stratified hold-out test split. Precision, recall and F1 are calculated for the positive class (malignant). AUC uses the predicted probability of the malignant class. MCC ranges from -1 to +1.

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression    |     0.9649 | 0.9960 |      0.9750 |   0.9286 |     0.9512 | 0.9245 |
| Decision Tree          |     0.9035 | 0.9471 |      0.9429 |   0.7857 |     0.8571 | 0.7926 |
| KNN                    |     0.9561 | 0.9823 |      0.9744 |   0.9048 |     0.9383 | 0.9058 |
| Gaussian Naive Bayes   |     0.9211 | 0.9891 |      0.9231 |   0.8571 |     0.8889 | 0.8292 |
| Support Vector Machine |     0.9737 | 0.9947 |      1.0000 |   0.9286 |     0.9630 | 0.9442 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A linear decision boundary provides a strong baseline. Standardisation is included because the feature scales differ. Its results should be interpreted alongside recall and AUC, not accuracy alone. |
| Decision Tree | The tree can model nonlinear feature interactions and is comparatively interpretable. Its depth and minimum leaf size are constrained to reduce overfitting; a single tree may still be less stable than ensemble methods. |
| KNN | KNN classifies using nearby training observations. Standardisation is essential because distance calculations are scale-sensitive. Performance depends on the selected neighbour count (k=5). |
| Naive Bayes | Gaussian Naive Bayes is fast and makes a conditional-independence assumption. Correlated diagnostic measurements can limit this assumption, so its scores may be lower than more flexible models. |
| Support Vector Machine | The RBF-kernel SVM can capture nonlinear boundaries. Standardisation is used and probability estimates are enabled to calculate AUC. It is a strong candidate for this dataset. |
| **Overall winner** | Support Vector Machine has the highest F1 score; Logistic Regression has the highest AUC. Select according to the priority between thresholded classification and ranking ability. |

The winner statement is based on this particular split and the F1 score, with AUC considered as a complementary ranking metric. For a high-stakes application, further validation, calibration, subgroup analysis and clinical review would be necessary.

## Streamlit application

The app includes:
1. CSV test-data upload.
2. Model-selection dropdown for all five trained classifiers.
3. Display of Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix.
5. Classification report.
6. Prediction preview and downloadable predictions.

The CSV must contain all 30 feature columns and the `target` column. The app checks feature compatibility and does not retrain on uploaded data.

**Live Streamlit Community Cloud URL:** https://bits-aiml-ml-assignment-2-ah77seege8rbcrpdonwfnr.streamlit.app/

## How to run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Upload `test_data.csv` in the sidebar to evaluate a model.

## Reproducibility and limitations

- Fixed random seed: 42.
- Stratified train/test split: 80/20.
- Model hyperparameters are explicitly set in `train_models.py`.
- The test split is used only for evaluation, not model training.
- Results can vary if the dataset version, software versions, split or model settings change.
- Do not use this educational model for clinical decisions.

## Assignment-specific final submission items

- GitHub repository URL: **https://github.com/kiran1488/bits-aiml-ml-assignment-2**.
- Live Streamlit app URL: **https://bits-aiml-ml-assignment-2-ah77seege8rbcrpdonwfnr.streamlit.app/**.
- BITS Virtual Lab execution screenshot: **Uploaded in final submission PDF**.
