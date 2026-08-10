# Imbalanced Dataset Classification with SMOTE, ADASYN & Borderline-SMOTE
 
A comparative study of resampling techniques for handling class imbalance in coronary artery disease (CAD) prediction, using the **Z-Alizadeh Sani dataset**. Trains four classifiers on the raw imbalanced data, then re-trains the same classifiers after applying three different oversampling methods, comparing performance across both settings.
 
## 📊 Dataset
 
- **Source:** Z-Alizadeh Sani dataset (heart disease / coronary artery disease clinical records)
- **Size:** 303 patients, 55 columns (34 numeric, 20 categorical, 1 target)
- **Target column:** `Cath` — binarized to `1` for CAD-positive (`Cad`) and `0` otherwise
- **Class balance:** 216 positive / 87 negative (imbalanced ~2.5:1)
## 🧪 Methodology
 
1. **Preprocessing** — `ColumnTransformer` applying `StandardScaler` to numeric features and `OneHotEncoder` to categorical features
2. **Split** — stratified 80/20 train-test split (`random_state=42`)
3. **Baseline** — train each model directly on the imbalanced training data
4. **Resampling** — re-train each model after balancing the training set with:
   - SMOTE
   - ADASYN
   - Borderline-SMOTE
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, AUC, and confusion matrix for every model × resampling combination
## 🤖 Models Compared
 
- Random Forest (`n_estimators=100`)
- Support Vector Machine (RBF kernel, `probability=True`)
- Artificial Neural Network (`MLPClassifier`, hidden layers `(32, 16)`)
- Logistic Regression (`liblinear`, L2 penalty)
## 📈 Results
 
| Model | Resampling | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|---|
| SVM | None | 0.90 | 0.88 | 1.00 | 0.93 | 0.89 |
| ANN | None | 0.85 | 0.85 | 0.95 | 0.90 | 0.89 |
| Logistic Regression | None | 0.84 | 0.87 | 0.91 | 0.89 | 0.89 |
| Random Forest | None | 0.80 | 0.79 | 0.98 | 0.88 | 0.89 |
| SVM | SMOTE | 0.85 | 0.87 | 0.93 | 0.90 | 0.89 |
| ANN | SMOTE | 0.85 | 0.87 | 0.93 | 0.90 | 0.89 |
| Random Forest | SMOTE | 0.82 | 0.83 | 0.93 | 0.88 | 0.85 |
| Logistic Regression | SMOTE | 0.77 | 0.85 | 0.81 | 0.83 | 0.88 |
| SVM | ADASYN | 0.85 | 0.87 | 0.93 | 0.90 | 0.88 |
| Random Forest | ADASYN | 0.82 | 0.83 | 0.93 | 0.88 | 0.88 |
| SVM | Borderline-SMOTE | 0.85 | 0.87 | 0.93 | 0.90 | 0.89 |
| Random Forest | Borderline-SMOTE | 0.82 | 0.85 | 0.91 | 0.88 | 0.86 |
 
Full results for all 16 model × resampling combinations are written to `model_results_all.csv` on each run.
 
### Key finding
 
The unresampled baseline SVM (Accuracy 0.90, Recall 1.00) outperforms every resampled variant tested. On this dataset, oversampling does not improve — and in most cases slightly reduces — classifier performance relative to training directly on the imbalanced data. This is a legitimate outcome worth discussing rather than a failure of the resampling methods: imbalance correction does not universally help, particularly on small datasets (303 samples) where synthetic minority samples can introduce noise.
 
## 🛠️ Requirements
 
```
numpy
pandas
scikit-learn
imbalanced-learn
tabulate
```
 
Install with:
 
```bash
pip install numpy pandas scikit-learn imbalanced-learn tabulate
```
 
## ▶️ How to Run
 
This notebook was built for **Google Colab** and expects the dataset at a Colab-specific path.
 
1. Open the notebook in Colab (badge link at the top of the notebook) or upload it manually.
2. Upload the dataset CSV to your Colab session (`/content/`).
3. Update the `DATA_PATH` variable near the top of the code cell to match your uploaded filename:
```python
   DATA_PATH = "/content/Z-Alizadeh sani dataset (2).csv"
```
4. Run the cell. Console output will print baseline and resampled confusion matrices, followed by comparison tables (`TABLE 4.1`, `TABLE 4.6`, `TABLE 4.11`), and save all results to `model_results_all.csv`.
To run locally instead of Colab, change `DATA_PATH` to a local file path and ensure the CSV is available at that location.
 
## ⚠️ Known Issues
 
- **Indentation error in the current source:** the code around the `TABLE: ALL MODEL RESULTS` print block has inconsistent indentation (an `if TABULATE_AVAILABLE:` line indented deeper than the preceding statement with no block opener), which raises an `IndentationError` if run as currently committed. This needs a fix before the notebook will execute top to bottom.
- **Stored output is stale:** the notebook's saved output includes `====` section separators that are not present anywhere in the current source code, meaning the output shown was captured from an earlier version of the code prior to a later edit. Re-run and re-save the notebook after fixing the indentation issue above so the displayed output matches the actual code.
- **Hardcoded Colab path:** `DATA_PATH` points to `/content/...` with a `# TODO` comment — must be edited before running outside of a fresh Colab upload.
 
