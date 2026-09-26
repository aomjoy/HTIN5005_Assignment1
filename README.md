# HTIN5005_Assignment1

Readme · TXT
================================================================================
README - Sepsis Prediction Reimplementation
================================================================================
 
Project: Reimplementation of Mahmud et al. (2026), "Interpretable machine 
learning-based real-time sepsis diagnosis," Scientific Reports, 16, 6702.
https://doi.org/10.1038/s41598-026-36945-w
 
This project reimplements the core pipeline of the paper above using the 
PhysioNet 2019 Computing in Cardiology Challenge dataset. Class imbalance is 
handled using SMOTE instead of the original paper's Non-Overlapping Subset 
Ensemble (NOSE) method, as a deliberate methodological comparison (see report, 
Section: Techniques/Methods).
 
================================================================================
1. DATASET
================================================================================
 
Source: PhysioNet/Computing in Cardiology Challenge 2019
 
Subset used: training_setA only (~20,336 patients), due to time and 
computational resource constraints. Training set B was not used.
 
To download training_setA (requires AWS CLI, --no-sign-request, no account 
needed)
 
Each patient is stored as a single pipe-separated (.psv) file, with one row 
per hour of ICU stay and a "SepsisLabel" column (1 = sepsis, 0 = non-sepsis).
 
================================================================================
2. ENVIRONMENT / REQUIREMENTS
================================================================================
 
Python 3.13 was used. Install required packages with:
 
    pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib joblib
 
================================================================================
3. FOLDER STRUCTURE
================================================================================
 
sepsis_project/
├── training_setA/            <- raw .psv files (downloaded, not included)
├── merge_data.py              
├── forward_fill.py            
├── feature_engineering.py     
├── prepare_train_test.py     
├── train_evaluate.py          
├── explain_shap.py            
└── README.txt                
 
================================================================================
4. HOW TO RUN (in order)
================================================================================
 
All scripts must be run from the sepsis_project/ root folder (not from inside 
training_setA/). Each script loads the .csv output of the previous script, so 
they must be run in the order below. Each step prints progress messages and 
saves its output before the next script can run.
 
    python merge_data.py
        - Merges all .psv files in training_setA/ into one table.
        - Output: merged_data.csv
 
    python forward_fill.py
        - Fills missing values per patient using forward fill (causal, 
          suitable for real-time prediction).
        - Output: filled_data.csv
 
    python feature_engineering.py
        - Computes qSOFA score (Table 1 of Mahmud et al.), NEWS score 
          (Table 2), lag features, and expanding-window mean/max/min for 
          five vital signs (HR, O2Sat, Temp, SBP, Resp).
        - Output: featured_data.csv
 
    python prepare_train_test.py
        - Selects the top 50 features using ANOVA F-score (SelectKBest).
        - Splits data into train (80%) / test (20%), stratified by label.
        - Applies SMOTE to the training set only (not the test set, to 
          avoid data leakage).
        - Output: X_train.csv, X_test.csv, y_train.csv, y_test.csv
 
    python train_evaluate.py
        - Trains Random Forest and XGBoost classifiers on the resampled 
          training data.
        - Evaluates both models on the held-out test set (Accuracy, AUROC, 
          precision/recall/F1 per class).
        - Output: rf_model.pkl, xgb_model.pkl (trained models)
 
    python explain_shap.py
        - Loads the trained Random Forest model.
        - Computes SHAP values on a random sample of 200 test instances 
          (TreeExplainer; full test set was not used due to runtime).
        - Output: shap_summary_plot.png, shap_bar_plot.png
 
================================================================================
5. RESULTS SUMMARY (for quick reference; full discussion in report)
================================================================================
 
                        Mahmud et al.   Reproduced      Reproduced
                        (2026), RF      Random Forest   XGBoost
    Accuracy            86.49%          99.25%          97.61%
    AUROC               0.940           0.988           0.894
    Recall (sepsis)     not reported    0.74            0.26
    Precision (sepsis)  not reported    0.89            0.42
 
Top SHAP features (Random Forest): qSOFA, ICULOS (ICU length of stay), NEWS.
 
================================================================================
6. KEY DEVIATIONS FROM THE ORIGINAL PAPER
================================================================================
 
1. Dataset scope: training_setA only, not the combined set A + B used in 
   related sepsis prediction literature.
2. Class imbalance handling: SMOTE was used instead of the original paper's 
   NOSE (Non-Overlapping Subset Ensemble) method, to allow a controlled 
   comparison of imbalance-handling strategy while holding all other 
   pipeline components constant.
3. Hyperparameters: Random Forest and XGBoost were trained using default 
   scikit-learn / XGBoost parameters; no cross-validation tuning was 
   performed, unlike the original study.
4. SHAP was computed on a random sample of 200 test instances rather than 
   the full test set, due to computation time.
 
================================================================================
7. CITATIONS / ACKNOWLEDGEMENTS
================================================================================
 
Core methodology (feature engineering, missing data strategy, model choice) 
reimplemented based on:
    Mahmud, F., Quamruzzaman, M., Sanka, A.I., Cheung, R.C.C., & Chowdhury, 
    M.H. (2026). Interpretable machine learning-based real-time sepsis 
    diagnosis. Scientific Reports, 16, 6702. 
    https://doi.org/10.1038/s41598-026-36945-w
 
Dataset:
    Reyna, M.A. et al. (2020). Early prediction of sepsis from clinical 
    data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical 
    Care Medicine, 48, 210. 
    https://physionet.org/content/challenge-2019/1.0.0/
================================================================================
 
