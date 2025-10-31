
# CTR_Predictive_Model.py

# ==============================================================================
# 1) Configuration
# ==============================================================================
# NOTE: The DATA_PATH here is relative to the execution environment.
# You MUST ensure 'advertising(1).csv' is in the expected location or update the path.
DATA_PATH = r'./advertising(1).csv'  # Update this path if necessary
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = 'Clicked on Ad'  # expected target column name

print('Notebook configured. DATA_PATH =', DATA_PATH)

# ==============================================================================
# 2) Imports
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import joblib

print('Libraries imported')

# ==============================================================================
# Evaluation Function (from cell 9)
# ==============================================================================
def evaluate(model, X_test, y_test, model_name='Model'):
    """Performs evaluation, prints metrics, and displays plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    # Print results
    print(f'\n--- {model_name} ---')
    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F1 Score:', f1)
    if roc_auc is not None:
        print('ROC AUC:', roc_auc)
    print('\nClassification Report:\n', classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', aspect='auto', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xticks([0, 1], ['Pred 0', 'Pred 1'])
    plt.yticks([0, 1], ['True 0', 'True 1'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.show()

    # ROC curve plot
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}

# ==============================================================================
# Preprocessing Function (from cell 6)
# ==============================================================================
def preprocess_dataframe(df, target_col=TARGET_COL):
    """Drops non-feature columns, handles missing values, and one-hot encodes non-numeric features."""
    df_proc = df.copy()
    # Drop common non-feature columns if they exist
    for c in ['Ad Topic Line', 'City', 'Country', 'Timestamp']:
        if c in df_proc.columns:
            df_proc = df_proc.drop(columns=[c])
    
    # Drop rows with missing values (simple strategy)
    df_proc = df_proc.dropna().reset_index(drop=True)
    
    # Ensure target exists
    if target_col not in df_proc.columns:
        raise KeyError(f"Target column '{target_col}' not found. Found columns: {df_proc.columns.tolist()}")
    
    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]
    
    # Encode non-numeric columns
    non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        print('One-hot encoding columns:', non_numeric)
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)
    
    return X, y

# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == '__main__':
    
    # 3) Load dataset
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}. Please ensure it is in the same directory or update DATA_PATH."
    df = pd.read_csv(DATA_PATH)
    print('\nDataset loaded. First 5 rows:\n', df.head())
    
    # 4) Basic EDA
    print('\nShape:', df.shape)
    print('\nColumns and types:')
    print(df.dtypes)
    print('\nMissing values:')
    print(df.isnull().sum())
    
    # 5) Exploratory Analysis (Plots)
    print('\nGenerating distribution plots and correlation matrix...')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    plt.rcParams['figure.figsize'] = (8, 4)
    for col in numeric_cols:
        plt.figure()
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show() # Display plot immediately

    corr = df.select_dtypes(include=[np.number]).corr()
    print('\nCorrelation matrix:\n', corr)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', aspect='auto', cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.show() # Display plot immediately

    # 6) Preprocessing
    X, y = preprocess_dataframe(df)
    print('\nFeature matrix shape:', X.shape)
    print('Target distribution:\n', y.value_counts(normalize=True))
    print('First 5 feature rows:\n', X.head())

    # 7) Train/Test Split and Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print('\nTrain shape:', X_train.shape, 'Test shape:', X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'scaler.joblib')
    print('Scaler saved to scaler.joblib')

    # 8) Logistic Regression: Training + Hyperparameter Tuning
    print('\nStarting Logistic Regression GridSearchCV...')
    logreg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
    grid = GridSearchCV(logreg, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    best_logreg = grid.best_estimator_
    print('Best Logistic Regression params:', grid.best_params_)
    
    joblib.dump(best_logreg, 'best_logreg.joblib')
    print('Saved best_logreg.joblib')

    # 9) Evaluate Logistic Regression
    lr_metrics = evaluate(best_logreg, X_test_scaled, y_test, model_name='Logistic Regression')
    print('Logistic Regression Metrics:', lr_metrics)

    # 10) Random Forest (comparison)
    print('\nStarting Random Forest GridSearchCV...')
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    # Note: RF does not use scaled data (X_train, X_test)
    rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 6, 12], 'min_samples_split': [2, 5]}
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=4, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train) # Use unscaled data for tree-based model

    best_rf = rf_grid.best_estimator_
    print('Best Random Forest params:', rf_grid.best_params_)
    joblib.dump(best_rf, 'best_rf.joblib')
    print('Saved best_rf.joblib')

    rf_metrics = evaluate(best_rf, X_test, y_test, model_name='Random Forest') # Use unscaled data for prediction
    print('Random Forest Metrics:', rf_metrics)

    # 11) Model Selection & Save Final Model
    final_model = best_logreg
    final_metrics = lr_metrics
    
    # Check for None values before comparison
    if rf_metrics['f1'] is not None and lr_metrics['f1'] is not None:
        if rf_metrics['f1'] > lr_metrics['f1']:
            final_model = best_rf
            final_metrics = rf_metrics
            print('\nRandom Forest selected (Higher F1-score)')
        else:
            print('\nLogistic Regression selected (Higher F1-score)')
    else:
        print('\nModel selection defaulted to Logistic Regression (F1-score comparison was inconclusive).')

    joblib.dump(final_model, 'final_ctr_model.joblib')
    print('Saved final_ctr_model.joblib')

    # Write a short report
    with open('model_report.txt', 'w') as f:
        f.write('CTR Predictive Modeling Report\n')
        f.write('Selected model: ' + type(final_model).__name__ + '\n')
        for k, v in final_metrics.items():
            f.write(f'{k}: {v}\n')
    print('Saved model_report.txt')

    print('\nModel building process complete.')