
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

def evaluate_baselines_subject_wise(X, y, pair_info_path):
    '''Evaluates tabular models using subject-wise CV.'''
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    try:
        pair_info = pd.read_csv(pair_info_path)
        subject_ids = pair_info['voice_subject'].values
    except Exception as e:
        print(f"Error loading pair_info: {e}"); return None

    group_kfold = GroupKFold(n_splits=5)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1, n_jobs=-1),
        'Linear SVM': SVC(kernel='linear', random_state=42, C=0.1)
    }
    baseline_results = {}
    print("Performing 5-fold subject-wise cross-validation...")
    for name, model in models.items():
        try:
            print(f"  Evaluating {name}...")
            # Use n_jobs=2 to avoid overwhelming Colab's CPU resources
            cv_scores = cross_val_score(model, X, y, cv=group_kfold, groups=subject_ids, scoring='accuracy', n_jobs=2)
            mean_acc, std_acc = cv_scores.mean(), cv_scores.std()
            baseline_results[name] = {'mean': mean_acc, 'std': std_acc}
            print(f"    {name}: {mean_acc:.4f} ± {std_acc:.4f}")
        except Exception as e:
            print(f"    Could not evaluate {name}: {e}")
    return baseline_results
