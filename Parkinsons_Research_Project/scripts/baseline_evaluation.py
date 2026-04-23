
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd # Needed for reading pair_info

def evaluate_baselines_subject_wise(X, y, pair_info_path):
    '''Evaluates models using proper subject-wise splitting.'''
    try:
        pair_info = pd.read_csv(pair_info_path)
        voice_subject_ids = pair_info['voice_subject'].values
    except Exception as e:
        print(f"Error loading pair_info: {e}")
        return None
        
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
            cv_scores = cross_val_score(model, X, y, cv=group_kfold, groups=voice_subject_ids, scoring='accuracy', n_jobs=-1)
            mean_acc = cv_scores.mean()
            std_acc = cv_scores.std()
            baseline_results[name] = {'mean': mean_acc, 'std': std_acc}
            print(f"    {name}: {mean_acc:.4f} ± {std_acc:.4f}")
        except Exception as e:
            print(f"    Could not evaluate {name}: {e}")
            
    return baseline_results
