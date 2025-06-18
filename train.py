import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone

def load_data(file_path):
    """加载数据，并将除'ID'和'label'之外的所有列作为特征。"""
    print(f"加载数据: {file_path}...")
    data = pd.read_csv(file_path)
    features = data.drop(columns=['ID', 'label'])
    labels = data['label']
    return features, labels

def create_final_models():
    """创建最终的模型。"""
    models = {
        'ComplexNetwork_Logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                C=0.001,
                penalty='l2',
                solver='liblinear',
                max_iter=10000,
                random_state=42
            ))
        ]),
        'ViT_Neural_Network': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='logistic',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                batch_size='auto',
                max_iter=1000,
                random_state=42
            ))
        ]),
        'ComplexNetwork_KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ])
    }
    return models

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """计算各种性能指标，包括医学统计指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall/Sensitivity': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'PLR': (tp/(tp+fn))/(fp/(tn+fp)) if (fp/(tn+fp)) > 0 else 0,
        'NLR': (fn/(tp+fn))/(tn/(tn+fp)) if (tn/(tn+fp)) > 0 else 0,
        'DOR': (tp*tn)/(fp*fn) if (fp*fn) > 0 else 0,
        'Youden_Index': (tp/(tp+fn)) + (tn/(tn+fp)) - 1
    }
    return metrics

def train_and_evaluate(X, y, model, model_name, n_splits=5, best_seed=1329):
    """使用交叉验证训练和评估单个模型。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=best_seed)
    fold_results = []
    trained_models = []

    print(f"\n使用 {n_splits}-折交叉验证评估 {model_name}...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model_copy = clone(model)
        model_copy.fit(X_train, y_train)
        
        y_pred = model_copy.predict(X_test)
        y_pred_proba = model_copy.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        fold_results.append(metrics)
        trained_models.append(model_copy)
        
        print(f"Fold {fold} - AUC: {metrics['AUC-ROC']:.3f}")
        
    summary = {
        metric: {
            'mean': np.mean([res[metric] for res in fold_results]),
            'std': np.std([res[metric] for res in fold_results])
        } for metric in fold_results[0].keys()
    }
    
    return summary, trained_models

def evaluate_federated_model(complex_models, vit_models, X_complex, X_vit, y, model_name, n_splits=5, best_seed=1329):
    """评估联邦模型"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=best_seed)
    fold_results = []

    print(f"\n使用 {n_splits}-折交叉验证评估 {model_name}...")

    for fold, (_, test_idx) in enumerate(skf.split(X_complex, y)):
        complex_model = complex_models[fold]
        vit_model = vit_models[fold]

        X_complex_test, X_vit_test, y_test = X_complex.iloc[test_idx], X_vit.iloc[test_idx], y.iloc[test_idx]
        
        y_pred_proba_complex = complex_model.predict_proba(X_complex_test)[:, 1]
        y_pred_proba_vit = vit_model.predict_proba(X_vit_test)[:, 1]
        
        y_pred_proba_federated = (y_pred_proba_complex + y_pred_proba_vit) / 2
        y_pred_federated = (y_pred_proba_federated > 0.5).astype(int)
        
        metrics = calculate_metrics(y_test, y_pred_federated, y_pred_proba_federated)
        fold_results.append(metrics)
        print(f"Fold {fold+1} - AUC: {metrics['AUC-ROC']:.3f}")

    summary = {
        metric: {
            'mean': np.mean([res[metric] for res in fold_results]),
            'std': np.std([res[metric] for res in fold_results])
        } for metric in fold_results[0].keys()
    }
    
    return summary

def plot_roc_curves(trained_models, X, y, model_name, best_seed, n_splits, save_dir="reviewer_plots"):
    """为交叉验证绘制ROC曲线。"""
    os.makedirs(save_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=best_seed)
    
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    
    for fold, (_, test_idx) in enumerate(skf.split(X, y)):
        model, X_test, y_test = trained_models[fold], X.iloc[test_idx], y.iloc[test_idx]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fold_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.8, label=f'Fold {fold + 1} (AUC = {fold_auc:.3f})')
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold_auc)

    mean_tpr, mean_tpr[-1] = np.mean(tprs, axis=0), 1.0
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='darkblue', lw=2, linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('1-Specificity'), plt.ylabel('Sensitivity')
    plt.title(f'Cross-Validation ROC - {model_name}'), plt.legend(loc='lower right'), plt.grid(True)
    plt.savefig(f'{save_dir}/CV_{model_name}_ROC.jpg', dpi=600), plt.close()

def plot_federated_roc(complex_models, vit_models, X_complex, X_vit, y, model_name, n_splits, best_seed, save_dir="reviewer_plots"):
    """为联邦模型绘制ROC曲线"""
    os.makedirs(save_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=best_seed)
    
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    for fold, (_, test_idx) in enumerate(skf.split(X_complex, y)):
        y_pred_proba_complex = complex_models[fold].predict_proba(X_complex.iloc[test_idx])[:, 1]
        y_pred_proba_vit = vit_models[fold].predict_proba(X_vit.iloc[test_idx])[:, 1]
        y_pred_proba_federated = (y_pred_proba_complex + y_pred_proba_vit) / 2
        
        fpr, tpr, _ = roc_curve(y.iloc[test_idx], y_pred_proba_federated)

        fold_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.8, label=f'Fold {fold + 1} (AUC = {fold_auc:.3f})')

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold_auc)

    mean_tpr, mean_tpr[-1] = np.mean(tprs, axis=0), 1.0
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='darkgreen', lw=2, linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('1-Specificity'), plt.ylabel('Sensitivity')
    plt.title(f'Cross-Validation ROC - {model_name}'), plt.legend(loc='lower right'), plt.grid(True)
    plt.savefig(f'{save_dir}/CV_{model_name}_ROC.jpg', dpi=600), plt.close()

def print_summary(title, summary):
    """打印性能总结。"""
    print(f"\n--- {title} ---")
    for metric, values in summary.items():
        print(f"{metric:20s}: {values['mean']:.3f} ± {values['std']:.3f}")

def main():
    best_seed, n_splits = 1329, 5
    
    X_complex, y_complex = load_data("complex_final_features.csv")
    X_vit, y_vit = load_data("vit_final_features.csv")
    
    models = create_final_models()
    
    # 1. ComplexNetwork Logistic
    cn_log_name = 'ComplexNetwork_Logistic'
    cn_log_summary, cn_log_models = train_and_evaluate(X_complex, y_complex, models[cn_log_name], cn_log_name, n_splits, best_seed)
    print_summary(f"{cn_log_name} CV Performance", cn_log_summary)
    plot_roc_curves(cn_log_models, X_complex, y_complex, cn_log_name, best_seed, n_splits)

    # 2. ViT Neural Network
    vit_nn_name = 'ViT_Neural_Network'
    vit_nn_summary, vit_nn_models = train_and_evaluate(X_vit, y_vit, models[vit_nn_name], vit_nn_name, n_splits, best_seed)
    print_summary(f"{vit_nn_name} CV Performance", vit_nn_summary)
    plot_roc_curves(vit_nn_models, X_vit, y_vit, vit_nn_name, best_seed, n_splits)

    # 3. ComplexNetwork KNN for Federated Model
    cn_knn_name = 'ComplexNetwork_KNN'
    _, cn_knn_models = train_and_evaluate(X_complex, y_complex, models[cn_knn_name], cn_knn_name, n_splits, best_seed)

    # 4. Federated Model: ComplexNetwork(KNN) + ViT(NN)
    fed_name = 'Federated_KNN+NN'
    fed_summary = evaluate_federated_model(cn_knn_models, vit_nn_models, X_complex, X_vit, y_complex, fed_name, n_splits, best_seed)
    print_summary(f"{fed_name} CV Performance", fed_summary)
    plot_federated_roc(cn_knn_models, vit_nn_models, X_complex, X_vit, y_complex, fed_name, n_splits, best_seed)

if __name__ == "__main__":
    main() 