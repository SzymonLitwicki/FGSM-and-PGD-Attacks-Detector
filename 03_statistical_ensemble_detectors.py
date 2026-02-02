#!/usr/bin/env python3
"""
Statistical Ensemble Detectors for Adversarial Attack Detection

Trains three types of detectors:
1. FGSM Detector - trained on clean vs FGSM adversarial images
2. PGD Detector - trained on clean vs PGD adversarial images
3. Combined Detector - trained on clean vs both FGSM and PGD

Includes per-epsilon training and feature importance analysis.

Usage:
    python 03_statistical_ensemble_detectors.py

Configure paths in the CONFIG section below before running.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import copy
import time
import argparse
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =============================================================================
# CONFIG - MODIFY THESE PATHS
# =============================================================================

CONFIG = {
    # Input directory
    "FEATURES_DIR": "./features",
    
    # Output directories
    "MODELS_DIR": "./models",
    "RESULTS_DIR": "./results",
    "PLOTS_DIR": "./plots",
    
    # Training parameters
    "TEST_SIZE": 0.2,
    "RANDOM_SEED": 42,
    
    # Epsilon values
    "EPSILONS": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
}

# Set random seed
RANDOM_SEED = CONFIG['RANDOM_SEED']
np.random.seed(RANDOM_SEED)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_models():
    """Returns dictionary of ML models for binary classification."""
    return {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5,
                                                min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED,
                                 n_jobs=-1, eval_metric='logloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, 
                                   random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
        'SVM (RBF)': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=RANDOM_SEED),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                        learning_rate=0.1, random_state=RANDOM_SEED),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', 
                                        max_iter=500, random_state=RANDOM_SEED)
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def prepare_paired_dataset(df_clean, df_adv, feature_cols, test_size=0.2):
    """Prepare balanced paired dataset for training."""
    n_clean = len(df_clean)
    n_adv = len(df_adv)
    
    if n_adv > n_clean:
        df_adv_balanced = df_adv.sample(n=n_clean, random_state=RANDOM_SEED)
        df_clean_balanced = df_clean.copy()
    else:
        df_adv_balanced = df_adv.copy()
        df_clean_balanced = df_clean.sample(n=n_adv, random_state=RANDOM_SEED)
    
    df_combined = pd.concat([df_clean_balanced, df_adv_balanced], ignore_index=True)
    
    X = df_combined[feature_cols].values
    y = df_combined['is_adversarial'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model."""
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'train_time': train_time
    }


def train_detector(df_clean, df_adv, feature_cols, detector_name, save_dir):
    """Train complete detector with all models."""
    print(f"\n{'='*70}")
    print(f"TRAINING {detector_name} DETECTOR")
    print(f"{'='*70}")
    
    X_train, X_test, y_train, y_test, scaler = prepare_paired_dataset(
        df_clean, df_adv, feature_cols, CONFIG['TEST_SIZE']
    )
    
    print(f"\nDataset: {len(y_train)} train, {len(y_test)} test")
    print(f"  Clean: {sum(y_train==0)} train, {sum(y_test==0)} test")
    print(f"  Adversarial: {sum(y_train==1)} train, {sum(y_test==1)} test")
    
    trained_models = {}
    results = []
    
    for model_name, model_template in get_models().items():
        print(f"\n  Training: {model_name}...", end=" ")
        model = copy.deepcopy(model_template)
        
        try:
            model_results = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
            results.append(model_results)
            trained_models[model_name] = model
            print(f"F1={model_results['f1']:.4f}, AUC={model_results['roc_auc']:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    df_results = pd.DataFrame(results).sort_values('f1', ascending=False)
    best_model_name = df_results.iloc[0]['model_name']
    
    print(f"\nBest model: {best_model_name}")
    print(f"  F1: {df_results.iloc[0]['f1']:.4f}, AUC: {df_results.iloc[0]['roc_auc']:.4f}")
    
    # Create voting ensemble
    estimators = [(name, model) for name, model in trained_models.items() 
                  if name in ['Random Forest', 'XGBoost', 'LightGBM']]
    
    if len(estimators) >= 2:
        voting_ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voting_ensemble.fit(X_train, y_train)
        
        y_pred = voting_ensemble.predict(X_test)
        y_proba = voting_ensemble.predict_proba(X_test)[:, 1]
        
        voting_results = {
            'model_name': 'Voting Ensemble',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'train_time': 0.0
        }
        
        df_results = pd.concat([df_results, pd.DataFrame([voting_results])], ignore_index=True)
        trained_models['Voting Ensemble'] = voting_ensemble
        print(f"\nVoting Ensemble: F1={voting_results['f1']:.4f}, AUC={voting_results['roc_auc']:.4f}")
    
    # Save detector
    os.makedirs(save_dir, exist_ok=True)
    
    detector_data = {
        'models': trained_models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'results': df_results,
        'best_model_name': best_model_name,
        'X_test': X_test,
        'y_test': y_test
    }
    
    with open(os.path.join(save_dir, 'detector_pipeline.pkl'), 'wb') as f:
        pickle.dump(detector_data, f)
    
    df_results.to_csv(os.path.join(save_dir, 'model_results.csv'), index=False)
    print(f"\nSaved to: {save_dir}")
    
    return detector_data


def train_per_epsilon_models(df_clean, df_adv, feature_cols, epsilons, attack_type):
    """Train models for each epsilon value."""
    print(f"\n{'='*70}")
    print(f"PER-EPSILON TRAINING ({attack_type.upper()})")
    print(f"{'='*70}")
    
    results = []
    
    for eps in epsilons:
        df_adv_eps = df_adv[df_adv['epsilon'] == eps]
        
        if len(df_adv_eps) < 50:
            print(f"\n  ε={eps:.2f}: Insufficient samples ({len(df_adv_eps)}), skipping")
            continue
        
        print(f"\n  ε={eps:.2f}:")
        
        X_train, X_test, y_train, y_test, scaler = prepare_paired_dataset(
            df_clean, df_adv_eps, feature_cols, CONFIG['TEST_SIZE']
        )
        
        for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
            model = copy.deepcopy(get_models()[model_name])
            
            try:
                model_results = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
                model_results['epsilon'] = eps
                model_results['attack_type'] = attack_type
                results.append(model_results)
                print(f"    {model_name}: F1={model_results['f1']:.4f}")
            except Exception as e:
                print(f"    {model_name}: Error - {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# FEATURE IMPORTANCE FUNCTIONS
# =============================================================================

def extract_feature_importance(detectors, feature_cols):
    """Extract feature importance from tree-based models."""
    results = []
    
    for detector_name, detector_data in detectors.items():
        models = detector_data['models']
        
        for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
            if model_name in models:
                importance = models[model_name].feature_importances_
                for feat, imp in zip(feature_cols, importance):
                    results.append({
                        'detector': detector_name,
                        'model': model_name,
                        'feature': feat,
                        'importance': imp
                    })
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_feature_importance(df_importance, output_dir):
    """Plot feature importance by detector."""
    df_mean = df_importance.groupby(['detector', 'feature'])['importance'].mean().reset_index()
    df_mean = df_mean.sort_values(['detector', 'importance'], ascending=[True, False])
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    for idx, detector_name in enumerate(['FGSM', 'PGD', 'Combined']):
        ax = axes[idx]
        data = df_mean[df_mean['detector'] == detector_name].tail(15)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
        ax.barh(data['feature'], data['importance'], color=colors)
        ax.set_xlabel('Mean Importance', fontsize=11)
        ax.set_title(f'{detector_name} Detector\nTop 15 Features', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_feature_importance_by_detector.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 07_feature_importance_by_detector.png")


def plot_importance_heatmap(df_importance, output_dir):
    """Plot feature importance heatmap."""
    df_mean = df_importance.groupby(['detector', 'feature'])['importance'].mean().reset_index()
    df_pivot = df_mean.pivot(index='feature', columns='detector', values='importance')
    df_pivot['sort_key'] = df_pivot.mean(axis=1)
    df_pivot = df_pivot.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(df_pivot.head(20), annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Importance'})
    ax.set_title('Feature Importance Across Detectors (Top 20)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_feature_importance_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_feature_importance_heatmap.png")


def plot_model_comparison(detectors, output_dir):
    """Plot model comparison across detectors."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['f1', 'roc_auc', 'accuracy']
    metric_names = ['F1 Score', 'ROC-AUC', 'Accuracy']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        comparison_data = []
        for detector_name, detector_data in detectors.items():
            for _, row in detector_data['results'].iterrows():
                if row['model_name'] != 'Voting Ensemble':
                    comparison_data.append({
                        'Detector': detector_name,
                        'Model': row['model_name'],
                        metric_name: row[metric]
                    })
        
        df_comp = pd.DataFrame(comparison_data)
        df_pivot = df_comp.pivot(index='Model', columns='Detector', values=metric_name)
        df_pivot.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Model and Detector', fontsize=12, fontweight='bold')
        ax.legend(title='Detector', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 09_model_comparison.png")


def plot_roc_curves(detectors, output_dir):
    """Plot ROC curves for best models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (detector_name, detector_data) in enumerate(detectors.items()):
        ax = axes[idx]
        
        X_test = detector_data['X_test']
        y_test = detector_data['y_test']
        models = detector_data['models']
        
        for model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Voting Ensemble']:
            if model_name in models:
                model = models[model_name]
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{detector_name} Detector\nROC Curves', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 10_roc_curves.png")


def plot_per_epsilon_f1(df_per_eps, output_dir):
    """Plot F1 score by epsilon."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_f1 = df_per_eps.groupby(['epsilon', 'attack_type'])['f1'].mean().reset_index()
    
    for attack_type, color, marker in [('fgsm', '#e74c3c', 'o'), ('pgd', '#3498db', 's')]:
        data = df_f1[df_f1['attack_type'] == attack_type]
        ax.plot(data['epsilon'], data['f1'], f'{marker}-', color=color, 
               linewidth=2, markersize=10, label=attack_type.upper())
    
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% F1')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% F1')
    
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Detection Performance vs Perturbation Strength', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.11])
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_f1_by_epsilon.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 11_f1_by_epsilon.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train statistical ensemble detectors')
    parser.add_argument('--features-dir', type=str, default=CONFIG['FEATURES_DIR'],
                        help='Directory containing feature CSV files')
    parser.add_argument('--models-dir', type=str, default=CONFIG['MODELS_DIR'],
                        help='Output directory for trained models')
    parser.add_argument('--results-dir', type=str, default=CONFIG['RESULTS_DIR'],
                        help='Output directory for result CSVs')
    parser.add_argument('--plots-dir', type=str, default=CONFIG['PLOTS_DIR'],
                        help='Output directory for plots')
    args = parser.parse_args()
    
    for d in [args.models_dir, args.results_dir, args.plots_dir]:
        os.makedirs(d, exist_ok=True)
    
    print("=" * 70)
    print("STATISTICAL ENSEMBLE DETECTORS")
    print("=" * 70)
    
    # Load data
    print("\nLoading feature CSVs...")
    df_clean = pd.read_csv(os.path.join(args.features_dir, 'features_clean.csv'))
    df_fgsm = pd.read_csv(os.path.join(args.features_dir, 'features_fgsm.csv'))
    df_pgd = pd.read_csv(os.path.join(args.features_dir, 'features_pgd.csv'))
    
    df_clean['is_adversarial'] = 0
    df_fgsm['is_adversarial'] = 1
    df_pgd['is_adversarial'] = 1
    
    print(f"  Clean: {len(df_clean):,}")
    print(f"  FGSM:  {len(df_fgsm):,}")
    print(f"  PGD:   {len(df_pgd):,}")
    
    metadata_cols = ['filename', 'class', 'image_type', 'attack_type', 'epsilon', 
                     'relative_path', 'pgd_eps_step', 'pgd_max_iter', 'is_adversarial']
    feature_cols = [col for col in df_clean.columns if col not in metadata_cols]
    print(f"  Features: {len(feature_cols)}")
    
    # Train detectors
    fgsm_detector = train_detector(df_clean, df_fgsm, feature_cols, 'FGSM',
                                   os.path.join(args.models_dir, 'fgsm_detector'))
    
    pgd_detector = train_detector(df_clean, df_pgd, feature_cols, 'PGD',
                                  os.path.join(args.models_dir, 'pgd_detector'))
    
    df_combined_adv = pd.concat([df_fgsm, df_pgd], ignore_index=True)
    combined_detector = train_detector(df_clean, df_combined_adv, feature_cols, 'Combined',
                                       os.path.join(args.models_dir, 'combined_detector'))
    
    detectors = {'FGSM': fgsm_detector, 'PGD': pgd_detector, 'Combined': combined_detector}
    
    # Per-epsilon training
    df_fgsm_per_eps = train_per_epsilon_models(df_clean, df_fgsm, feature_cols, CONFIG['EPSILONS'], 'fgsm')
    df_pgd_per_eps = train_per_epsilon_models(df_clean, df_pgd, feature_cols, CONFIG['EPSILONS'], 'pgd')
    df_per_eps_all = pd.concat([df_fgsm_per_eps, df_pgd_per_eps], ignore_index=True)
    
    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    df_importance = extract_feature_importance(detectors, feature_cols)
    df_importance.to_csv(os.path.join(args.results_dir, 'feature_importance.csv'), index=False)
    
    df_imp_mean = df_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print("\nTop 10 Most Important Features (overall):")
    for feat, imp in df_imp_mean.head(10).items():
        print(f"  {feat:<30}: {imp:.4f}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_feature_importance(df_importance, args.plots_dir)
    plot_importance_heatmap(df_importance, args.plots_dir)
    plot_model_comparison(detectors, args.plots_dir)
    plot_roc_curves(detectors, args.plots_dir)
    plot_per_epsilon_f1(df_per_eps_all, args.plots_dir)
    
    # Save results
    print("\nSaving results...")
    all_results = []
    for detector_name, detector_data in detectors.items():
        df_results = detector_data['results'].copy()
        df_results['detector'] = detector_name
        all_results.append(df_results)
    
    df_all_results = pd.concat(all_results, ignore_index=True)
    df_all_results.to_csv(os.path.join(args.results_dir, 'all_detector_results.csv'), index=False)
    df_per_eps_all.to_csv(os.path.join(args.results_dir, 'per_epsilon_results.csv'), index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    print("\nBest models per detector:")
    for detector_name, detector_data in detectors.items():
        best = detector_data['results'].iloc[0]
        print(f"  {detector_name}: {best['model_name']} (F1={best['f1']:.4f})")
    
    print(f"\nModels saved to: {args.models_dir}")
    print(f"Results saved to: {args.results_dir}")
    print(f"Plots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main()