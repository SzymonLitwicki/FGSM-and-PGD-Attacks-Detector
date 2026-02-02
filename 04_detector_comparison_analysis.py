#!/usr/bin/env python3
"""
Comprehensive Detector Comparison Analysis

Compares performance of:
- FGSM Detector on FGSM attacks
- FGSM Detector on PGD attacks (cross-attack generalization)
- PGD Detector on PGD attacks
- PGD Detector on FGSM attacks (cross-attack generalization)
- Combined Detector on both attack types

Usage:
    python 04_detector_comparison_analysis.py

Configure paths in the CONFIG section below before running.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import os
import pickle
import argparse
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =============================================================================
# CONFIG - MODIFY THESE PATHS
# =============================================================================

CONFIG = {
    # Input directories
    "FEATURES_DIR": "./features",
    "MODELS_DIR": "./models",
    
    # Output directories
    "RESULTS_DIR": "./results",
    "PLOTS_DIR": "./plots",
    
    # Random seed
    "RANDOM_SEED": 42,
    
    # Epsilon values
    "EPSILONS": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
}

RANDOM_SEED = CONFIG['RANDOM_SEED']
np.random.seed(RANDOM_SEED)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_detector_on_data(detector_data, X_clean, X_adv, model_name='Random Forest'):
    """Evaluate a detector on given clean and adversarial data."""
    scaler = detector_data['scaler']
    models = detector_data['models']
    
    if model_name not in models:
        available = list(models.keys())
        model_name = available[0] if available else None
        if model_name is None:
            return None
    
    model = models[model_name]
    
    X = np.vstack([X_clean, X_adv])
    y = np.array([0] * len(X_clean) + [1] * len(X_adv))
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba),
        'n_clean': len(X_clean),
        'n_adversarial': len(X_adv),
        'y_true': y,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def evaluate_detector_per_epsilon(detector_data, df_clean, df_adv, feature_cols, epsilons, model_name='Random Forest'):
    """Evaluate detector performance at each epsilon level."""
    results = []
    X_clean = df_clean[feature_cols].values
    
    for eps in epsilons:
        df_adv_eps = df_adv[df_adv['epsilon'] == eps]
        if len(df_adv_eps) < 10:
            continue
        
        X_adv = df_adv_eps[feature_cols].values
        
        n_samples = min(len(X_clean), len(X_adv))
        X_clean_bal = X_clean[np.random.choice(len(X_clean), n_samples, replace=False)]
        X_adv_bal = X_adv[np.random.choice(len(X_adv), n_samples, replace=False)]
        
        eval_result = evaluate_detector_on_data(detector_data, X_clean_bal, X_adv_bal, model_name)
        
        if eval_result:
            results.append({
                'epsilon': eps,
                'accuracy': eval_result['accuracy'],
                'precision': eval_result['precision'],
                'recall': eval_result['recall'],
                'f1': eval_result['f1'],
                'roc_auc': eval_result['roc_auc']
            })
    
    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cross_attack_heatmap(df_cross, output_dir):
    """Plot cross-attack performance heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_rf = df_cross[df_cross['model'] == 'Random Forest']
    
    # F1 heatmap
    df_f1 = df_rf.pivot(index='detector', columns='attack_type', values='f1')
    ax1 = axes[0]
    sns.heatmap(df_f1, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax1,
                cbar_kws={'label': 'F1 Score'})
    ax1.set_title('F1 Score: Detector × Attack Type', fontsize=12, fontweight='bold')
    
    # AUC heatmap
    df_auc = df_rf.pivot(index='detector', columns='attack_type', values='roc_auc')
    ax2 = axes[1]
    sns.heatmap(df_auc, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax2,
                cbar_kws={'label': 'ROC-AUC'})
    ax2.set_title('ROC-AUC: Detector × Attack Type', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_cross_attack_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 12_cross_attack_heatmap.png")


def plot_per_epsilon_cross(df_per_eps, output_dir):
    """Plot per-epsilon cross-attack performance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # FGSM detector
    ax1 = axes[0]
    fgsm_native = df_per_eps[(df_per_eps['detector'] == 'FGSM') & (df_per_eps['attack_evaluated'] == 'FGSM')]
    fgsm_cross = df_per_eps[(df_per_eps['detector'] == 'FGSM') & (df_per_eps['attack_evaluated'] == 'PGD')]
    
    if len(fgsm_native) > 0:
        ax1.plot(fgsm_native['epsilon'], fgsm_native['f1'], 'o-', color='#e74c3c', 
                linewidth=2, markersize=8, label='FGSM→FGSM (native)')
    if len(fgsm_cross) > 0:
        ax1.plot(fgsm_cross['epsilon'], fgsm_cross['f1'], 's--', color='#3498db', 
                linewidth=2, markersize=8, label='FGSM→PGD (cross)')
    
    ax1.axhline(y=0.9, color='green', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Epsilon (ε)', fontsize=11)
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('FGSM Detector Generalization', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.11])
    ax1.set_ylim([0.4, 1.0])
    
    # PGD detector
    ax2 = axes[1]
    pgd_native = df_per_eps[(df_per_eps['detector'] == 'PGD') & (df_per_eps['attack_evaluated'] == 'PGD')]
    pgd_cross = df_per_eps[(df_per_eps['detector'] == 'PGD') & (df_per_eps['attack_evaluated'] == 'FGSM')]
    
    if len(pgd_native) > 0:
        ax2.plot(pgd_native['epsilon'], pgd_native['f1'], 's-', color='#3498db', 
                linewidth=2, markersize=8, label='PGD→PGD (native)')
    if len(pgd_cross) > 0:
        ax2.plot(pgd_cross['epsilon'], pgd_cross['f1'], 'o--', color='#e74c3c', 
                linewidth=2, markersize=8, label='PGD→FGSM (cross)')
    
    ax2.axhline(y=0.9, color='green', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Epsilon (ε)', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('PGD Detector Generalization', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.11])
    ax2.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '13_per_epsilon_cross_attack.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 13_per_epsilon_cross_attack.png")


def plot_combined_vs_native(df_per_eps, output_dir):
    """Plot combined detector vs native detectors."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    comb_fgsm = df_per_eps[(df_per_eps['detector'] == 'COMBINED') & (df_per_eps['attack_evaluated'] == 'FGSM')]
    comb_pgd = df_per_eps[(df_per_eps['detector'] == 'COMBINED') & (df_per_eps['attack_evaluated'] == 'PGD')]
    fgsm_native = df_per_eps[(df_per_eps['detector'] == 'FGSM') & (df_per_eps['attack_evaluated'] == 'FGSM')]
    pgd_native = df_per_eps[(df_per_eps['detector'] == 'PGD') & (df_per_eps['attack_evaluated'] == 'PGD')]
    
    if len(comb_fgsm) > 0:
        ax.plot(comb_fgsm['epsilon'], comb_fgsm['f1'], 'o-', color='#e74c3c', 
               linewidth=2, markersize=8, label='Combined→FGSM')
    if len(comb_pgd) > 0:
        ax.plot(comb_pgd['epsilon'], comb_pgd['f1'], 's-', color='#3498db', 
               linewidth=2, markersize=8, label='Combined→PGD')
    if len(fgsm_native) > 0:
        ax.plot(fgsm_native['epsilon'], fgsm_native['f1'], '^--', color='#e74c3c', 
               linewidth=1, markersize=6, alpha=0.5, label='FGSM→FGSM (native)')
    if len(pgd_native) > 0:
        ax.plot(pgd_native['epsilon'], pgd_native['f1'], 'v--', color='#3498db', 
               linewidth=1, markersize=6, alpha=0.5, label='PGD→PGD (native)')
    
    ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.5, label='90% F1')
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Combined Detector vs Native Detectors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.11])
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '14_combined_vs_native.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 14_combined_vs_native.png")


def plot_scenario_comparison(df_cross, output_dir):
    """Plot model comparison across scenarios."""
    main_scenarios = df_cross[df_cross['model'].isin(['Random Forest', 'XGBoost', 'LightGBM'])]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scenarios_order = ['FGSM', 'PGD', 'COMBINED']
    df_plot = main_scenarios.copy()
    df_plot['scenario_label'] = df_plot['detector'] + '→' + df_plot['attack_type']
    
    scenario_labels = []
    for det in scenarios_order:
        for atk in ['FGSM', 'PGD']:
            label = f'{det}→{atk}'
            if label in df_plot['scenario_label'].values:
                scenario_labels.append(label)
    
    x = np.arange(len(scenario_labels))
    width = 0.25
    
    for idx, (metric, metric_name) in enumerate([('f1', 'F1 Score'), ('roc_auc', 'ROC-AUC')]):
        ax = axes[idx]
        
        for i, model in enumerate(['Random Forest', 'XGBoost', 'LightGBM']):
            model_data = []
            for label in scenario_labels:
                row = df_plot[(df_plot['scenario_label'] == label) & (df_plot['model'] == model)]
                model_data.append(row[metric].values[0] if len(row) > 0 else 0)
            
            ax.bar(x + i * width, model_data, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Scenario (Detector→Attack)', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Scenario and Model', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenario_labels, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '15_model_scenario_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 15_model_scenario_comparison.png")


def plot_generalization_gap(df_gap, output_dir):
    """Plot generalization gap."""
    if len(df_gap) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_gap))
    width = 0.35
    
    ax.bar(x - width/2, df_gap['native_f1'], width, label='Native Attack', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, df_gap['cross_f1'], width, label='Cross Attack', color='#e74c3c', alpha=0.8)
    
    for i, row in df_gap.iterrows():
        gap = row['f1_gap']
        y_pos = max(row['native_f1'], row['cross_f1']) + 0.02
        ax.annotate(f'Gap: {gap:.3f}', (i, y_pos), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Detector', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Generalization Gap: Native vs Cross-Attack Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_gap['detector'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '16_generalization_gap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 16_generalization_gap.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Detector comparison analysis')
    parser.add_argument('--features-dir', type=str, default=CONFIG['FEATURES_DIR'],
                        help='Directory containing feature CSV files')
    parser.add_argument('--models-dir', type=str, default=CONFIG['MODELS_DIR'],
                        help='Directory containing trained models')
    parser.add_argument('--results-dir', type=str, default=CONFIG['RESULTS_DIR'],
                        help='Output directory for result CSVs')
    parser.add_argument('--plots-dir', type=str, default=CONFIG['PLOTS_DIR'],
                        help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print("=" * 70)
    print("DETECTOR COMPARISON ANALYSIS")
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
    
    # Load detectors
    print("\nLoading trained detectors...")
    detectors = {}
    for name in ['fgsm', 'pgd', 'combined']:
        path = os.path.join(args.models_dir, f'{name}_detector', 'detector_pipeline.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                detectors[name] = pickle.load(f)
            print(f"  Loaded: {name.upper()}")
        else:
            print(f"  Not found: {name.upper()}")
    
    metadata_cols = ['filename', 'class', 'image_type', 'attack_type', 'epsilon', 
                     'relative_path', 'pgd_eps_step', 'pgd_max_iter', 'is_adversarial']
    feature_cols = [col for col in df_clean.columns if col not in metadata_cols]
    
    # Prepare test sets
    n_test = min(len(df_clean), len(df_fgsm), len(df_pgd)) // 2
    
    np.random.seed(RANDOM_SEED)
    X_clean_test = df_clean.sample(n=n_test, random_state=RANDOM_SEED)[feature_cols].values
    X_fgsm_test = df_fgsm.sample(n=n_test, random_state=RANDOM_SEED)[feature_cols].values
    X_pgd_test = df_pgd.sample(n=n_test, random_state=RANDOM_SEED)[feature_cols].values
    
    # Cross-attack evaluation
    print("\n" + "=" * 70)
    print("CROSS-ATTACK EVALUATION")
    print("=" * 70)
    
    cross_attack_results = []
    model_names = ['Random Forest', 'XGBoost', 'LightGBM', 'Voting Ensemble']
    
    scenarios = [
        ('fgsm', 'FGSM', X_fgsm_test, "FGSM Detector → FGSM Attacks"),
        ('fgsm', 'PGD', X_pgd_test, "FGSM Detector → PGD Attacks (cross)"),
        ('pgd', 'PGD', X_pgd_test, "PGD Detector → PGD Attacks"),
        ('pgd', 'FGSM', X_fgsm_test, "PGD Detector → FGSM Attacks (cross)"),
        ('combined', 'FGSM', X_fgsm_test, "Combined Detector → FGSM Attacks"),
        ('combined', 'PGD', X_pgd_test, "Combined Detector → PGD Attacks"),
    ]
    
    for detector_name, attack_type, X_adv, scenario_desc in scenarios:
        print(f"\n{scenario_desc}")
        
        if detector_name not in detectors:
            print("  Detector not available")
            continue
        
        detector_data = detectors[detector_name]
        
        for model_name in model_names:
            if model_name not in detector_data['models']:
                continue
            
            result = evaluate_detector_on_data(detector_data, X_clean_test, X_adv, model_name)
            
            if result:
                cross_attack_results.append({
                    'detector': detector_name.upper(),
                    'attack_type': attack_type,
                    'model': model_name,
                    'scenario': scenario_desc,
                    'is_cross_attack': detector_name != attack_type.lower() and detector_name != 'combined',
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1': result['f1'],
                    'roc_auc': result['roc_auc']
                })
                print(f"  {model_name:<20}: F1={result['f1']:.4f}, AUC={result['roc_auc']:.4f}")
    
    df_cross_attack = pd.DataFrame(cross_attack_results)
    
    # Per-epsilon cross-attack analysis
    print("\n" + "=" * 70)
    print("PER-EPSILON CROSS-ATTACK ANALYSIS")
    print("=" * 70)
    
    per_epsilon_cross = []
    
    scenarios_eps = [
        ('fgsm', df_fgsm, 'FGSM'),
        ('fgsm', df_pgd, 'PGD'),
        ('pgd', df_pgd, 'PGD'),
        ('pgd', df_fgsm, 'FGSM'),
        ('combined', df_fgsm, 'FGSM'),
        ('combined', df_pgd, 'PGD'),
    ]
    
    for detector_name, df_adv, attack_eval in scenarios_eps:
        if detector_name not in detectors:
            continue
        
        results = evaluate_detector_per_epsilon(
            detectors[detector_name], df_clean, df_adv, feature_cols, CONFIG['EPSILONS']
        )
        
        for r in results:
            r['detector'] = detector_name.upper()
            r['attack_evaluated'] = attack_eval
            per_epsilon_cross.append(r)
    
    df_per_eps_cross = pd.DataFrame(per_epsilon_cross)
    
    # Generalization gap analysis
    print("\n" + "=" * 70)
    print("GENERALIZATION GAP ANALYSIS")
    print("=" * 70)
    
    gap_analysis = []
    
    # FGSM detector gap
    fgsm_on_fgsm = df_cross_attack[(df_cross_attack['detector'] == 'FGSM') & 
                                   (df_cross_attack['attack_type'] == 'FGSM') &
                                   (df_cross_attack['model'] == 'Random Forest')]
    fgsm_on_pgd = df_cross_attack[(df_cross_attack['detector'] == 'FGSM') & 
                                  (df_cross_attack['attack_type'] == 'PGD') &
                                  (df_cross_attack['model'] == 'Random Forest')]
    
    if len(fgsm_on_fgsm) > 0 and len(fgsm_on_pgd) > 0:
        gap = fgsm_on_fgsm.iloc[0]['f1'] - fgsm_on_pgd.iloc[0]['f1']
        gap_analysis.append({
            'detector': 'FGSM',
            'native_attack': 'FGSM',
            'cross_attack': 'PGD',
            'native_f1': fgsm_on_fgsm.iloc[0]['f1'],
            'cross_f1': fgsm_on_pgd.iloc[0]['f1'],
            'f1_gap': gap
        })
        print(f"\nFGSM Detector:")
        print(f"  Native (FGSM): F1={fgsm_on_fgsm.iloc[0]['f1']:.4f}")
        print(f"  Cross (PGD):   F1={fgsm_on_pgd.iloc[0]['f1']:.4f}")
        print(f"  Gap: {gap:.4f}")
    
    # PGD detector gap
    pgd_on_pgd = df_cross_attack[(df_cross_attack['detector'] == 'PGD') & 
                                 (df_cross_attack['attack_type'] == 'PGD') &
                                 (df_cross_attack['model'] == 'Random Forest')]
    pgd_on_fgsm = df_cross_attack[(df_cross_attack['detector'] == 'PGD') & 
                                  (df_cross_attack['attack_type'] == 'FGSM') &
                                  (df_cross_attack['model'] == 'Random Forest')]
    
    if len(pgd_on_pgd) > 0 and len(pgd_on_fgsm) > 0:
        gap = pgd_on_pgd.iloc[0]['f1'] - pgd_on_fgsm.iloc[0]['f1']
        gap_analysis.append({
            'detector': 'PGD',
            'native_attack': 'PGD',
            'cross_attack': 'FGSM',
            'native_f1': pgd_on_pgd.iloc[0]['f1'],
            'cross_f1': pgd_on_fgsm.iloc[0]['f1'],
            'f1_gap': gap
        })
        print(f"\nPGD Detector:")
        print(f"  Native (PGD):  F1={pgd_on_pgd.iloc[0]['f1']:.4f}")
        print(f"  Cross (FGSM):  F1={pgd_on_fgsm.iloc[0]['f1']:.4f}")
        print(f"  Gap: {gap:.4f}")
    
    df_gap = pd.DataFrame(gap_analysis)
    
    # Statistical significance tests
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    
    significance_results = []
    
    fgsm_native = df_per_eps_cross[(df_per_eps_cross['detector'] == 'FGSM') & 
                                    (df_per_eps_cross['attack_evaluated'] == 'FGSM')]['f1'].values
    fgsm_cross = df_per_eps_cross[(df_per_eps_cross['detector'] == 'FGSM') & 
                                   (df_per_eps_cross['attack_evaluated'] == 'PGD')]['f1'].values
    
    if len(fgsm_native) > 0 and len(fgsm_cross) > 0 and len(fgsm_native) == len(fgsm_cross):
        stat, p_value = wilcoxon(fgsm_native, fgsm_cross)
        significance_results.append({
            'comparison': 'FGSM Detector: FGSM vs PGD',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print(f"\nFGSM Detector (native vs cross):")
        print(f"  Wilcoxon p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    pgd_native = df_per_eps_cross[(df_per_eps_cross['detector'] == 'PGD') & 
                                   (df_per_eps_cross['attack_evaluated'] == 'PGD')]['f1'].values
    pgd_cross = df_per_eps_cross[(df_per_eps_cross['detector'] == 'PGD') & 
                                  (df_per_eps_cross['attack_evaluated'] == 'FGSM')]['f1'].values
    
    if len(pgd_native) > 0 and len(pgd_cross) > 0 and len(pgd_native) == len(pgd_cross):
        stat, p_value = wilcoxon(pgd_native, pgd_cross)
        significance_results.append({
            'comparison': 'PGD Detector: PGD vs FGSM',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        print(f"\nPGD Detector (native vs cross):")
        print(f"  Wilcoxon p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    df_significance = pd.DataFrame(significance_results)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_cross_attack_heatmap(df_cross_attack, args.plots_dir)
    plot_per_epsilon_cross(df_per_eps_cross, args.plots_dir)
    plot_combined_vs_native(df_per_eps_cross, args.plots_dir)
    plot_scenario_comparison(df_cross_attack, args.plots_dir)
    plot_generalization_gap(df_gap, args.plots_dir)
    
    # Save results
    print("\nSaving results...")
    df_cross_attack.to_csv(os.path.join(args.results_dir, 'cross_attack_evaluation.csv'), index=False)
    df_per_eps_cross.to_csv(os.path.join(args.results_dir, 'per_epsilon_cross_attack.csv'), index=False)
    df_gap.to_csv(os.path.join(args.results_dir, 'generalization_gap.csv'), index=False)
    df_significance.to_csv(os.path.join(args.results_dir, 'significance_tests.csv'), index=False)
    df_cross_attack.to_csv(os.path.join(args.results_dir, 'detector_comparison.csv'), index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nCross-Attack Performance (Random Forest):")
    rf_results = df_cross_attack[df_cross_attack['model'] == 'Random Forest']
    for _, row in rf_results.iterrows():
        cross = " (cross)" if row['is_cross_attack'] else ""
        print(f"  {row['detector']}→{row['attack_type']}: F1={row['f1']:.4f}{cross}")
    
    best_overall = df_cross_attack.groupby('detector')['f1'].mean().idxmax()
    best_f1 = df_cross_attack.groupby('detector')['f1'].mean().max()
    print(f"\nBest average performance: {best_overall} (mean F1={best_f1:.4f})")
    
    print("\nRecommendations:")
    print("  - For unknown attack types: Use Combined detector")
    print("  - For known FGSM attacks: Use FGSM-specific detector")
    print("  - For known PGD attacks: Use PGD-specific detector")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Plots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main()