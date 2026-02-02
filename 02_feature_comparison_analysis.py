#!/usr/bin/env python3
"""
Statistical Feature Comparison Analysis

Compares statistical features between:
- Clean images
- FGSM adversarial images
- PGD adversarial images

Includes statistical tests, effect sizes, and visualizations.

Usage:
    python 02_feature_comparison_analysis.py

Configure paths in the CONFIG section below before running.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp, skew, kurtosis
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG - MODIFY THESE PATHS
# =============================================================================

CONFIG = {
    # Input directory (where feature CSVs are located)
    "FEATURES_DIR": "./features",
    
    # Output directories
    "RESULTS_DIR": "./results",
    "PLOTS_DIR": "./plots",
    
    # Statistical significance threshold
    "ALPHA": 0.05,
    
    # Epsilon values
    "EPSILONS": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
}

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def compare_distributions(clean_data, adv_data, feature_name, alpha=0.05):
    """Compare distributions of a feature between clean and adversarial images."""
    clean_valid = clean_data[~np.isnan(clean_data)]
    adv_valid = adv_data[~np.isnan(adv_data)]
    
    if len(clean_valid) < 10 or len(adv_valid) < 10:
        return None
    
    results = {
        'feature': feature_name,
        'clean_mean': np.mean(clean_valid),
        'clean_std': np.std(clean_valid),
        'clean_median': np.median(clean_valid),
        'adv_mean': np.mean(adv_valid),
        'adv_std': np.std(adv_valid),
        'adv_median': np.median(adv_valid),
        'mean_diff': np.mean(adv_valid) - np.mean(clean_valid),
        'mean_diff_pct': ((np.mean(adv_valid) - np.mean(clean_valid)) / 
                         (np.abs(np.mean(clean_valid)) + 1e-10)) * 100
    }
    
    try:
        stat_mw, p_mw = mannwhitneyu(clean_valid, adv_valid, alternative='two-sided')
        results['mannwhitney_stat'] = stat_mw
        results['mannwhitney_p'] = p_mw
    except:
        results['mannwhitney_stat'] = np.nan
        results['mannwhitney_p'] = np.nan
    
    try:
        stat_ks, p_ks = ks_2samp(clean_valid, adv_valid)
        results['ks_stat'] = stat_ks
        results['ks_p'] = p_ks
    except:
        results['ks_stat'] = np.nan
        results['ks_p'] = np.nan
    
    results['cohens_d'] = cohens_d(adv_valid, clean_valid)
    results['effect_size'] = interpret_cohens_d(results['cohens_d'])
    results['significant'] = results['mannwhitney_p'] < alpha if not np.isnan(results['mannwhitney_p']) else False
    
    return results


def get_feature_categories(feature_cols):
    """Categorize features by type."""
    return {
        'Pixel Statistics': [f for f in feature_cols if f.startswith('pixel_')],
        'Channel Statistics': [f for f in feature_cols if f.startswith(('R_', 'G_', 'B_'))],
        'Frequency Features': [f for f in feature_cols if f.startswith(('freq_', 'low_freq', 'high_freq', 'spectral'))],
        'Gradient Features': [f for f in feature_cols if f.startswith(('gradient_', 'total_variation', 'tv_', 'edge_'))],
        'Color Features': [f for f in feature_cols if f.startswith(('corr_', 'luminance', 'saturation', 'color_'))],
        'Texture Features': [f for f in feature_cols if f.startswith(('texture_', 'local_contrast', 'homogeneity'))]
    }


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def run_comparison(df_clean, df_adv, feature_cols, comparison_name, alpha=0.05):
    """Run statistical comparison between clean and adversarial images."""
    print(f"\n{'='*70}")
    print(f"COMPARING: {comparison_name}")
    print(f"{'='*70}")
    
    results = []
    for feature in feature_cols:
        clean_data = df_clean[feature].values
        adv_data = df_adv[feature].values
        result = compare_distributions(clean_data, adv_data, feature, alpha)
        if result:
            result['comparison'] = comparison_name
            results.append(result)
    
    df_results = pd.DataFrame(results)
    df_results['cohens_d_abs'] = df_results['cohens_d'].abs()
    df_results = df_results.sort_values('cohens_d_abs', ascending=False)
    
    print(f"\nTop 15 Most Discriminative Features:")
    print(f"{'Feature':<30} {'Cohen d':<12} {'Effect Size':<12} {'p-value':<12}")
    print("-" * 66)
    
    for _, row in df_results.head(15).iterrows():
        sig = "✓" if row['significant'] else ""
        print(f"{row['feature']:<30} {row['cohens_d']:>10.4f}   {row['effect_size']:<12} {row['mannwhitney_p']:>10.2e} {sig}")
    
    return df_results


def run_per_epsilon_analysis(df_clean, df_adv, feature_cols, top_features, epsilons):
    """Analyze effect sizes per epsilon value."""
    results = []
    attack_type = df_adv['attack_type'].iloc[0]
    
    for eps in epsilons:
        df_adv_eps = df_adv[df_adv['epsilon'] == eps]
        if len(df_adv_eps) < 10:
            continue
        
        for feature in top_features:
            clean_data = df_clean[feature].values
            adv_data = df_adv_eps[feature].values
            d = cohens_d(adv_data, clean_data)
            
            results.append({
                'feature': feature,
                'epsilon': eps,
                'attack_type': attack_type,
                'cohens_d': d,
                'effect_size': interpret_cohens_d(d),
                'adv_mean': np.mean(adv_data),
                'clean_mean': np.mean(clean_data)
            })
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_feature_distributions(df_clean, df_fgsm, df_pgd, features, df_fgsm_comp, df_pgd_comp, output_dir):
    """Plot distribution comparison for top features."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features[:6]):
        ax = axes[idx]
        
        ax.hist(df_clean[feature].values, bins=50, alpha=0.5, label='Clean', color='#2ecc71', density=True)
        ax.hist(df_fgsm[feature].values, bins=50, alpha=0.5, label='FGSM', color='#e74c3c', density=True)
        ax.hist(df_pgd[feature].values, bins=50, alpha=0.5, label='PGD', color='#3498db', density=True)
        
        fgsm_d = df_fgsm_comp[df_fgsm_comp['feature'] == feature]['cohens_d'].values[0]
        pgd_d = df_pgd_comp[df_pgd_comp['feature'] == feature]['cohens_d'].values[0]
        
        ax.set_title(f'{feature}\nFGSM d={fgsm_d:.3f}, PGD d={pgd_d:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions: Clean vs FGSM vs PGD', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_feature_distributions.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 01_feature_distributions.png")


def plot_effect_size_heatmap(df_fgsm_comp, df_pgd_comp, df_fgsm_vs_pgd, feature_cols, output_dir):
    """Plot effect size heatmap."""
    effect_size_data = []
    
    for feature in feature_cols:
        fgsm_row = df_fgsm_comp[df_fgsm_comp['feature'] == feature]
        pgd_row = df_pgd_comp[df_pgd_comp['feature'] == feature]
        fgsm_pgd_row = df_fgsm_vs_pgd[df_fgsm_vs_pgd['feature'] == feature]
        
        effect_size_data.append({
            'feature': feature,
            'Clean vs FGSM': fgsm_row['cohens_d'].values[0] if len(fgsm_row) > 0 else 0,
            'Clean vs PGD': pgd_row['cohens_d'].values[0] if len(pgd_row) > 0 else 0,
            'FGSM vs PGD': fgsm_pgd_row['cohens_d'].values[0] if len(fgsm_pgd_row) > 0 else 0,
        })
    
    df_effect = pd.DataFrame(effect_size_data).set_index('feature')
    df_effect['sort_key'] = df_effect['Clean vs FGSM'].abs()
    df_effect = df_effect.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(df_effect.head(20), annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': "Cohen's d"})
    ax.set_title("Effect Sizes: Top 20 Most Discriminative Features", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_effect_size_heatmap.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 02_effect_size_heatmap.png")


def plot_effect_by_epsilon(df_per_epsilon, top_features, output_dir):
    """Plot effect size vs epsilon."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features[:6]):
        ax = axes[idx]
        feature_data = df_per_epsilon[df_per_epsilon['feature'] == feature]
        
        fgsm_data = feature_data[feature_data['attack_type'] == 'fgsm'].sort_values('epsilon')
        pgd_data = feature_data[feature_data['attack_type'] == 'pgd'].sort_values('epsilon')
        
        if len(fgsm_data) > 0:
            ax.plot(fgsm_data['epsilon'], fgsm_data['cohens_d'].abs(), 'o-', color='#e74c3c', 
                   linewidth=2, markersize=8, label='FGSM')
        if len(pgd_data) > 0:
            ax.plot(pgd_data['epsilon'], pgd_data['cohens_d'].abs(), 's-', color='#3498db', 
                   linewidth=2, markersize=8, label='PGD')
        
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
        ax.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
        ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
        
        ax.set_xlabel('Epsilon (ε)', fontsize=11)
        ax.set_ylabel("|Cohen's d|", fontsize=11)
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.11])
    
    plt.suptitle('Effect Size vs Perturbation Strength', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_effect_size_by_epsilon.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 03_effect_size_by_epsilon.png")


def plot_category_comparison(df_category_summary, output_dir):
    """Plot category-level comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(df_category_summary))
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, df_category_summary['fgsm_mean_d'], width, label='FGSM', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, df_category_summary['pgd_mean_d'], width, label='PGD', color='#3498db', alpha=0.8)
    ax1.set_xlabel('Feature Category', fontsize=12)
    ax1.set_ylabel("Mean |Cohen's d|", fontsize=12)
    ax1.set_title('Mean Effect Size by Feature Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_category_summary['category'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[1]
    ax2.bar(x - width/2, df_category_summary['fgsm_n_significant'], width, label='FGSM', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width/2, df_category_summary['pgd_n_significant'], width, label='PGD', color='#3498db', alpha=0.8)
    ax2.set_xlabel('Feature Category', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Significant Features by Category (p < 0.05)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_category_summary['category'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_category_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 04_category_comparison.png")


def plot_boxplots(df_clean, df_fgsm, df_pgd, features, output_dir):
    """Plot boxplots by attack type."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features[:8]):
        ax = axes[idx]
        data_to_plot = [df_clean[feature].values, df_fgsm[feature].values, df_pgd[feature].values]
        
        bp = ax.boxplot(data_to_plot, labels=['Clean', 'FGSM', 'PGD'], patch_artist=True)
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Value Distributions by Attack Type', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_boxplots_by_attack.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 05_boxplots_by_attack.png")


def plot_correlation_matrices(df_clean, df_fgsm, df_pgd, features, output_dir):
    """Plot correlation matrices for each image type."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    for idx, (df, title) in enumerate([(df_clean, 'Clean'), (df_fgsm, 'FGSM'), (df_pgd, 'PGD')]):
        ax = axes[idx]
        corr = df[features[:15]].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax, vmin=-1, vmax=1, square=True)
        ax.set_title(f'{title} Images', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    
    plt.suptitle('Feature Correlation Matrices by Image Type', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_correlation_matrices.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 06_correlation_matrices.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Feature comparison analysis')
    parser.add_argument('--features-dir', type=str, default=CONFIG['FEATURES_DIR'],
                        help='Directory containing feature CSV files')
    parser.add_argument('--results-dir', type=str, default=CONFIG['RESULTS_DIR'],
                        help='Output directory for result CSVs')
    parser.add_argument('--plots-dir', type=str, default=CONFIG['PLOTS_DIR'],
                        help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print("=" * 70)
    print("STATISTICAL FEATURE COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading feature CSVs...")
    df_clean = pd.read_csv(os.path.join(args.features_dir, 'features_clean.csv'))
    df_fgsm = pd.read_csv(os.path.join(args.features_dir, 'features_fgsm.csv'))
    df_pgd = pd.read_csv(os.path.join(args.features_dir, 'features_pgd.csv'))
    
    print(f"  Clean: {len(df_clean):,} images")
    print(f"  FGSM:  {len(df_fgsm):,} images")
    print(f"  PGD:   {len(df_pgd):,} images")
    
    # Identify feature columns
    metadata_cols = ['filename', 'class', 'image_type', 'attack_type', 'epsilon', 
                     'relative_path', 'pgd_eps_step', 'pgd_max_iter']
    feature_cols = [col for col in df_clean.columns if col not in metadata_cols]
    print(f"  Features: {len(feature_cols)}")
    
    # Run comparisons
    df_fgsm_comp = run_comparison(df_clean, df_fgsm, feature_cols, 'Clean vs FGSM', CONFIG['ALPHA'])
    df_pgd_comp = run_comparison(df_clean, df_pgd, feature_cols, 'Clean vs PGD', CONFIG['ALPHA'])
    
    # FGSM vs PGD comparison
    print(f"\n{'='*70}")
    print("COMPARING: FGSM vs PGD")
    print(f"{'='*70}")
    
    fgsm_vs_pgd_results = []
    for feature in feature_cols:
        result = compare_distributions(df_fgsm[feature].values, df_pgd[feature].values, feature, CONFIG['ALPHA'])
        if result:
            result['comparison'] = 'fgsm_vs_pgd'
            fgsm_vs_pgd_results.append(result)
    
    df_fgsm_vs_pgd = pd.DataFrame(fgsm_vs_pgd_results)
    df_fgsm_vs_pgd['cohens_d_abs'] = df_fgsm_vs_pgd['cohens_d'].abs()
    df_fgsm_vs_pgd = df_fgsm_vs_pgd.sort_values('cohens_d_abs', ascending=False)
    
    # Per-epsilon analysis
    top_features = df_fgsm_comp.head(10)['feature'].tolist()
    df_per_eps_fgsm = run_per_epsilon_analysis(df_clean, df_fgsm, feature_cols, top_features, CONFIG['EPSILONS'])
    df_per_eps_pgd = run_per_epsilon_analysis(df_clean, df_pgd, feature_cols, top_features, CONFIG['EPSILONS'])
    df_per_epsilon = pd.concat([df_per_eps_fgsm, df_per_eps_pgd], ignore_index=True)
    
    # Category analysis
    feature_categories = get_feature_categories(feature_cols)
    category_summary = []
    for cat_name, cat_features in feature_categories.items():
        fgsm_cat = df_fgsm_comp[df_fgsm_comp['feature'].isin(cat_features)]
        pgd_cat = df_pgd_comp[df_pgd_comp['feature'].isin(cat_features)]
        if len(fgsm_cat) > 0:
            category_summary.append({
                'category': cat_name,
                'n_features': len(cat_features),
                'fgsm_mean_d': fgsm_cat['cohens_d_abs'].mean(),
                'fgsm_max_d': fgsm_cat['cohens_d_abs'].max(),
                'fgsm_n_significant': fgsm_cat['significant'].sum(),
                'pgd_mean_d': pgd_cat['cohens_d_abs'].mean(),
                'pgd_max_d': pgd_cat['cohens_d_abs'].max(),
                'pgd_n_significant': pgd_cat['significant'].sum(),
            })
    df_category_summary = pd.DataFrame(category_summary).sort_values('fgsm_mean_d', ascending=False)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_feature_distributions(df_clean, df_fgsm, df_pgd, top_features, df_fgsm_comp, df_pgd_comp, args.plots_dir)
    plot_effect_size_heatmap(df_fgsm_comp, df_pgd_comp, df_fgsm_vs_pgd, feature_cols, args.plots_dir)
    plot_effect_by_epsilon(df_per_epsilon, top_features, args.plots_dir)
    plot_category_comparison(df_category_summary, args.plots_dir)
    plot_boxplots(df_clean, df_fgsm, df_pgd, top_features, args.plots_dir)
    plot_correlation_matrices(df_clean, df_fgsm, df_pgd, top_features, args.plots_dir)
    
    # Save results
    print("\nSaving results...")
    df_all_comparisons = pd.concat([
        df_fgsm_comp.assign(comparison_type='clean_vs_fgsm'),
        df_pgd_comp.assign(comparison_type='clean_vs_pgd')
    ])
    df_all_comparisons.to_csv(os.path.join(args.results_dir, 'feature_comparison.csv'), index=False)
    df_per_epsilon.to_csv(os.path.join(args.results_dir, 'feature_comparison_detailed.csv'), index=False)
    df_category_summary.to_csv(os.path.join(args.results_dir, 'category_summary.csv'), index=False)
    df_fgsm_vs_pgd.to_csv(os.path.join(args.results_dir, 'fgsm_vs_pgd_comparison.csv'), index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_large_fgsm = len(df_fgsm_comp[df_fgsm_comp['effect_size'] == 'large'])
    n_large_pgd = len(df_pgd_comp[df_pgd_comp['effect_size'] == 'large'])
    
    print(f"\nClean vs FGSM:")
    print(f"  Large effect size features: {n_large_fgsm}")
    print(f"  Top feature: {df_fgsm_comp.iloc[0]['feature']} (d={df_fgsm_comp.iloc[0]['cohens_d']:.4f})")
    
    print(f"\nClean vs PGD:")
    print(f"  Large effect size features: {n_large_pgd}")
    print(f"  Top feature: {df_pgd_comp.iloc[0]['feature']} (d={df_pgd_comp.iloc[0]['cohens_d']:.4f})")
    
    print(f"\nMost discriminative category: {df_category_summary.iloc[0]['category']}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Plots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main()