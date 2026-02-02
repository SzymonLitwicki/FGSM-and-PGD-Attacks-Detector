#!/usr/bin/env python3
"""
Feature Importance Analysis: Statistical vs Model Perspectives

Compares two views of feature importance:
1. STATISTICAL PERSPECTIVE - Features discriminative based on effect sizes (Cohen's d),
   statistical tests between clean vs adversarial images
2. MODEL PERSPECTIVE - Features important for ML detectors when making decisions
   (from Random Forest, XGBoost, LightGBM feature importances)

Key questions answered:
- Do statistically discriminative features align with what models actually use?
- Which features are important statistically but ignored by models (and vice versa)?
- Do different detectors (FGSM, PGD, Combined) rely on different features?

Usage:
    python 06_compare_statistical_vs_model_importance.py

Requires: Run scripts 02 and 03 first to generate comparison and importance CSVs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, kendalltau
from matplotlib_venn import venn2, venn3
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Input directories
    "RESULTS_DIR": "./results",
    "FEATURES_DIR": "./features",
    
    # Output directory
    "PLOTS_DIR": "./plots",
    
    # Analysis settings
    "TOP_N_FEATURES": 15,  # Top N features to compare
    "CORRELATION_METHOD": "spearman",  # spearman, pearson, or kendall
    
    # Feature categories for grouped analysis
    "FEATURE_CATEGORIES": {
        'Pixel': ['pixel_mean', 'pixel_std', 'pixel_min', 'pixel_max', 'pixel_range',
                  'pixel_median', 'pixel_skewness', 'pixel_kurtosis', 'pixel_p25', 
                  'pixel_p75', 'pixel_iqr'],
        'Channel': ['R_mean', 'R_std', 'R_min', 'R_max', 'R_skewness', 'R_kurtosis',
                    'G_mean', 'G_std', 'G_min', 'G_max', 'G_skewness', 'G_kurtosis',
                    'B_mean', 'B_std', 'B_min', 'B_max', 'B_skewness', 'B_kurtosis'],
        'Frequency': ['freq_mean', 'freq_std', 'freq_max', 'freq_min', 'low_freq_energy',
                      'high_freq_energy', 'freq_energy_ratio', 'spectral_entropy', 
                      'spectral_flatness'],
        'Gradient': ['gradient_mean', 'gradient_std', 'gradient_max', 'gradient_min',
                     'gradient_median', 'gradient_skewness', 'gradient_kurtosis',
                     'total_variation', 'tv_horizontal', 'tv_vertical', 'edge_density',
                     'gradient_dir_std'],
        'Color': ['corr_RG', 'corr_RB', 'corr_GB', 'color_variance', 'color_entropy',
                  'luminance_mean', 'luminance_std', 'saturation_mean', 'saturation_std',
                  'color_moment_1', 'color_moment_2', 'color_moment_3'],
        'Texture': ['texture_uniformity', 'local_contrast_mean', 'homogeneity']
    }
}

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


# =============================================================================
# DATA LOADING
# =============================================================================

def load_statistical_importance(results_dir):
    """Load statistical comparison results (Cohen's d effect sizes)."""
    
    comparison_file = os.path.join(results_dir, 'feature_comparison.csv')
    
    if not os.path.exists(comparison_file):
        print(f"  Warning: {comparison_file} not found")
        return None
    
    df = pd.read_csv(comparison_file)
    
    # Compute absolute Cohen's d if not present
    if 'cohens_d_abs' not in df.columns:
        df['cohens_d_abs'] = df['cohens_d'].abs()
    
    return df


def load_model_importance(results_dir):
    """Load model feature importance results."""
    
    importance_file = os.path.join(results_dir, 'feature_importance.csv')
    
    if not os.path.exists(importance_file):
        print(f"  Warning: {importance_file} not found")
        return None
    
    df = pd.read_csv(importance_file)
    return df


def prepare_comparison_data(df_statistical, df_model_importance):
    """Prepare unified comparison DataFrame."""
    
    results = []
    
    # Get unique features
    all_features = set()
    
    if df_statistical is not None:
        all_features.update(df_statistical['feature'].unique())
    if df_model_importance is not None:
        all_features.update(df_model_importance['feature'].unique())
    
    for feature in all_features:
        row = {'feature': feature}
        
        # Statistical importance (Cohen's d for FGSM and PGD)
        if df_statistical is not None:
            fgsm_stat = df_statistical[(df_statistical['feature'] == feature) & 
                                       (df_statistical['comparison_type'] == 'clean_vs_fgsm')]
            pgd_stat = df_statistical[(df_statistical['feature'] == feature) & 
                                      (df_statistical['comparison_type'] == 'clean_vs_pgd')]
            
            row['stat_fgsm_d'] = fgsm_stat['cohens_d_abs'].values[0] if len(fgsm_stat) > 0 else 0
            row['stat_pgd_d'] = pgd_stat['cohens_d_abs'].values[0] if len(pgd_stat) > 0 else 0
            row['stat_mean_d'] = (row['stat_fgsm_d'] + row['stat_pgd_d']) / 2
            row['stat_max_d'] = max(row['stat_fgsm_d'], row['stat_pgd_d'])
        
        # Model importance (per detector, averaged across models)
        if df_model_importance is not None:
            for detector in ['FGSM', 'PGD', 'Combined']:
                det_data = df_model_importance[(df_model_importance['feature'] == feature) & 
                                               (df_model_importance['detector'] == detector)]
                row[f'model_{detector.lower()}_imp'] = det_data['importance'].mean() if len(det_data) > 0 else 0
            
            # Overall model importance
            feat_data = df_model_importance[df_model_importance['feature'] == feature]
            row['model_mean_imp'] = feat_data['importance'].mean() if len(feat_data) > 0 else 0
        
        # Determine category
        row['category'] = 'Other'
        for cat_name, cat_features in CONFIG['FEATURE_CATEGORIES'].items():
            if feature in cat_features:
                row['category'] = cat_name
                break
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Compute ranks
    if 'stat_mean_d' in df.columns:
        df['stat_rank'] = df['stat_mean_d'].rank(ascending=False)
    if 'model_mean_imp' in df.columns:
        df['model_rank'] = df['model_mean_imp'].rank(ascending=False)
    
    # Compute rank difference (positive = model ranks higher than stats)
    if 'stat_rank' in df.columns and 'model_rank' in df.columns:
        df['rank_diff'] = df['stat_rank'] - df['model_rank']
    
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_correlation(df, col1, col2, method='spearman'):
    """Compute correlation between two columns."""
    valid = df[[col1, col2]].dropna()
    
    if len(valid) < 3:
        return np.nan, np.nan
    
    if method == 'spearman':
        corr, p_value = spearmanr(valid[col1], valid[col2])
    elif method == 'pearson':
        corr, p_value = pearsonr(valid[col1], valid[col2])
    elif method == 'kendall':
        corr, p_value = kendalltau(valid[col1], valid[col2])
    else:
        corr, p_value = spearmanr(valid[col1], valid[col2])
    
    return corr, p_value


def find_discrepancies(df, top_n=10):
    """Find features with largest discrepancy between statistical and model importance."""
    
    if 'rank_diff' not in df.columns:
        return None, None
    
    # Features ranked high by stats but low by models
    stat_high_model_low = df.nlargest(top_n, 'rank_diff')[
        ['feature', 'stat_rank', 'model_rank', 'rank_diff', 'stat_mean_d', 'model_mean_imp', 'category']
    ]
    
    # Features ranked high by models but low by stats
    model_high_stat_low = df.nsmallest(top_n, 'rank_diff')[
        ['feature', 'stat_rank', 'model_rank', 'rank_diff', 'stat_mean_d', 'model_mean_imp', 'category']
    ]
    
    return stat_high_model_low, model_high_stat_low


def compute_overlap(df, top_n=15):
    """Compute overlap between top features from each perspective."""
    
    top_statistical = set(df.nlargest(top_n, 'stat_mean_d')['feature'])
    top_model = set(df.nlargest(top_n, 'model_mean_imp')['feature'])
    
    overlap = top_statistical & top_model
    stat_only = top_statistical - top_model
    model_only = top_model - top_statistical
    
    return {
        'overlap': overlap,
        'stat_only': stat_only,
        'model_only': model_only,
        'overlap_ratio': len(overlap) / top_n,
        'jaccard': len(overlap) / len(top_statistical | top_model)
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_scatter_comparison(df, output_dir):
    """Scatter plot: Statistical effect size vs Model importance."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Overall comparison
    ax1 = axes[0]
    scatter = ax1.scatter(df['stat_mean_d'], df['model_mean_imp'], 
                         c=df['category'].astype('category').cat.codes,
                         cmap='tab10', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Add correlation line
    if df['stat_mean_d'].notna().sum() > 2:
        z = np.polyfit(df['stat_mean_d'].dropna(), 
                      df.loc[df['stat_mean_d'].notna(), 'model_mean_imp'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['stat_mean_d'].min(), df['stat_mean_d'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r--', alpha=0.7, label='Trend')
    
    corr, p_val = compute_correlation(df, 'stat_mean_d', 'model_mean_imp', CONFIG['CORRELATION_METHOD'])
    
    ax1.set_xlabel("Statistical Effect Size (|Cohen's d|)", fontsize=11)
    ax1.set_ylabel('Model Feature Importance', fontsize=11)
    ax1.set_title(f'Overall Comparison\nρ = {corr:.3f} (p = {p_val:.2e})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Label top features
    top_features = df.nlargest(5, 'stat_mean_d')['feature'].tolist() + \
                   df.nlargest(5, 'model_mean_imp')['feature'].tolist()
    top_features = list(set(top_features))[:8]
    
    for feat in top_features:
        row = df[df['feature'] == feat].iloc[0]
        ax1.annotate(feat, (row['stat_mean_d'], row['model_mean_imp']),
                    fontsize=7, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points')
    
    # FGSM detector
    ax2 = axes[1]
    if 'stat_fgsm_d' in df.columns and 'model_fgsm_imp' in df.columns:
        ax2.scatter(df['stat_fgsm_d'], df['model_fgsm_imp'],
                   c='#e74c3c', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        corr_fgsm, p_fgsm = compute_correlation(df, 'stat_fgsm_d', 'model_fgsm_imp')
        ax2.set_title(f'FGSM Detector\nρ = {corr_fgsm:.3f} (p = {p_fgsm:.2e})', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Statistical Effect Size (FGSM)", fontsize=11)
    ax2.set_ylabel('FGSM Detector Importance', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # PGD detector
    ax3 = axes[2]
    if 'stat_pgd_d' in df.columns and 'model_pgd_imp' in df.columns:
        ax3.scatter(df['stat_pgd_d'], df['model_pgd_imp'],
                   c='#3498db', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        corr_pgd, p_pgd = compute_correlation(df, 'stat_pgd_d', 'model_pgd_imp')
        ax3.set_title(f'PGD Detector\nρ = {corr_pgd:.3f} (p = {p_pgd:.2e})', fontsize=12, fontweight='bold')
    ax3.set_xlabel("Statistical Effect Size (PGD)", fontsize=11)
    ax3.set_ylabel('PGD Detector Importance', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stat_vs_model_scatter.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: stat_vs_model_scatter.png")


def plot_rank_comparison(df, output_dir, top_n=15):
    """Side-by-side bar chart comparing top features from each perspective."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Top by statistical effect size
    ax1 = axes[0]
    top_stat = df.nlargest(top_n, 'stat_mean_d')[['feature', 'stat_mean_d', 'model_mean_imp', 'category']]
    
    y_pos = np.arange(len(top_stat))
    colors = plt.cm.Set3(top_stat['category'].astype('category').cat.codes % 12)
    
    bars1 = ax1.barh(y_pos, top_stat['stat_mean_d'], height=0.4, label='Statistical |d|', 
                     color='#3498db', alpha=0.8)
    bars2 = ax1.barh(y_pos + 0.4, top_stat['model_mean_imp'] * top_stat['stat_mean_d'].max() / top_stat['model_mean_imp'].max(), 
                     height=0.4, label='Model Importance (scaled)', color='#e74c3c', alpha=0.8)
    
    ax1.set_yticks(y_pos + 0.2)
    ax1.set_yticklabels(top_stat['feature'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_title(f'Top {top_n} by Statistical Effect Size\n(How different are clean vs adversarial?)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top by model importance
    ax2 = axes[1]
    top_model = df.nlargest(top_n, 'model_mean_imp')[['feature', 'stat_mean_d', 'model_mean_imp', 'category']]
    
    y_pos = np.arange(len(top_model))
    
    bars3 = ax2.barh(y_pos, top_model['model_mean_imp'], height=0.4, label='Model Importance', 
                     color='#e74c3c', alpha=0.8)
    bars4 = ax2.barh(y_pos + 0.4, top_model['stat_mean_d'] * top_model['model_mean_imp'].max() / top_model['stat_mean_d'].max(), 
                     height=0.4, label='Statistical |d| (scaled)', color='#3498db', alpha=0.8)
    
    ax2.set_yticks(y_pos + 0.2)
    ax2.set_yticklabels(top_model['feature'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_title(f'Top {top_n} by Model Importance\n(What do detectors actually use?)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stat_vs_model_ranked.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: stat_vs_model_ranked.png")


def plot_venn_overlap(df, output_dir, top_n=15):
    """Venn diagram showing overlap between top features."""
    
    try:
        from matplotlib_venn import venn2, venn3
        has_venn = True
    except ImportError:
        has_venn = False
        print("  Warning: matplotlib-venn not installed, skipping Venn diagram")
    
    if not has_venn:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2-way Venn: Statistical vs Model
    ax1 = axes[0]
    
    top_stat = set(df.nlargest(top_n, 'stat_mean_d')['feature'])
    top_model = set(df.nlargest(top_n, 'model_mean_imp')['feature'])
    
    venn2([top_stat, top_model], set_labels=('Statistical\n(Effect Size)', 'Model\n(Importance)'), ax=ax1)
    ax1.set_title(f'Top {top_n} Features Overlap', fontsize=12, fontweight='bold')
    
    # 3-way Venn: FGSM, PGD, Combined detectors
    ax2 = axes[1]
    
    if all(col in df.columns for col in ['model_fgsm_imp', 'model_pgd_imp', 'model_combined_imp']):
        top_fgsm = set(df.nlargest(top_n, 'model_fgsm_imp')['feature'])
        top_pgd = set(df.nlargest(top_n, 'model_pgd_imp')['feature'])
        top_combined = set(df.nlargest(top_n, 'model_combined_imp')['feature'])
        
        venn3([top_fgsm, top_pgd, top_combined], 
              set_labels=('FGSM\nDetector', 'PGD\nDetector', 'Combined\nDetector'), ax=ax2)
        ax2.set_title(f'Top {top_n} Features by Detector', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Detector-specific data\nnot available', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Detector Comparison', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stat_vs_model_venn.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: stat_vs_model_venn.png")


def plot_discrepancy_analysis(df, output_dir):
    """Highlight features with largest discrepancy between perspectives."""
    
    stat_high, model_high = find_discrepancies(df, top_n=10)
    
    if stat_high is None or model_high is None:
        print("  Skipping discrepancy plot (insufficient data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Features high in stats, low in models
    ax1 = axes[0]
    y_pos = np.arange(len(stat_high))
    
    ax1.barh(y_pos - 0.2, stat_high['stat_rank'], height=0.4, label='Statistical Rank', color='#3498db')
    ax1.barh(y_pos + 0.2, stat_high['model_rank'], height=0.4, label='Model Rank', color='#e74c3c')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stat_high['feature'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Rank (lower = more important)', fontsize=11)
    ax1.set_title('Statistically Important but\nUnderutilized by Models', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add annotations
    for i, (_, row) in enumerate(stat_high.iterrows()):
        ax1.annotate(f'Δ={row["rank_diff"]:.0f}', 
                    xy=(max(row['stat_rank'], row['model_rank']) + 1, i),
                    fontsize=8, va='center')
    
    # Features high in models, low in stats
    ax2 = axes[1]
    y_pos = np.arange(len(model_high))
    
    ax2.barh(y_pos - 0.2, model_high['stat_rank'], height=0.4, label='Statistical Rank', color='#3498db')
    ax2.barh(y_pos + 0.2, model_high['model_rank'], height=0.4, label='Model Rank', color='#e74c3c')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_high['feature'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Rank (lower = more important)', fontsize=11)
    ax2.set_title('Model-Favored but\nStatistically Subtle', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (_, row) in enumerate(model_high.iterrows()):
        ax2.annotate(f'Δ={row["rank_diff"]:.0f}', 
                    xy=(max(row['stat_rank'], row['model_rank']) + 1, i),
                    fontsize=8, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stat_vs_model_discrepancy.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: stat_vs_model_discrepancy.png")


def plot_category_comparison(df, output_dir):
    """Compare importance by feature category."""
    
    # Aggregate by category
    category_stats = df.groupby('category').agg({
        'stat_mean_d': 'mean',
        'model_mean_imp': 'mean',
        'feature': 'count'
    }).rename(columns={'feature': 'n_features'})
    
    category_stats = category_stats.sort_values('stat_mean_d', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart comparison
    ax1 = axes[0]
    y_pos = np.arange(len(category_stats))
    
    # Normalize for comparison
    stat_norm = category_stats['stat_mean_d'] / category_stats['stat_mean_d'].max()
    model_norm = category_stats['model_mean_imp'] / category_stats['model_mean_imp'].max()
    
    ax1.barh(y_pos - 0.2, stat_norm, height=0.4, label='Statistical (normalized)', color='#3498db', alpha=0.8)
    ax1.barh(y_pos + 0.2, model_norm, height=0.4, label='Model (normalized)', color='#e74c3c', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{cat}\n(n={int(category_stats.loc[cat, 'n_features'])})" 
                        for cat in category_stats.index], fontsize=10)
    ax1.set_xlabel('Normalized Importance', fontsize=11)
    ax1.set_title('Category Importance Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Scatter by category
    ax2 = axes[1]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(category_stats)))
    
    for i, cat in enumerate(category_stats.index):
        cat_data = df[df['category'] == cat]
        ax2.scatter(cat_data['stat_mean_d'], cat_data['model_mean_imp'],
                   label=cat, alpha=0.7, s=80, color=colors[i], edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel("Statistical Effect Size (|Cohen's d|)", fontsize=11)
    ax2.set_ylabel('Model Feature Importance', fontsize=11)
    ax2.set_title('Features by Category', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stat_vs_model_by_category.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: stat_vs_model_by_category.png")


def plot_detector_comparison(df, output_dir):
    """Compare feature importance across different detectors."""
    
    detector_cols = ['model_fgsm_imp', 'model_pgd_imp', 'model_combined_imp']
    if not all(col in df.columns for col in detector_cols):
        print("  Skipping detector comparison (data not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Heatmap of top features across detectors
    ax1 = axes[0, 0]
    
    # Get union of top 15 from each detector
    top_features = set()
    for col in detector_cols:
        top_features.update(df.nlargest(15, col)['feature'].tolist())
    
    top_features = list(top_features)[:20]  # Limit to 20
    
    heatmap_data = df[df['feature'].isin(top_features)][['feature'] + detector_cols + ['stat_mean_d']]
    heatmap_data = heatmap_data.set_index('feature')
    heatmap_data.columns = ['FGSM', 'PGD', 'Combined', 'Statistical']
    
    # Sort by mean importance
    heatmap_data['sort_key'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    
    # Normalize columns for visualization
    heatmap_norm = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))
    
    sns.heatmap(heatmap_norm, annot=heatmap_data.round(4), fmt='', cmap='YlOrRd', 
                ax=ax1, cbar_kws={'label': 'Normalized Importance'})
    ax1.set_title('Feature Importance Across Detectors', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature')
    
    # Correlation matrix between detectors
    ax2 = axes[0, 1]
    
    corr_data = df[detector_cols + ['stat_mean_d']].copy()
    corr_data.columns = ['FGSM', 'PGD', 'Combined', 'Statistical']
    corr_matrix = corr_data.corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                ax=ax2, vmin=-1, vmax=1, square=True)
    ax2.set_title('Correlation Between Perspectives\n(Spearman)', fontsize=12, fontweight='bold')
    
    # Scatter: FGSM vs PGD detector
    ax3 = axes[1, 0]
    ax3.scatter(df['model_fgsm_imp'], df['model_pgd_imp'], 
               alpha=0.6, s=50, c='#9b59b6', edgecolors='black', linewidth=0.5)
    
    corr, p_val = compute_correlation(df, 'model_fgsm_imp', 'model_pgd_imp')
    
    # Add diagonal line
    max_val = max(df['model_fgsm_imp'].max(), df['model_pgd_imp'].max())
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
    
    ax3.set_xlabel('FGSM Detector Importance', fontsize=11)
    ax3.set_ylabel('PGD Detector Importance', fontsize=11)
    ax3.set_title(f'FGSM vs PGD Detector\nρ = {corr:.3f}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Top features unique to each detector
    ax4 = axes[1, 1]
    
    top_n = 10
    top_fgsm = set(df.nlargest(top_n, 'model_fgsm_imp')['feature'])
    top_pgd = set(df.nlargest(top_n, 'model_pgd_imp')['feature'])
    top_combined = set(df.nlargest(top_n, 'model_combined_imp')['feature'])
    
    fgsm_unique = top_fgsm - top_pgd - top_combined
    pgd_unique = top_pgd - top_fgsm - top_combined
    combined_unique = top_combined - top_fgsm - top_pgd
    shared_all = top_fgsm & top_pgd & top_combined
    
    text = f"""Top {top_n} Features Analysis:

FGSM Detector Unique: {len(fgsm_unique)}
  {', '.join(list(fgsm_unique)[:5]) if fgsm_unique else 'None'}

PGD Detector Unique: {len(pgd_unique)}
  {', '.join(list(pgd_unique)[:5]) if pgd_unique else 'None'}

Combined Detector Unique: {len(combined_unique)}
  {', '.join(list(combined_unique)[:5]) if combined_unique else 'None'}

Shared by All Detectors: {len(shared_all)}
  {', '.join(list(shared_all)[:5]) if shared_all else 'None'}
"""
    
    ax4.text(0.1, 0.5, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Detector-Specific Features', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detector_importance_comparison.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: detector_importance_comparison.png")


def plot_comprehensive_summary(df, output_dir, top_n=20):
    """Create a comprehensive summary visualization."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Layout: 3 rows
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Main scatter and rank comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Middle row: Category analysis
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Bottom row: Summary table
    ax6 = fig.add_subplot(gs[2, :])
    
    # 1. Main scatter plot
    scatter = ax1.scatter(df['stat_mean_d'], df['model_mean_imp'],
                         c=df['category'].astype('category').cat.codes,
                         cmap='tab10', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    corr, p_val = compute_correlation(df, 'stat_mean_d', 'model_mean_imp')
    ax1.set_xlabel("|Cohen's d|", fontsize=10)
    ax1.set_ylabel('Model Importance', fontsize=10)
    ax1.set_title(f'Statistical vs Model\nρ = {corr:.3f}', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Top statistical features
    top_stat = df.nlargest(10, 'stat_mean_d')
    y_pos = np.arange(len(top_stat))
    ax2.barh(y_pos, top_stat['stat_mean_d'], color='#3498db', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_stat['feature'], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("|Cohen's d|", fontsize=10)
    ax2.set_title('Top 10 Statistical', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Top model features
    top_model = df.nlargest(10, 'model_mean_imp')
    y_pos = np.arange(len(top_model))
    ax3.barh(y_pos, top_model['model_mean_imp'], color='#e74c3c', alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_model['feature'], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('Importance', fontsize=10)
    ax3.set_title('Top 10 Model', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Category comparison
    category_stats = df.groupby('category').agg({
        'stat_mean_d': 'mean',
        'model_mean_imp': 'mean',
        'feature': 'count'
    }).rename(columns={'feature': 'n_features'})
    
    x = np.arange(len(category_stats))
    width = 0.35
    
    stat_norm = category_stats['stat_mean_d'] / category_stats['stat_mean_d'].max()
    model_norm = category_stats['model_mean_imp'] / category_stats['model_mean_imp'].max()
    
    ax4.bar(x - width/2, stat_norm, width, label='Statistical', color='#3498db', alpha=0.8)
    ax4.bar(x + width/2, model_norm, width, label='Model', color='#e74c3c', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(category_stats.index, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Normalized Importance', fontsize=10)
    ax4.set_title('Importance by Feature Category', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Overlap analysis
    overlap_info = compute_overlap(df, top_n=15)
    
    text = f"""Top 15 Features Overlap:

Statistical Only: {len(overlap_info['stat_only'])}
Model Only: {len(overlap_info['model_only'])}
Both: {len(overlap_info['overlap'])}

Overlap Ratio: {overlap_info['overlap_ratio']:.1%}
Jaccard Index: {overlap_info['jaccard']:.3f}

Shared Features:
{chr(10).join(list(overlap_info['overlap'])[:7])}
"""
    
    ax5.text(0.1, 0.5, text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax5.axis('off')
    ax5.set_title('Overlap Analysis', fontsize=11, fontweight='bold')
    
    # 6. Summary table
    summary_data = df.nlargest(top_n, 'stat_mean_d')[
        ['feature', 'category', 'stat_mean_d', 'model_mean_imp', 'stat_rank', 'model_rank']
    ].round(4)
    
    ax6.axis('off')
    table = ax6.table(cellText=summary_data.values,
                      colLabels=['Feature', 'Category', 'Stat |d|', 'Model Imp', 'Stat Rank', 'Model Rank'],
                      cellLoc='center',
                      loc='center',
                      colColours=['#f0f0f0'] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax6.set_title(f'Top {top_n} Features by Statistical Effect Size', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Statistical vs Model Feature Importance: Comprehensive Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(output_dir, 'stat_vs_model_comprehensive.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: stat_vs_model_comprehensive.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare statistical vs model feature importance')
    parser.add_argument('--results-dir', type=str, default=CONFIG['RESULTS_DIR'],
                        help='Directory containing result CSV files')
    parser.add_argument('--plots-dir', type=str, default=CONFIG['PLOTS_DIR'],
                        help='Output directory for plots')
    parser.add_argument('--top-n', type=int, default=CONFIG['TOP_N_FEATURES'],
                        help='Number of top features to compare')
    args = parser.parse_args()
    
    # Create output directory
    output_subdir = os.path.join(args.plots_dir, 'importance_comparison')
    os.makedirs(output_subdir, exist_ok=True)
    
    print("=" * 70)
    print("FEATURE IMPORTANCE COMPARISON")
    print("Statistical Perspective vs Model Perspective")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df_statistical = load_statistical_importance(args.results_dir)
    df_model = load_model_importance(args.results_dir)
    
    if df_statistical is None and df_model is None:
        print("\nError: No data files found. Run scripts 02 and 03 first.")
        return
    
    if df_statistical is not None:
        print(f"  Statistical data: {len(df_statistical)} rows")
    if df_model is not None:
        print(f"  Model importance data: {len(df_model)} rows")
    
    # Prepare comparison data
    print("\nPreparing comparison data...")
    df_comparison = prepare_comparison_data(df_statistical, df_model)
    print(f"  Features compared: {len(df_comparison)}")
    
    # Compute and display correlation
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    if 'stat_mean_d' in df_comparison.columns and 'model_mean_imp' in df_comparison.columns:
        corr, p_val = compute_correlation(df_comparison, 'stat_mean_d', 'model_mean_imp', CONFIG['CORRELATION_METHOD'])
        print(f"\nOverall correlation ({CONFIG['CORRELATION_METHOD']}):")
        print(f"  ρ = {corr:.4f} (p = {p_val:.2e})")
        
        if corr > 0.7:
            interpretation = "Strong agreement - models use statistically discriminative features"
        elif corr > 0.4:
            interpretation = "Moderate agreement - some alignment between perspectives"
        elif corr > 0.2:
            interpretation = "Weak agreement - perspectives capture different aspects"
        else:
            interpretation = "Little agreement - models find patterns beyond statistical differences"
        
        print(f"  Interpretation: {interpretation}")
    
    # Compute overlap
    print("\n" + "=" * 70)
    print("OVERLAP ANALYSIS")
    print("=" * 70)
    
    overlap = compute_overlap(df_comparison, args.top_n)
    print(f"\nTop {args.top_n} features:")
    print(f"  Statistical perspective only: {len(overlap['stat_only'])}")
    print(f"  Model perspective only: {len(overlap['model_only'])}")
    print(f"  Shared (overlap): {len(overlap['overlap'])}")
    print(f"  Overlap ratio: {overlap['overlap_ratio']:.1%}")
    print(f"  Jaccard index: {overlap['jaccard']:.3f}")
    
    if overlap['overlap']:
        print(f"\n  Shared features: {', '.join(list(overlap['overlap'])[:10])}")
    
    # Find discrepancies
    print("\n" + "=" * 70)
    print("DISCREPANCY ANALYSIS")
    print("=" * 70)
    
    stat_high, model_high = find_discrepancies(df_comparison, top_n=5)
    
    if stat_high is not None:
        print("\nHigh statistical importance, low model use:")
        for _, row in stat_high.head(5).iterrows():
            print(f"  {row['feature']}: stat_rank={row['stat_rank']:.0f}, model_rank={row['model_rank']:.0f}")
    
    if model_high is not None:
        print("\nHigh model use, low statistical importance:")
        for _, row in model_high.head(5).iterrows():
            print(f"  {row['feature']}: stat_rank={row['stat_rank']:.0f}, model_rank={row['model_rank']:.0f}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_scatter_comparison(df_comparison, output_subdir)
    plot_rank_comparison(df_comparison, output_subdir, args.top_n)
    plot_discrepancy_analysis(df_comparison, output_subdir)
    plot_category_comparison(df_comparison, output_subdir)
    plot_detector_comparison(df_comparison, output_subdir)
    plot_comprehensive_summary(df_comparison, output_subdir, args.top_n)
    
    # Try Venn diagram (requires matplotlib-venn)
    try:
        plot_venn_overlap(df_comparison, output_subdir, args.top_n)
    except Exception as e:
        print(f"  Skipping Venn diagram: {e}")
    
    # Save comparison data
    comparison_csv = os.path.join(args.results_dir, 'stat_vs_model_comparison.csv')
    df_comparison.to_csv(comparison_csv, index=False)
    print(f"\n  Saved: {comparison_csv}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nPlots saved to: {output_subdir}")
    print(f"\nKey findings:")
    
    if 'stat_mean_d' in df_comparison.columns and 'model_mean_imp' in df_comparison.columns:
        top_stat_feat = df_comparison.nlargest(1, 'stat_mean_d')['feature'].values[0]
        top_model_feat = df_comparison.nlargest(1, 'model_mean_imp')['feature'].values[0]
        
        print(f"  • Top statistical feature: {top_stat_feat}")
        print(f"  • Top model feature: {top_model_feat}")
        print(f"  • Correlation: ρ = {corr:.3f}")
        print(f"  • Top-{args.top_n} overlap: {overlap['overlap_ratio']:.1%}")


if __name__ == "__main__":
    main()