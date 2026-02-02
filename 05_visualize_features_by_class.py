#!/usr/bin/env python3
"""
Per-Class Feature Visualization

Visualizes statistical features for each image class (8 classes),
comparing clean vs FGSM vs PGD across epsilon values.

Usage:
    python 05_visualize_features_by_class.py

Configure paths and features in the CONFIG section below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings
from scipy.stats import sem

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG - MODIFY THESE SETTINGS
# =============================================================================

CONFIG = {
    # Input directory
    "FEATURES_DIR": "./features",
    
    # Output directory
    "PLOTS_DIR": "./plots",
    
    # Features to visualize (most discriminative based on typical results)
    # You can modify this list based on your feature comparison results
    "FEATURES_TO_PLOT": [
        "spectral_flatness",
        "high_freq_energy",
        "freq_energy_ratio",
        "gradient_mean",
        "total_variation",
        "pixel_std",
    ],
    
    # Alternatively, auto-select top N features by variance
    "AUTO_SELECT_FEATURES": True,
    "N_AUTO_FEATURES": 6,
    
    # Plot settings
    "FIGURE_SIZE_PER_CLASS": (18, 12),  # Size for each class figure
    "SUBPLOTS_ROWS": 2,
    "SUBPLOTS_COLS": 3,
    
    # Epsilon values
    "EPSILONS": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    
    # Show confidence intervals (standard error)
    "SHOW_CONFIDENCE": True,
    "CONFIDENCE_ALPHA": 0.2,
}

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_feature_data(features_dir):
    """Load all feature CSV files."""
    print("Loading feature data...")
    
    df_clean = pd.read_csv(os.path.join(features_dir, 'features_clean.csv'))
    df_fgsm = pd.read_csv(os.path.join(features_dir, 'features_fgsm.csv'))
    df_pgd = pd.read_csv(os.path.join(features_dir, 'features_pgd.csv'))
    
    print(f"  Clean: {len(df_clean):,} images")
    print(f"  FGSM:  {len(df_fgsm):,} images")
    print(f"  PGD:   {len(df_pgd):,} images")
    
    return df_clean, df_fgsm, df_pgd


def get_feature_columns(df):
    """Get list of feature columns (exclude metadata)."""
    metadata_cols = ['filename', 'class', 'image_type', 'attack_type', 'epsilon', 
                     'relative_path', 'pgd_eps_step', 'pgd_max_iter', 'is_adversarial']
    return [col for col in df.columns if col not in metadata_cols]


def select_discriminative_features(df_clean, df_fgsm, df_pgd, feature_cols, n_features=6):
    """Auto-select most discriminative features based on effect size."""
    from scipy.stats import mannwhitneyu
    
    print(f"\nAuto-selecting top {n_features} discriminative features...")
    
    effect_sizes = []
    
    for feature in feature_cols:
        clean_vals = df_clean[feature].dropna().values
        fgsm_vals = df_fgsm[feature].dropna().values
        pgd_vals = df_pgd[feature].dropna().values
        
        if len(clean_vals) < 10 or len(fgsm_vals) < 10:
            continue
        
        # Calculate effect size (Cohen's d approximation)
        # Using pooled data of FGSM and PGD vs clean
        adv_vals = np.concatenate([fgsm_vals, pgd_vals])
        
        pooled_std = np.sqrt((np.var(clean_vals) + np.var(adv_vals)) / 2)
        if pooled_std > 0:
            d = abs(np.mean(adv_vals) - np.mean(clean_vals)) / pooled_std
        else:
            d = 0
        
        effect_sizes.append((feature, d))
    
    # Sort by effect size and take top N
    effect_sizes.sort(key=lambda x: x[1], reverse=True)
    selected = [f[0] for f in effect_sizes[:n_features]]
    
    print("  Selected features:")
    for feat, d in effect_sizes[:n_features]:
        print(f"    {feat}: |d| = {d:.4f}")
    
    return selected


def compute_stats_by_epsilon(df, feature, class_name, epsilons):
    """Compute mean and standard error for a feature across epsilon values."""
    means = []
    sems = []
    
    for eps in epsilons:
        if 'epsilon' in df.columns:
            subset = df[(df['class'] == class_name) & (df['epsilon'] == eps)]
        else:
            subset = df[df['class'] == class_name]
        
        if len(subset) > 0:
            vals = subset[feature].dropna().values
            means.append(np.mean(vals))
            sems.append(sem(vals) if len(vals) > 1 else 0)
        else:
            means.append(np.nan)
            sems.append(np.nan)
    
    return np.array(means), np.array(sems)


def compute_clean_baseline(df_clean, feature, class_name):
    """Compute clean image baseline (mean and std error)."""
    subset = df_clean[df_clean['class'] == class_name]
    vals = subset[feature].dropna().values
    
    if len(vals) > 0:
        return np.mean(vals), sem(vals) if len(vals) > 1 else 0
    return np.nan, np.nan


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_class_features(df_clean, df_fgsm, df_pgd, class_name, features, epsilons, output_dir, config):
    """Create a multi-subplot figure for one class."""
    
    n_features = len(features)
    n_rows = config['SUBPLOTS_ROWS']
    n_cols = config['SUBPLOTS_COLS']
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=config['FIGURE_SIZE_PER_CLASS'])
    axes = axes.flatten()
    
    # Colors
    color_clean = '#2ecc71'  # Green
    color_fgsm = '#e74c3c'   # Red
    color_pgd = '#3498db'    # Blue
    
    for idx, feature in enumerate(features):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get clean baseline
        clean_mean, clean_sem = compute_clean_baseline(df_clean, feature, class_name)
        
        # Get FGSM stats by epsilon
        fgsm_means, fgsm_sems = compute_stats_by_epsilon(df_fgsm, feature, class_name, epsilons)
        
        # Get PGD stats by epsilon
        pgd_means, pgd_sems = compute_stats_by_epsilon(df_pgd, feature, class_name, epsilons)
        
        # Plot clean baseline as horizontal line
        ax.axhline(y=clean_mean, color=color_clean, linestyle='-', linewidth=2, 
                   label=f'Clean (μ={clean_mean:.4f})')
        
        if config['SHOW_CONFIDENCE'] and not np.isnan(clean_sem):
            ax.axhspan(clean_mean - clean_sem, clean_mean + clean_sem, 
                      alpha=config['CONFIDENCE_ALPHA'], color=color_clean)
        
        # Plot FGSM
        valid_fgsm = ~np.isnan(fgsm_means)
        if np.any(valid_fgsm):
            ax.plot(np.array(epsilons)[valid_fgsm], fgsm_means[valid_fgsm], 
                   'o-', color=color_fgsm, linewidth=2, markersize=6, label='FGSM')
            
            if config['SHOW_CONFIDENCE']:
                ax.fill_between(np.array(epsilons)[valid_fgsm],
                               fgsm_means[valid_fgsm] - fgsm_sems[valid_fgsm],
                               fgsm_means[valid_fgsm] + fgsm_sems[valid_fgsm],
                               alpha=config['CONFIDENCE_ALPHA'], color=color_fgsm)
        
        # Plot PGD
        valid_pgd = ~np.isnan(pgd_means)
        if np.any(valid_pgd):
            ax.plot(np.array(epsilons)[valid_pgd], pgd_means[valid_pgd], 
                   's-', color=color_pgd, linewidth=2, markersize=6, label='PGD')
            
            if config['SHOW_CONFIDENCE']:
                ax.fill_between(np.array(epsilons)[valid_pgd],
                               pgd_means[valid_pgd] - pgd_sems[valid_pgd],
                               pgd_means[valid_pgd] + pgd_sems[valid_pgd],
                               alpha=config['CONFIDENCE_ALPHA'], color=color_pgd)
        
        # Formatting
        ax.set_xlabel('Epsilon (ε)', fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.set_title(feature, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(epsilons) + 0.01])
        
        # Set x-ticks
        ax.set_xticks(epsilons)
        ax.set_xticklabels([f'{e:.2f}' for e in epsilons], rotation=45, fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    # Main title
    n_clean = len(df_clean[df_clean['class'] == class_name])
    n_fgsm = len(df_fgsm[df_fgsm['class'] == class_name])
    n_pgd = len(df_pgd[df_pgd['class'] == class_name])
    
    fig.suptitle(f'Class: {class_name.upper()}\n'
                 f'(Clean: {n_clean}, FGSM: {n_fgsm}, PGD: {n_pgd} images)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    filename = f'class_{class_name}_features.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename


def plot_summary_heatmap(df_clean, df_fgsm, df_pgd, classes, features, epsilons, output_dir):
    """Create summary heatmaps showing feature deviation from clean baseline."""
    
    print("\nGenerating summary heatmaps...")
    
    for feature in features:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Compute deviation matrices
        fgsm_deviations = np.zeros((len(classes), len(epsilons)))
        pgd_deviations = np.zeros((len(classes), len(epsilons)))
        
        for i, class_name in enumerate(classes):
            # Clean baseline
            clean_mean, _ = compute_clean_baseline(df_clean, feature, class_name)
            
            # FGSM deviations
            fgsm_means, _ = compute_stats_by_epsilon(df_fgsm, feature, class_name, epsilons)
            if not np.isnan(clean_mean) and clean_mean != 0:
                fgsm_deviations[i, :] = (fgsm_means - clean_mean) / abs(clean_mean) * 100
            
            # PGD deviations
            pgd_means, _ = compute_stats_by_epsilon(df_pgd, feature, class_name, epsilons)
            if not np.isnan(clean_mean) and clean_mean != 0:
                pgd_deviations[i, :] = (pgd_means - clean_mean) / abs(clean_mean) * 100
        
        # Plot FGSM heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(fgsm_deviations, cmap='RdBu_r', aspect='auto', 
                        vmin=-50, vmax=50)
        ax1.set_xticks(range(len(epsilons)))
        ax1.set_xticklabels([f'{e:.2f}' for e in epsilons], rotation=45)
        ax1.set_yticks(range(len(classes)))
        ax1.set_yticklabels(classes)
        ax1.set_xlabel('Epsilon (ε)')
        ax1.set_ylabel('Class')
        ax1.set_title(f'FGSM: {feature}\n(% deviation from clean)', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='% Deviation')
        
        # Plot PGD heatmap
        ax2 = axes[1]
        im2 = ax2.imshow(pgd_deviations, cmap='RdBu_r', aspect='auto',
                        vmin=-50, vmax=50)
        ax2.set_xticks(range(len(epsilons)))
        ax2.set_xticklabels([f'{e:.2f}' for e in epsilons], rotation=45)
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Epsilon (ε)')
        ax2.set_ylabel('Class')
        ax2.set_title(f'PGD: {feature}\n(% deviation from clean)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='% Deviation')
        
        plt.tight_layout()
        
        # Save
        filename = f'heatmap_{feature}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved: {filename}")


def plot_all_classes_comparison(df_clean, df_fgsm, df_pgd, classes, feature, epsilons, output_dir):
    """Create a single plot comparing all classes for one feature."""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    colors_fgsm = plt.cm.Reds(np.linspace(0.3, 0.9, len(epsilons)))
    colors_pgd = plt.cm.Blues(np.linspace(0.3, 0.9, len(epsilons)))
    
    for idx, class_name in enumerate(classes):
        ax = axes[idx]
        
        # Clean baseline
        clean_mean, clean_sem = compute_clean_baseline(df_clean, feature, class_name)
        
        # FGSM and PGD means
        fgsm_means, fgsm_sems = compute_stats_by_epsilon(df_fgsm, feature, class_name, epsilons)
        pgd_means, pgd_sems = compute_stats_by_epsilon(df_pgd, feature, class_name, epsilons)
        
        # Plot
        ax.axhline(y=clean_mean, color='#2ecc71', linestyle='-', linewidth=3, 
                   label='Clean', zorder=10)
        
        valid_fgsm = ~np.isnan(fgsm_means)
        valid_pgd = ~np.isnan(pgd_means)
        
        if np.any(valid_fgsm):
            ax.plot(np.array(epsilons)[valid_fgsm], fgsm_means[valid_fgsm], 
                   'o-', color='#e74c3c', linewidth=2, markersize=5, label='FGSM')
        
        if np.any(valid_pgd):
            ax.plot(np.array(epsilons)[valid_pgd], pgd_means[valid_pgd], 
                   's-', color='#3498db', linewidth=2, markersize=5, label='PGD')
        
        ax.set_title(class_name.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xlabel('ε', fontsize=10)
        ax.set_ylabel(feature, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(epsilons) + 0.01])
    
    fig.suptitle(f'Feature: {feature}\nComparison Across All Classes', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    filename = f'all_classes_{feature}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize features by class')
    parser.add_argument('--features-dir', type=str, default=CONFIG['FEATURES_DIR'],
                        help='Directory containing feature CSV files')
    parser.add_argument('--plots-dir', type=str, default=CONFIG['PLOTS_DIR'],
                        help='Output directory for plots')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                        help='Specific features to plot (default: auto-select)')
    parser.add_argument('--n-features', type=int, default=CONFIG['N_AUTO_FEATURES'],
                        help='Number of features to auto-select')
    args = parser.parse_args()
    
    # Create output directory
    output_subdir = os.path.join(args.plots_dir, 'per_class_features')
    os.makedirs(output_subdir, exist_ok=True)
    
    print("=" * 70)
    print("PER-CLASS FEATURE VISUALIZATION")
    print("=" * 70)
    
    # Load data
    df_clean, df_fgsm, df_pgd = load_feature_data(args.features_dir)
    
    # Get feature columns
    feature_cols = get_feature_columns(df_clean)
    print(f"\nTotal features available: {len(feature_cols)}")
    
    # Select features to plot
    if args.features:
        features_to_plot = args.features
        print(f"\nUsing specified features: {features_to_plot}")
    elif CONFIG['AUTO_SELECT_FEATURES']:
        features_to_plot = select_discriminative_features(
            df_clean, df_fgsm, df_pgd, feature_cols, args.n_features
        )
    else:
        features_to_plot = CONFIG['FEATURES_TO_PLOT']
        print(f"\nUsing predefined features: {features_to_plot}")
    
    # Get classes
    classes = sorted(df_clean['class'].unique())
    print(f"\nClasses found: {classes}")
    
    # Get epsilon values
    epsilons = sorted(df_fgsm['epsilon'].unique())
    print(f"Epsilon values: {epsilons}")
    
    # Generate per-class plots
    print("\n" + "=" * 70)
    print("GENERATING PER-CLASS PLOTS")
    print("=" * 70)
    
    for class_name in classes:
        print(f"\n  Processing: {class_name}")
        filename = plot_class_features(
            df_clean, df_fgsm, df_pgd, 
            class_name, features_to_plot, epsilons,
            output_subdir, CONFIG
        )
        print(f"    Saved: {filename}")
    
    # Generate all-classes comparison for each feature
    print("\n" + "=" * 70)
    print("GENERATING ALL-CLASSES COMPARISON PLOTS")
    print("=" * 70)
    
    for feature in features_to_plot:
        print(f"\n  Feature: {feature}")
        filename = plot_all_classes_comparison(
            df_clean, df_fgsm, df_pgd,
            classes, feature, epsilons,
            output_subdir
        )
        print(f"    Saved: {filename}")
    
    # Generate summary heatmaps
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY HEATMAPS")
    print("=" * 70)
    
    plot_summary_heatmap(
        df_clean, df_fgsm, df_pgd,
        classes, features_to_plot, epsilons,
        output_subdir
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    
    n_class_plots = len(classes)
    n_comparison_plots = len(features_to_plot)
    n_heatmaps = len(features_to_plot)
    
    print(f"\nGenerated plots:")
    print(f"  • {n_class_plots} per-class feature plots (one per class)")
    print(f"  • {n_comparison_plots} all-classes comparison plots (one per feature)")
    print(f"  • {n_heatmaps} summary heatmaps (one per feature)")
    print(f"\nTotal: {n_class_plots + n_comparison_plots + n_heatmaps} plots")
    print(f"\nSaved to: {output_subdir}")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_subdir)):
        if f.endswith('.png'):
            print(f"  • {f}")


if __name__ == "__main__":
    main()