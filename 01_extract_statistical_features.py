#!/usr/bin/env python3
"""
Statistical Feature Extraction for Adversarial Detection Research

Extracts identical statistical features from:
- Clean images (Natural Images dataset)
- FGSM adversarial images
- PGD adversarial images

Outputs: 3 CSV files with consistent feature columns

Usage:
    python 01_extract_statistical_features.py

Configure paths in the CONFIG section below before running.
"""

import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from scipy import fft, signal
from scipy.stats import skew, kurtosis
import warnings
import gc
import argparse

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG - MODIFY THESE PATHS
# =============================================================================

CONFIG = {
    # Input directories
    "CLEAN_DIR": "./data/clean_images",           # Natural Images dataset folder
    "FGSM_DIR": "./data/adversarial_fgsm",        # FGSM adversarial images
    "PGD_DIR": "./data/adversarial_pgd",          # PGD adversarial images
    
    # Output directory
    "OUTPUT_DIR": "./features",
    
    # Image settings
    "IMG_SIZE": 224,
    
    # Random seed
    "RANDOM_SEED": 42,
    
    # PGD parameters (for metadata only)
    "PGD_PARAMS": {
        0.01: {'eps_step': 0.001, 'max_iter': 10},
        0.02: {'eps_step': 0.002, 'max_iter': 10},
        0.03: {'eps_step': 0.003, 'max_iter': 15},
        0.04: {'eps_step': 0.004, 'max_iter': 15},
        0.05: {'eps_step': 0.005, 'max_iter': 20},
        0.06: {'eps_step': 0.005, 'max_iter': 20},
        0.07: {'eps_step': 0.004, 'max_iter': 30},
        0.08: {'eps_step': 0.004, 'max_iter': 30},
        0.09: {'eps_step': 0.005, 'max_iter': 40},
        0.10: {'eps_step': 0.005, 'max_iter': 40},
    }
}

# =============================================================================
# FEATURE EXTRACTOR CLASS
# =============================================================================

class StatisticalFeatureExtractor:
    """
    Comprehensive statistical feature extraction for images.
    Designed for adversarial perturbation detection.
    
    Extracts 52+ features across 5 categories:
    - Pixel statistics (11 features + per-channel)
    - Frequency domain (8 features)
    - Gradient features (13 features)
    - Color features (12 features)
    - Texture features (3 features)
    """
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        
    def extract_all_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img) / 255.0
        return self._extract_from_array(img_array)
    
    def _extract_from_array(self, img):
        features = {}
        features.update(self._extract_pixel_statistics(img))
        features.update(self._extract_frequency_features(img))
        features.update(self._extract_gradient_features(img))
        features.update(self._extract_color_features(img))
        features.update(self._extract_texture_features(img))
        return features
    
    def _extract_pixel_statistics(self, img):
        features = {}
        features['pixel_mean'] = np.mean(img)
        features['pixel_std'] = np.std(img)
        features['pixel_min'] = np.min(img)
        features['pixel_max'] = np.max(img)
        features['pixel_range'] = features['pixel_max'] - features['pixel_min']
        features['pixel_median'] = np.median(img)
        features['pixel_skewness'] = skew(img.flatten())
        features['pixel_kurtosis'] = kurtosis(img.flatten())
        features['pixel_p25'] = np.percentile(img, 25)
        features['pixel_p75'] = np.percentile(img, 75)
        features['pixel_iqr'] = features['pixel_p75'] - features['pixel_p25']
        
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = img[:, :, i]
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_min'] = np.min(channel_data)
            features[f'{channel}_max'] = np.max(channel_data)
            features[f'{channel}_skewness'] = skew(channel_data.flatten())
            features[f'{channel}_kurtosis'] = kurtosis(channel_data.flatten())
        
        return features
    
    def _extract_frequency_features(self, img):
        features = {}
        gray = np.mean(img, axis=2)
        
        f_transform = fft.fft2(gray)
        f_shift = fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        magnitude_spectrum_log = np.log1p(magnitude_spectrum)
        
        features['freq_mean'] = np.mean(magnitude_spectrum_log)
        features['freq_std'] = np.std(magnitude_spectrum_log)
        features['freq_max'] = np.max(magnitude_spectrum_log)
        features['freq_min'] = np.min(magnitude_spectrum_log)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        low_freq_region = magnitude_spectrum[
            center_h - h//8:center_h + h//8,
            center_w - w//8:center_w + w//8
        ]
        
        high_freq_mask = np.ones_like(magnitude_spectrum)
        high_freq_mask[center_h - h//8:center_h + h//8,
                       center_w - w//8:center_w + w//8] = 0
        high_freq_region = magnitude_spectrum * high_freq_mask
        
        total_energy = np.sum(magnitude_spectrum ** 2)
        low_freq_energy = np.sum(low_freq_region ** 2)
        high_freq_energy = np.sum(high_freq_region ** 2)
        
        features['low_freq_energy'] = low_freq_energy / (total_energy + 1e-10)
        features['high_freq_energy'] = high_freq_energy / (total_energy + 1e-10)
        features['freq_energy_ratio'] = high_freq_energy / (low_freq_energy + 1e-10)
        
        psd = magnitude_spectrum.flatten() ** 2
        psd_normalized = psd / (np.sum(psd) + 1e-10)
        psd_normalized = psd_normalized[psd_normalized > 0]
        features['spectral_entropy'] = -np.sum(
            psd_normalized * np.log2(psd_normalized + 1e-10)
        )
        
        geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum.flatten() + 1e-10)))
        arithmetic_mean = np.mean(magnitude_spectrum.flatten())
        features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
        
        return features
    
    def _extract_gradient_features(self, img):
        features = {}
        gray = np.mean(img, axis=2)
        
        sobel_x = signal.convolve2d(
            gray, 
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            mode='same', boundary='symm'
        )
        sobel_y = signal.convolve2d(
            gray,
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            mode='same', boundary='symm'
        )
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        features['gradient_max'] = np.max(gradient_magnitude)
        features['gradient_min'] = np.min(gradient_magnitude)
        features['gradient_median'] = np.median(gradient_magnitude)
        features['gradient_skewness'] = skew(gradient_magnitude.flatten())
        features['gradient_kurtosis'] = kurtosis(gradient_magnitude.flatten())
        
        tv_h = np.sum(np.abs(np.diff(gray, axis=0)))
        tv_w = np.sum(np.abs(np.diff(gray, axis=1)))
        features['total_variation'] = tv_h + tv_w
        features['tv_horizontal'] = tv_h
        features['tv_vertical'] = tv_w
        
        threshold = np.percentile(gradient_magnitude, 90)
        features['edge_density'] = np.mean(gradient_magnitude > threshold)
        features['gradient_dir_std'] = np.std(gradient_direction)
        
        return features
    
    def _extract_color_features(self, img):
        features = {}
        r = img[:, :, 0].flatten()
        g = img[:, :, 1].flatten()
        b = img[:, :, 2].flatten()
        
        features['corr_RG'] = np.corrcoef(r, g)[0, 1]
        features['corr_RB'] = np.corrcoef(r, b)[0, 1]
        features['corr_GB'] = np.corrcoef(g, b)[0, 1]
        
        for key in ['corr_RG', 'corr_RB', 'corr_GB']:
            if np.isnan(features[key]):
                features[key] = 0.0
        
        features['color_variance'] = np.var(img)
        features['color_entropy'] = -np.sum(
            (img.flatten() + 1e-10) * np.log2(img.flatten() + 1e-10)
        ) / len(img.flatten())
        
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        features['luminance_mean'] = np.mean(luminance)
        features['luminance_std'] = np.std(luminance)
        
        max_rgb = np.max(img, axis=2)
        min_rgb = np.min(img, axis=2)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
        features['saturation_mean'] = np.mean(saturation)
        features['saturation_std'] = np.std(saturation)
        
        features['color_moment_1'] = np.mean(img)
        features['color_moment_2'] = np.std(img)
        features['color_moment_3'] = skew(img.flatten())
        
        return features
    
    def _extract_texture_features(self, img):
        features = {}
        gray = np.mean(img, axis=2)
        
        center = gray[1:-1, 1:-1]
        top = gray[:-2, 1:-1]
        bottom = gray[2:, 1:-1]
        left = gray[1:-1, :-2]
        right = gray[1:-1, 2:]
        
        features['texture_uniformity'] = np.mean(
            (center > top).astype(int) +
            (center > bottom).astype(int) +
            (center > left).astype(int) +
            (center > right).astype(int)
        )
        
        features['local_contrast_mean'] = np.mean(
            np.abs(center - top) + np.abs(center - bottom) +
            np.abs(center - left) + np.abs(center - right)
        )
        
        features['homogeneity'] = 1.0 / (1.0 + features['local_contrast_mean'])
        
        return features
    
    def get_feature_names(self):
        dummy_img = np.random.rand(self.img_size, self.img_size, 3)
        features = self._extract_from_array(dummy_img)
        return list(features.keys())


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_clean_features(clean_dir, output_csv, img_size=224):
    """Extract features from clean images."""
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES FROM CLEAN IMAGES")
    print("=" * 70)
    
    extractor = StatisticalFeatureExtractor(img_size=img_size)
    features_list = []
    
    clean_classes = sorted([d for d in os.listdir(clean_dir) 
                           if os.path.isdir(os.path.join(clean_dir, d))])
    
    print(f"Processing {len(clean_classes)} classes...")
    
    for class_name in clean_classes:
        class_path = os.path.join(clean_dir, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"  {class_name}", leave=False):
            img_path = os.path.join(class_path, img_file)
            
            try:
                features = extractor.extract_all_features(img_path)
                features['filename'] = img_file
                features['class'] = class_name
                features['image_type'] = 'clean'
                features['epsilon'] = 0.0
                features['attack_type'] = 'none'
                features_list.append(features)
            except Exception as e:
                print(f"\n  Warning: Error processing {img_file}: {e}")
                continue
    
    df = pd.DataFrame(features_list)
    metadata_cols = ['filename', 'class', 'image_type', 'attack_type', 'epsilon']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + feature_cols]
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"  Total images: {len(df)}")
    print(f"  Classes: {df['class'].nunique()}")
    print(f"  Features: {len(feature_cols)}")
    
    gc.collect()
    return df


def extract_adversarial_features(adv_dir, output_csv, attack_type, img_size=224, pgd_params=None):
    """Extract features from adversarial images."""
    print("\n" + "=" * 70)
    print(f"EXTRACTING FEATURES FROM {attack_type.upper()} ADVERSARIAL IMAGES")
    print("=" * 70)
    
    extractor = StatisticalFeatureExtractor(img_size=img_size)
    features_list = []
    
    eps_dirs = sorted([d for d in os.listdir(adv_dir) if d.startswith('eps')])
    print(f"Processing {len(eps_dirs)} epsilon values...")
    
    for eps_dir in eps_dirs:
        eps_path = os.path.join(adv_dir, eps_dir)
        eps_value = float(eps_dir.replace('eps', '')) / 1000
        
        params_str = ""
        if attack_type == 'pgd' and pgd_params and eps_value in pgd_params:
            p = pgd_params[eps_value]
            params_str = f" (step={p['eps_step']}, iter={p['max_iter']})"
        
        print(f"\n  ε = {eps_value:.2f}{params_str}")
        
        classes = sorted([d for d in os.listdir(eps_path) 
                         if os.path.isdir(os.path.join(eps_path, d))])
        
        for class_name in classes:
            class_path = os.path.join(eps_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files, desc=f"    {class_name}", leave=False):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    features = extractor.extract_all_features(img_path)
                    features['filename'] = img_file
                    features['class'] = class_name
                    features['image_type'] = 'adversarial'
                    features['epsilon'] = eps_value
                    features['attack_type'] = attack_type
                    
                    if attack_type == 'pgd' and pgd_params and eps_value in pgd_params:
                        features['pgd_eps_step'] = pgd_params[eps_value]['eps_step']
                        features['pgd_max_iter'] = pgd_params[eps_value]['max_iter']
                    
                    features_list.append(features)
                except Exception as e:
                    print(f"\n    Warning: Error processing {img_file}: {e}")
                    continue
        
        gc.collect()
    
    df = pd.DataFrame(features_list)
    
    metadata_cols = ['filename', 'class', 'image_type', 'attack_type', 'epsilon']
    if attack_type == 'pgd':
        metadata_cols.extend(['pgd_eps_step', 'pgd_max_iter'])
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + feature_cols]
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"  Total images: {len(df)}")
    print(f"  Epsilon values: {sorted(df['epsilon'].unique())}")
    print(f"  Features: {len(feature_cols)}")
    
    gc.collect()
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract statistical features from images')
    parser.add_argument('--clean-dir', type=str, default=CONFIG['CLEAN_DIR'],
                        help='Path to clean images directory')
    parser.add_argument('--fgsm-dir', type=str, default=CONFIG['FGSM_DIR'],
                        help='Path to FGSM adversarial images directory')
    parser.add_argument('--pgd-dir', type=str, default=CONFIG['PGD_DIR'],
                        help='Path to PGD adversarial images directory')
    parser.add_argument('--output-dir', type=str, default=CONFIG['OUTPUT_DIR'],
                        help='Output directory for CSV files')
    parser.add_argument('--img-size', type=int, default=CONFIG['IMG_SIZE'],
                        help='Image size for processing')
    args = parser.parse_args()
    
    np.random.seed(CONFIG['RANDOM_SEED'])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("STATISTICAL FEATURE EXTRACTION FOR ADVERSARIAL DETECTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Clean images:  {args.clean_dir}")
    print(f"  FGSM images:   {args.fgsm_dir}")
    print(f"  PGD images:    {args.pgd_dir}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Image size:    {args.img_size}x{args.img_size}")
    
    # Verify directories exist
    for name, path in [('Clean', args.clean_dir), ('FGSM', args.fgsm_dir), ('PGD', args.pgd_dir)]:
        if not os.path.exists(path):
            print(f"\nError: {name} directory not found: {path}")
            return
    
    # Extract clean features
    clean_csv = os.path.join(args.output_dir, 'features_clean.csv')
    extract_clean_features(args.clean_dir, clean_csv, args.img_size)
    
    # Extract FGSM features
    fgsm_csv = os.path.join(args.output_dir, 'features_fgsm.csv')
    extract_adversarial_features(args.fgsm_dir, fgsm_csv, 'fgsm', args.img_size)
    
    # Extract PGD features
    pgd_csv = os.path.join(args.output_dir, 'features_pgd.csv')
    extract_adversarial_features(args.pgd_dir, pgd_csv, 'pgd', args.img_size, CONFIG['PGD_PARAMS'])
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files:")
    for csv_file in [clean_csv, fgsm_csv, pgd_csv]:
        size_mb = os.path.getsize(csv_file) / (1024 * 1024)
        print(f"  {os.path.basename(csv_file)}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()