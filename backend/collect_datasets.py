"""
Dataset Collection and Preparation Script for Fake Currency Detection

This script downloads and prepares multiple datasets for training an enhanced fake currency
detection model with 15+ security features.

Datasets:
1. Indian Currency Real vs Fake Notes (Kaggle: preetrank) - ~2,048 images
2. Currency Dataset 500 INR (Kaggle: iayushanand) - ~1000 images
3. Indian Currency Detection (Kaggle: playatanu) - Multiple denominations
4. Existing training data (akash5k/fake-currency-detection)

Total expected: ~5,000+ images (balanced real/fake across multiple denominations)
"""

import os
import shutil
import argparse
from pathlib import Path
import kaggle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


# Dataset configurations
DATASETS = {
    "preetrank": {
        "name": "Indian Currency Real vs Fake Notes Dataset",
        "kaggle_id": "preetrank/indian-currency-real-vs-fake-notes-dataset",
        "description": "High-quality images across 6 denominations (₹10-₹2000), balanced real/fake",
        "expected_size": "~2,048 images",
    },
    "iayushanand": {
        "name": "Currency Dataset (500 INR note)",
        "kaggle_id": "iayushanand/currency-dataset500-inr-note-real-fake",
        "description": "Focused on ₹500 notes with augmentations",
        "expected_size": "~1,000 images",
    },
    "playatanu": {
        "name": "Indian Currency Detection",
        "kaggle_id": "playatanu/indian-currency-detection",
        "description": "Multiple denominations, real and fake samples",
        "expected_size": "~1,500 images",
    },
}


def download_dataset(dataset_key: str, download_dir: str):
    """Download a specific dataset from Kaggle."""
    dataset_config = DATASETS[dataset_key]
    kaggle_id = dataset_config["kaggle_id"]
    
    print(f"\n{'='*80}")
    print(f"Downloading: {dataset_config['name']}")
    print(f"Description: {dataset_config['description']}")
    print(f"Expected size: {dataset_config['expected_size']}")
    print(f"{'='*80}\n")
    
    # Create download directory
    dest_path = os.path.join(download_dir, dataset_key)
    os.makedirs(dest_path, exist_ok=True)
    
    try:
        # Download from Kaggle
        print(f"Downloading from Kaggle: {kaggle_id}")
        kaggle.api.dataset_download_files(kaggle_id, path=dest_path, unzip=True)
        print(f"✓ Downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {kaggle_id}: {e}")
        print("\nManual download instructions:")
        print(f"1. Go to: https://www.kaggle.com/datasets/{kaggle_id}")
        print(f"2. Download the dataset manually")
        print(f"3. Extract to: {dest_path}")
        return False


def download_all_datasets(download_dir: str):
    """Download all configured datasets."""
    print("\n" + "="*80)
    print("FAKE CURRENCY DATASET COLLECTION")
    print("="*80)
    print("\nThis script will download multiple datasets for training")
    print("an enhanced fake currency detection model.\n")
    
    for key, config in DATASETS.items():
        download_dataset(key, download_dir)
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)


def explore_dataset(dataset_key: str, download_dir: str):
    """Explore and analyze a downloaded dataset."""
    dataset_path = os.path.join(download_dir, dataset_key)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Exploring: {DATASETS[dataset_key]['name']}")
    print(f"{'='*80}\n")
    
    # Count files by category
    stats = {
        "total_files": 0,
        "images": 0,
        "real": 0,
        "fake": 0,
        "denominations": {},
        "image_sizes": [],
        "formats": {},
    }
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                stats["total_files"] += 1
                stats["images"] += 1
                
                # Track format
                ext = os.path.splitext(file)[1].lower()
                stats["formats"][ext] = stats["formats"].get(ext, 0) + 1
                
                # Try to determine category from path
                rel_path = os.path.relpath(root, dataset_path).lower()
                if 'real' in rel_path:
                    stats["real"] += 1
                elif 'fake' in rel_path:
                    stats["fake"] += 1
                
                # Try to determine denomination from path or filename
                for denom in ['10', '20', '50', '100', '200', '500', '2000']:
                    if denom in rel_path or denom in file.lower():
                        stats["denominations"][f"₹{denom}"] = stats["denominations"].get(f"₹{denom}", 0) + 1
                        break
                
                # Check image size
                try:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        stats["image_sizes"].append((w, h))
                except:
                    pass
    
    # Print statistics
    print(f"Total images: {stats['images']}")
    print(f"Real notes: {stats['real']}")
    print(f"Fake notes: {stats['fake']}")
    print(f"\nDenominations found:")
    for denom, count in sorted(stats["denominations"].items()):
        print(f"  {denom}: {count}")
    
    print(f"\nImage formats:")
    for fmt, count in stats["formats"].items():
        print(f"  {fmt}: {count}")
    
    if stats["image_sizes"]:
        widths = [s[0] for s in stats["image_sizes"]]
        heights = [s[1] for s in stats["image_sizes"]]
        print(f"\nImage sizes:")
        print(f"  Width: {min(widths)}-{max(widths)} (avg: {np.mean(widths):.0f})")
        print(f"  Height: {min(heights)}-{max(heights)} (avg: {np.mean(heights):.0f})")


def prepare_unified_dataset(
    download_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Combine all datasets into a unified training structure.
    
    Output structure:
    dataset/
    ├── train/
    │   ├── real/
    │   │   ├── 500/
    │   │   └── 2000/
    │   └── fake/
    │       ├── 500/
    │       └── 2000/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
    """
    print("\n" + "="*80)
    print("PREPARING UNIFIED DATASET")
    print("="*80 + "\n")
    
    # Create output directories
    splits = ['train', 'val', 'test']
    categories = ['real', 'fake']
    denominations = ['500', '2000']  # Focus on these for now
    
    for split in splits:
        for category in categories:
            for denom in denominations:
                dir_path = os.path.join(output_dir, split, category, denom)
                os.makedirs(dir_path, exist_ok=True)
    
    # Collect all images from downloaded datasets
    all_images = {
        'real': {'500': [], '2000': []},
        'fake': {'500': [], '2000': []}
    }
    
    for dataset_key in DATASETS.keys():
        dataset_path = os.path.join(download_dir, dataset_key)
        if not os.path.exists(dataset_path):
            continue
        
        print(f"\nProcessing: {DATASETS[dataset_key]['name']}")
        
        for root, dirs, files in os.walk(dataset_path):
            for file in tqdm(files, desc=f"  {dataset_key}"):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, dataset_path).lower()
                
                # Determine category
                category = None
                if 'real' in rel_path or 'genuine' in rel_path:
                    category = 'real'
                elif 'fake' in rel_path or 'counterfeit' in rel_path:
                    category = 'fake'
                
                if not category:
                    continue
                
                # Determine denomination
                denom = None
                for d in denominations:
                    if d in rel_path or d in file.lower():
                        denom = d
                        break
                
                if not denom:
                    # Default to 500 if can't determine
                    denom = '500'
                
                if denom in all_images[category]:
                    all_images[category][denom].append(img_path)
    
    # Print collection statistics
    print("\n" + "="*80)
    print("COLLECTION STATISTICS")
    print("="*80)
    for category in categories:
        print(f"\n{category.upper()} notes:")
        for denom in denominations:
            count = len(all_images[category][denom])
            print(f"  ₹{denom}: {count} images")
    
    # Split and copy images
    print("\n" + "="*80)
    print("SPLITTING DATASET")
    print("="*80)
    
    total_images = 0
    for category in categories:
        for denom in denominations:
            images = all_images[category][denom]
            if not images:
                continue
            
            # Split: train/val/test
            train_imgs, temp_imgs = train_test_split(
                images, train_size=train_ratio, random_state=42
            )
            val_imgs, test_imgs = train_test_split(
                temp_imgs,
                train_size=val_ratio/(val_ratio + test_ratio),
                random_state=42
            )
            
            # Copy to train directory
            for img_path in train_imgs:
                dest = os.path.join(output_dir, 'train', category, denom, os.path.basename(img_path))
                shutil.copy2(img_path, dest)
            
            # Copy to val directory
            for img_path in val_imgs:
                dest = os.path.join(output_dir, 'val', category, denom, os.path.basename(img_path))
                shutil.copy2(img_path, dest)
            
            # Copy to test directory
            for img_path in test_imgs:
                dest = os.path.join(output_dir, 'test', category, denom, os.path.basename(img_path))
                shutil.copy2(img_path, dest)
            
            total_images += len(images)
            print(f"✓ ₹{denom} {category}: {len(images)} images → "
                  f"train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}")
    
    # Save dataset info
    dataset_info = {
        "total_images": total_images,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "statistics": {
            category: {
                denom: len(all_images[category][denom])
                for denom in denominations
            }
            for category in categories
        },
        "datasets_used": list(DATASETS.keys()),
        "creation_date": "2026-04-05",
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✓ Dataset info saved to: {info_path}")
    print(f"\nUnified dataset prepared at: {output_dir}")
    print(f"Total images: {total_images}")
    
    return dataset_info


def main():
    parser = argparse.ArgumentParser(description='Fake Currency Dataset Collection and Preparation')
    parser.add_argument('command', choices=['download', 'explore', 'prepare', 'all'],
                       help='Command to execute')
    parser.add_argument('--download-dir', default='./dataset_downloads',
                       help='Directory for downloaded datasets')
    parser.add_argument('--output-dir', default='./training_data',
                       help='Output directory for unified dataset')
    parser.add_argument('--dataset', choices=list(DATASETS.keys()),
                       help='Specific dataset to download/explore')
    
    args = parser.parse_args()
    
    os.makedirs(args.download_dir, exist_ok=True)
    
    if args.command == 'download':
        if args.dataset:
            download_dataset(args.dataset, args.download_dir)
        else:
            download_all_datasets(args.download_dir)
    
    elif args.command == 'explore':
        if not args.dataset:
            print("Please specify --dataset for explore command")
            return
        explore_dataset(args.dataset, args.download_dir)
    
    elif args.command == 'prepare':
        prepare_unified_dataset(args.download_dir, args.output_dir)
    
    elif args.command == 'all':
        download_all_datasets(args.download_dir)
        for key in DATASETS.keys():
            explore_dataset(key, args.download_dir)
        prepare_unified_dataset(args.download_dir, args.output_dir)


if __name__ == "__main__":
    main()
