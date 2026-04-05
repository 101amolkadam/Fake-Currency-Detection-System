# Manual Dataset Download Instructions

Since Kaggle API requires authentication, follow these steps to download additional datasets:

## Option 1: Manual Download (Recommended)

### Dataset 1: Indian Currency Real vs Fake Notes (~2,048 images)
1. Go to: https://www.kaggle.com/datasets/preetrank/indian-currency-real-vs-fake-notes-dataset
2. Click "Download" button (requires free Kaggle account)
3. Extract ZIP to: `backend/dataset_downloads/preetrank/`
4. Run: `python collect_datasets.py prepare`

### Dataset 2: Currency Dataset 500 INR (~1,000 images)
1. Go to: https://www.kaggle.com/datasets/iayushanand/currency-dataset500-inr-note-real-fake
2. Click "Download" button
3. Extract ZIP to: `backend/dataset_downloads/iayushanand/`
4. Run: `python collect_datasets.py prepare`

### Dataset 3: Indian Currency Detection (~1,500 images)
1. Go to: https://www.kaggle.com/datasets/playatanu/indian-currency-detection
2. Click "Download" button
3. Extract ZIP to: `backend/dataset_downloads/playatanu/`
4. Run: `python collect_datasets.py prepare`

## Option 2: Setup Kaggle API (For Automation)

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Create directory: `%USERPROFILE%\.kaggle\`
6. Move `kaggle.json` to that directory
7. Run: `pip install kaggle`
8. Then: `python collect_datasets.py all`

## Option 3: Train with Current Data (Immediate)

You already have 148 images (95 real + 8 fake + 45 test). While imbalanced, you can start training now:

```bash
cd backend
python train_enhanced.py --epochs 30 --augment 20
```

With 20x augmentation, this will give you ~2,960 training samples.

## Option 4: Web Scraping Alternative

I can create a script to search for and download from alternative free sources. Would you like me to do that?
