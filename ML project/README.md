# Tourism Destination Segmentation & Visitor Prediction

This project applies machine learning to:
1) segment tourism destinations (unsupervised learning), and
2) predict hotel booking cancellations (supervised learning).

## Overview
- **Unsupervised**: K-Means clustering on global tourism economy data to group countries with similar tourism profiles.
- **Supervised**: Hotel booking cancellation classification using Logistic Regression, Decision Tree, Random Forest, and Linear SVM, then selecting the best model.

## Folder Structure
- `clustering.ipynb` - destination segmentation notebook (K-Means + PCA).
- `supervised_model.ipynb` - hotel booking cancellation prediction notebook.
- `world_tourism_economy_data.csv` - raw global tourism dataset.
- `world_tourism_economy_data_processed.csv` - preprocessed dataset for clustering.
- `hotel_bookings.csv` - hotel booking dataset.

## Datasets
1) **World Tourism Economy Data**
   - Contains metrics such as tourism_receipts, tourism_arrivals, tourism_exports,
     tourism_departures, tourism_expenditures, gdp, inflation, unemployment, and country.
   - Used for destination segmentation.
   - Source (Kaggle):
     ```
     https://www.kaggle.com/datasets/bushraqurban/tourism-and-economic-impact
     ```

2) **Hotel Booking Demand**
   - Contains booking details (lead_time, guest type, channel, deposit, etc.).
   - Target: `is_canceled` (whether a booking is canceled).
   - Source (Kaggle):
     ```
     https://www.kaggle.com/datasets/ahmedsafwatgb20/hotel-bookingscsv
     ```

## Download Data
You can download the datasets manually from Kaggle or via the Kaggle CLI.

### Option 1: Manual Download
1) Open the dataset links in the **Datasets** section.
2) Click **Download** and extract the files.
3) Place the CSV files in this folder:
   - `world_tourism_economy_data.csv`
   - `hotel_bookings.csv`

### Option 2: Kaggle CLI
1) Install the Kaggle CLI:
```
pip install kaggle
```
2) Configure your Kaggle API credentials:
   - Go to Kaggle > Account > Create New API Token.
   - Place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or
     `%USERPROFILE%\.kaggle\` (Windows).
3) Download and unzip the datasets:
```
kaggle datasets download -d bushraqurban/tourism-and-economic-impact -p . -f world_tourism_economy_data.csv
kaggle datasets download -d ahmedsafwatgb20/hotel-bookingscsv -p . -f hotel_bookings.csv
```
If the filenames differ after download, rename them to match the expected names above.

## Method Summary
### 1. Unsupervised (clustering.ipynb)
- Clean missing values per country and fill remaining with median.
- Feature engineering: `travel_balance`, `receipts_per_arrival`.
- Standardize numeric features + one-hot encode `country`.
- Choose K using **Elbow** and **Silhouette**.
- Visualize clusters using **PCA 2D**.

### 2. Supervised (supervised_model.ipynb)
- Impute missing values (mode for `children`, `country`; 0 for `agent`, `company`).
- Encode categorical features (label + one-hot).
- 80/20 train-test split (stratified, random_state=42).
- Models: Logistic Regression, Decision Tree, Random Forest, Linear SVM.
- Add a baseline model (DummyClassifier) for comparison.
- Handle class imbalance with `class_weight`.
- Evaluate using classification metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC).
- Validate performance with 5-fold Stratified Cross-Validation.

## Key Results (brief)
- **Clustering** reveals distinct destination segments based on tourism/economic profiles.
- **Best model** is selected based on evaluation metrics shown in the notebook.
- Most influential features (from feature analysis): `lead_time`, `deposit_type`, `adr`,
  `total_of_special_requests`, `market_segment`, `agent`, and `previous_cancellations`.

## How to Run
1) Open the notebooks in Jupyter/VS Code:
   - `clustering.ipynb`
   - `supervised_model.ipynb`
2) Run cells from top to bottom.

## Requirements
Use Python 3.x and install:
```
pip install -r requirements.txt
```
