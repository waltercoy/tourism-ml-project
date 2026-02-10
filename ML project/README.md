# Tourism Destination Segmentation & Visitor Prediction

This project applies machine learning to:
1) segment tourism destinations (unsupervised learning), and
2) forecast international tourist arrivals (supervised learning) to identify countries likely to become top tourism destinations.

## 1.1 Problem Statement
Tourism destination segmentation and visitor prediction are key components in
understanding tourist behavior and enhancing the development of tourism potential.
Segmentation allows us to categorize destinations based on commonalities such as
tourism expenditures, arrivals, and socioeconomic indicators.
By identifying these patterns, tourism stakeholders can better accommodate market
demands and identify niche markets, enabling more strategic development and tailored
services.

Visitor prediction complements segmentation by applying empirical and behavioral
analysis to forecast future arrivals/visitor volume. This allows stakeholders to
allocate resources more efficiently, anticipate policy needs, and adjust marketing
efforts in advance.

Together, segmentation and prediction serve as powerful tools for boosting the
efficiency, sustainability, and economic viability of tourism strategies.

## 1.2 Project Objectives
The objectives of this project are:
- To identify and analyze patterns and profiles in global tourism destinations through clustering techniques.
- To segment tourist destinations based on factors such as expenditures, arrivals, and socioeconomic characteristics.
- To develop a predictive model for forecasting future international tourist arrivals based on historical arrivals and tourism/economic indicators.

## Overview
- **Unsupervised**: K-Means clustering on global tourism economy data to group countries with similar tourism profiles.
- **Supervised**: International tourist arrivals forecasting (1-year-ahead) using time-aware evaluation and regression models.

## Folder Structure
- `clustering.ipynb` - destination segmentation notebook (K-Means + PCA).
- `supervised_model.ipynb` - arrivals forecasting notebook (merge + feature engineering + ranking future destinations).
- `world_tourism_economy_data.csv` - raw global tourism dataset.
- `world_tourism_economy_data_processed.csv` - preprocessed dataset for clustering.
- `international-tourist-arrivals.csv` - international tourist arrivals time series dataset.

## Datasets
1) **World Tourism Economy Data**
   - Contains metrics such as tourism_receipts, tourism_arrivals, tourism_exports,
     tourism_departures, tourism_expenditures, gdp, inflation, unemployment, and country.
   - Used for destination segmentation and as explanatory variables for forecasting.
   - Source (Kaggle):
     ```
     https://www.kaggle.com/datasets/bushraqurban/tourism-and-economic-impact
     ```

2) **International Tourist Arrivals**
   - Country-year time series of international tourist arrivals.
   - Used as the supervised learning target (arrivals forecasting).

## Download Data
You can download the datasets manually from Kaggle or via the Kaggle CLI.

### Option 1: Manual Download
1) Open the dataset links in the **Datasets** section.
2) Click **Download** and extract the files.
3) Place the CSV files in this folder:
   - `world_tourism_economy_data.csv`
   - `international-tourist-arrivals.csv`

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
```
For `international-tourist-arrivals.csv`, download it from its source (e.g., Kaggle/export) and place it in this folder with the same filename.
If the filenames differ after download, rename them to match the expected names above.

## Method Summary
### 1. Unsupervised (clustering.ipynb)
- Clean missing values per country and fill remaining with median.
- Feature engineering: `travel_balance`, `receipts_per_arrival`.
- Aggregate to a country-level table (mean over a stable period, e.g., 2010–2019).
- Impute missing values and standardize numeric features.
- Choose K using **Elbow** and **Silhouette**.
- Visualize clusters using **PCA 2D**.

### 2. Supervised (supervised_model.ipynb)
- Merge the two datasets by `(country_code/Code, year/Year)`.
- Create time-series lag features for arrivals (e.g., previous-year arrivals, rolling mean).
- Predict next-year arrivals (1-year-ahead forecast) and rank countries by predicted arrivals and/or growth.
- Evaluate using a time-aware split (e.g., train up to 2018, test 2019; optionally treat 2020 as a stress-test year).

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
