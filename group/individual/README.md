# London Rental Price Prediction
## MSIN0097 Predictive Analytics — UCL 2025/26

## Dataset
Download from Kaggle: https://www.kaggle.com/datasets/jakewright/house-price-data
Place the CSV file in the project root directory as `kaggle_london_house_price_data.csv`

## Environment Setup
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python "Predictive Analysis.py"
```

## Output
Running the script generates:
- rent_distribution.png
- missing_values.png
- correlation.png
- residuals_analysis.png
- feature_importance.png
- error_by_price.png
```

**.gitignore** — ：
```
*.csv
__pycache__/
*.pyc
.DS_Store
