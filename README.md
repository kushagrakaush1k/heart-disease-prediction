# Heart Disease Prediction

Tried to figure out if we can predict heart disease from basic clinical data. Short answer — yes, reasonably well. Used the Cleveland Heart Disease dataset and tested a couple of ML approaches to see which one held up better under cross-validation.

---

## What this is

The dataset has 303 patient records with 14 features — things like age, chest pain type, cholesterol, max heart rate, etc. The goal is binary classification: does this person have heart disease or not.

I went with **KNN** as the main model and **Random Forest** as a comparison. Nothing fancy, but wanted to understand the data properly before throwing a black-box model at it.

---

## Dataset

Cleveland Heart Disease Dataset — 303 patients, 14 clinical features.

| Feature | Description |
|--------|-------------|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia type |
| **target** | **1 = disease, 0 = no disease** |

Download: [heart.csv](dataset/heart.csv)

---

## Project Structure

```
heart-disease-prediction/
│
├── dataset/
│   └── heart.csv
├── notebook/
│   └── heart_disease_prediction.ipynb
├── images/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   └── knn_accuracy.png
├── requirements.txt
└── README.md
```

---

## What I did

**1. Explored the data**
Checked class balance first — roughly 54% positive (disease) vs 46% negative (no disease), so no major imbalance issue.

**2. Feature correlation**
Ran a heatmap to see what actually correlates with the target. `thalach` (max heart rate) and `cp` (chest pain type) came out as the strongest positive indicators. `exang` and `oldpeak` were negatively correlated — higher values meant lower disease likelihood.

**3. Preprocessing**
- One-hot encoded all categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`)
- StandardScaler on continuous features: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`

**4. Model training**
Tested KNN for k = 1 to 20 using 10-fold cross-validation. k=12 gave the best result. Then ran Random Forest with 10 estimators for comparison.

---

## Results

| Model | CV Accuracy |
|-------|------------|
| KNN (k=12) | **84.48%** |
| Random Forest | 81.14% |

KNN edged out Random Forest here — probably because the dataset is small and the feature space after one-hot encoding isn't huge, so a distance-based model does fine.

---

## Visualizations

**Target Distribution**

![Target Distribution](images/target_distribution.png)

**Correlation Heatmap**

![Heatmap](images/correlation_heatmap.png)

**KNN Accuracy vs K Value**

![KNN Accuracy](images/knn_accuracy.png)

---

## How to run

```bash
git clone https://github.com/kushagrakaushik1k/heart-disease-prediction
cd heart-disease-prediction
pip install -r requirements.txt
```

Then open `notebook/heart_disease_prediction.ipynb` in VS Code or Jupyter and run all cells.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

-> Takeaways

- Max heart rate (`thalach`) being a strong predictor makes clinical sense — the heart's response to stress is a direct indicator
- KNN worked better than Random Forest on this dataset likely due to small data size
- Cross-validation was important here since 303 samples isn't a lot — a simple train/test split would've been too noisy


-> References

- [Original article by Aman Kharwal](https://amanxai.com/2020/05/20/heart-disease-prediction-with-machine-learning/)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)