
# 🚀 AutoML Streamlit App

An end-to-end **AutoML web application** built with **Streamlit**, allowing users to:

* Upload datasets (`.csv`)
* Automatically clean and profile data
* Train multiple ML models (classification or regression)
* View comparisons and accuracy metrics
* Download predictions, compare reports, and the best saved model

---

## 📁 Folder Structure

```
├── app.py                  # Main Streamlit app
├── requirements.txt        # All dependencies for setup
├── runtime.txt             # Runtime environment details
├── logs.log                # Logs file (runtime output)
├── README.md               # Project documentation
├── data/                   # Stores uploaded and cleaned datasets
├── downloaded files/       # Stores predictions, reports, and model files
```

---

## 🌟 Features

### ✅ Upload CSV File

Upload any structured dataset in `.csv` format.

### ✅ Auto Data Cleaning

* Drops columns with excessive nulls
* Fills missing values:

  * Numerical → Median
  * Categorical → Mode
* Removes constant or corrupt columns

### ✅ Data Profiling (EDA)

* Uses **`ydata-profiling`** for comprehensive profiling
* Includes:

  * Correlation heatmaps
  * Value distributions
  * Missing values matrix
  * Variable interactions

### ✅ ML Model Training

Supports two tasks:

* **Classification**
* **Regression**

Functionality:

* Uses **PyCaret** to automatically compare models
* Displays model performance
* Saves:

  * Best performing model (`.pkl`)
  * Predictions on test data
  * Comparison report of all models

### ✅ Downloads

* Download:

  * `best_model.pkl`
  * `predictions.csv`
  * `compare_report.csv`

---

## 🖥️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/automl-streamlit-app.git
cd automl-streamlit-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 🔗 Dependencies

Major dependencies include:

* `streamlit`
* `pycaret`
* `pandas`
* `scikit-learn`
* `ydata-profiling`
* `xgboost`, `lightgbm`
* `matplotlib`, `seaborn`, `numpy`

See `requirements.txt` for full list.

---

## 📦 Deployment

This app is designed to work on **Streamlit Cloud** out of the box. Just push to GitHub and deploy using your repo link.

---

## 📸 Demo

> *Add a screenshot or GIF of the deployed app here.*

---

## 📁 Sample Outputs

All generated files will appear inside `downloaded files/`:

* `predictions.csv`
* `compare_report.csv`
* `best_model.pkl`

Uploaded & cleaned datasets are stored in `data/`.

---

## ✍️ Author

**Tanishka Nagawade**


