
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
git clone https://github.com/Tanishka712/automated-ml-project.git
cd automated-ml-project


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

![image](https://github.com/user-attachments/assets/a90faa7c-715c-43c0-819a-4cba1dcc9b39)
![image](https://github.com/user-attachments/assets/132e2b00-7ad9-4793-8637-1fd6e03718c3)
![image](https://github.com/user-attachments/assets/9c16cc52-16f7-4620-8629-cdeda54a2ed9)
![image](https://github.com/user-attachments/assets/d96a4a07-87df-4c6e-baa6-93eedc69af12)
![image](https://github.com/user-attachments/assets/55dff6d2-b388-4476-8692-e37909cd631f)
![image](https://github.com/user-attachments/assets/1c85a843-ce10-4532-a18e-b6bd6b815dda)
![image](https://github.com/user-attachments/assets/d56f826b-6e88-4f63-9265-45fb50497c62)
![image](https://github.com/user-attachments/assets/7458eb05-78d9-4f80-806a-44430b2def55)
![image](https://github.com/user-attachments/assets/c6c72a5d-0595-4a14-98ea-89f3e5af39c3)
![image](https://github.com/user-attachments/assets/1ad6ff33-b1de-4d9e-9a0b-42e445bc13b0)



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


