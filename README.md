# CUSTHEALTH - Aplikasi Cluster Prediksi Asuransi

**CUSTHEALTH** is a Streamlit-based web application designed to predict hospital revenue based on clustering analysis of insurance companies and time series regression models.

---

## 🚀 Key Features

### 🧠 Insurance Company Clustering
- **Dynamic Time Warping (DTW)** and **PCA** for time series data preprocessing
- **KMeans Clustering** with interactive visualizations
- **Davies-Bouldin Index** for cluster evaluation
- **Downloadable cluster summary reports**

### 📈 Revenue Forecasting
- **Perusahaan-asuransi-based revenue prediction**
- **Time series regression** using linear models
- **Future revenue forecasting** with lag variables
- Interactive visualizations for actual vs predicted revenue

### 🗂️ Data Handling
- Automatic data parsing from uploaded Excel files
- Revenue breakdown by category: Emergency, Check-up, Inpatient, Outpatient
- Yearly aggregation from 2018–2022

---

## 📂 File Structure

```
.
├── main.py                      # Main Streamlit app interface
├── functions_.py                # Data preprocessing, modeling & visualization functions
├── requirements.txt             # List of Python dependencies
├── dataset/                     # Folder for raw Excel input files
```

---

## 🛠️ Tech Stack

| Layer     | Tech Used                                |
|-----------|-------------------------------------------|
| Frontend  | Streamlit, Plotly                         |
| Backend   | Python, Scikit-learn, Keras, NumPy, Pandas|
| Modeling  | KMeans, PCA, LSTM, Linear Regression      |

---

## 📸 Screenshots

> _Add screenshots of cluster results, revenue predictions, etc._

---

## 🔧 How to Run Locally

```bash
# 1. Clone the repository
https://github.com/alexandertiopan1212/Streamlit_App_Hospital_Revenue_Predictor.git

# 2. Navigate into the folder
cd Streamlit_App_Hospital_Revenue_Predictor

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run main.py
```

---

## 📜 License
This project is open-source and available for both academic and commercial use.

---

## 🙌 Acknowledgments
Developed for analytical use in predicting and planning hospital revenue strategies from insurance companies.
