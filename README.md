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

![image](https://github.com/user-attachments/assets/0e2a71de-4d9a-488a-9cb7-e8da2302f2a4)
![image](https://github.com/user-attachments/assets/61465452-7a85-47fa-a47c-ff43a2cd3b1d)
![image](https://github.com/user-attachments/assets/6487181b-f111-4ca4-bc80-04ccb8765af8)
![image](https://github.com/user-attachments/assets/a6698da3-bcce-4f7a-8390-9286d160c68b)
![image](https://github.com/user-attachments/assets/c76c1847-92bb-4562-8bee-e18b9c3a1574)
![image](https://github.com/user-attachments/assets/27b559de-34a2-45af-8a8a-27f87832dc9e)


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
