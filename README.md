# 💳 Fraud Transaction Detection using LightGBM


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d205c433-9abf-4b67-a3e0-b9058959569a" />


## 📌 Overview  

This project builds a **Fraud Transaction Detection System** using a large-scale transactions dataset (`1.75M+ rows`).  
We leverage **LightGBM (Gradient Boosted Decision Trees)** to classify whether a financial transaction is **fraudulent (1)** or **legitimate (0)**.  

The model is designed to handle **high class imbalance** (fraudulent transactions are very rare compared to normal ones).  
Key evaluation metrics include **Precision, Recall, F1-score, ROC-AUC, and Average Precision Score (PR AUC)**.

---

## 🚀 Features  

- 📊 **Exploratory Data Analysis (EDA)**  
  - Fraud rate analysis (frauds are <1% of all transactions)  
  - Customer activity patterns  
  - Terminal-level fraud detection  

- ⚡ **Feature Engineering**  
  - Time-based features from `TX_DATETIME`  
  - Customer historical behavior (fraud history, spending)  
  - Terminal risk profiles  

- 🤖 **Machine Learning Model**  
  - **LightGBM Classifier** with class imbalance handling (`scale_pos_weight`)  
  - Probability threshold tuning for best precision-recall tradeoff  

- 🎯 **Evaluation Metrics**  
  - Precision, Recall, F1-score  
  - **ROC-AUC**  
  - **Average Precision Score (PR AUC)**  

---

## 🏗️ Project Pipeline  

<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline Diagram" width="75%">
</p>

1. **Data Preprocessing**  
   - Datetime conversion (`TX_DATETIME`)  
   - Memory optimization with categorical and float32 types  

2. **Feature Engineering**  
   - Customer fraud/transaction history  
   - Terminal risk factors  
   - Transaction amount behavior  

3. **Model Training with LightGBM**  
   - Handled imbalance with `scale_pos_weight`  
   - Trained on ~70% transactions, tested on ~30% (time-based split)  

4. **Threshold Optimization**  
   - Best threshold: **0.8234**  
   - Chosen using Precision-Recall curve  

---

## 📊 Results  

### 📌 Classification Report (Threshold = 0.8234)

          precision    recall  f1-score   support

       0       1.00      0.99      1.00    347895
       1       0.60      0.89      0.72      2936

accuracy                           0.99    350831


- ✅ **Accuracy**: 99%  
- ✅ **Fraud Recall**: 89% (detects most frauds)  
- ✅ **Fraud Precision**: 60% (reasonable tradeoff)  
- ✅ **Average Precision Score (PR AUC)**: **0.8676**  

---

### 📈 Precision-Recall Curve  

<p align="center">
  <img src="assets/pr-curve.png" alt="Precision Recall Curve" width="70%">
</p>

---

## 🛠️ Tech Stack  

- **Languages:** Python (Pandas, NumPy, Matplotlib, Seaborn)  
- **ML Framework:** LightGBM  
- **Environment:** Google Colab 

---




