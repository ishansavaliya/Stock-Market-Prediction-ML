# 📈 Stock Market Prediction System (ML)

A **Deep Learning-based Stock Market Prediction System** that forecasts stock prices using historical data with **LSTM Neural Networks** and provides alternative predictions using traditional ML models.

## ✅ Features

- **Trained using LSTM Neural Networks (Deep Learning)**
- **Multiple Moving Average Visualizations**
- **Company Name Search Functionality**
- **Alternative ML Models: Random Forest & Gradient Boosting**
- **Includes a Web App (Streamlit) for easy usage**

---

## 📂 Project Structure

```
├── app.py                              # Streamlit UI for stock predictions
├── Stock_Market_Prediction_Model_Creation.ipynb  # Notebook for model creation
├── Stock Predictions Model.keras       # Saved trained LSTM model
├── random_forest_stock_model.joblib    # Alternative Random Forest model
├── gradient_boosting_stock_model.joblib # Alternative Gradient Boosting model
├── README.md                           # Documentation
```

---

## ⚡ Installation & Setup

### 1️⃣ Create a Python Environment

Create a conda environment with Python 3.11 (for TensorFlow compatibility):

```bash
conda create -n stock_env python=3.11
conda activate stock_env
```

### 2️⃣ Install Dependencies

Install the required Python packages:

```bash
pip install tensorflow numpy pandas matplotlib yfinance scikit-learn streamlit ipykernel
```

### 3️⃣ Train the Model (Optional - If not already trained)

Run the Jupyter notebook to train the LSTM model on historical stock data:

```bash
jupyter notebook Stock_Market_Prediction_Model_Creation.ipynb
```

✅ Downloads historical stock data  
✅ Prepares data with proper scaling  
✅ Trains the LSTM neural network  
✅ Evaluates performance & saves the model

### 🚀 Running the Streamlit Web App

Launch the web app using:

```bash
streamlit run app.py
```

➡️ Open the browser URL (usually http://localhost:8501)  
➡️ Enter company name (e.g., Apple, Microsoft, Tesla)  
➡️ View stock data visualizations and predictions instantly!

---

## 📊 Model Architecture

| Layer | Type  | Units | Activation |
| ----- | ----- | ----- | ---------- |
| 1     | LSTM  | 50    | ReLU       |
| 2     | LSTM  | 60    | ReLU       |
| 3     | LSTM  | 80    | ReLU       |
| 4     | LSTM  | 120   | ReLU       |
| 5     | Dense | 1     | Linear     |

🚀 **Key Features:** Multiple LSTM layers with dropout for robust predictions

---

## 📊 Alternative Models

| Model             | Description                                            |
| ----------------- | ------------------------------------------------------ |
| Random Forest     | Ensemble learning method using multiple decision trees |
| Gradient Boosting | Sequential ensemble technique for improved performance |

---

## 🛠 Future Improvements

- 🔹 Incorporate sentiment analysis from news and social media
- 🔹 Add technical indicators for improved predictions
- 🔹 Implement portfolio optimization features
- 🔹 Add more visualization options for technical analysis

---

## 📌 Now, Run It! 🚀

```bash
streamlit run app.py
```

📈 **Invest Smarter with AI!** 📊
