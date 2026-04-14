# 🏠 Boston House Price Prediction App

A Machine Learning web application built using **Streamlit** that predicts housing prices based on various features like crime rate, number of rooms, tax rate, etc.

---

## 📌 Project Overview

This project uses the **Boston Housing Dataset** to train a regression model that predicts house prices. The model is deployed using Streamlit for an interactive user experience.

---

## 🚀 Features

* 📊 Data visualization and dataset preview
* ⚙️ Data preprocessing (handling missing values & scaling)
* 🤖 Machine Learning model (Random Forest Regressor)
* 📈 Model evaluation (RMSE & R² Score)
* 🔮 Real-time house price prediction
* 💻 Simple and interactive UI using Streamlit

---

## 🧠 Tech Stack

* Python 🐍
* Streamlit 🌐
* Pandas & NumPy 📊
* Scikit-learn 🤖

---

## 📁 Project Structure

```
boston-app/
│
├── app.py              # Streamlit application
├── train_model.py      # Model training script
├── HousingData.csv     # Dataset
├── model.pkl           # Trained model
├── scaler.pkl          # Scaler object
├── imputer.pkl         # Imputer object
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/your-username/boston-house-app.git
cd boston-house-app
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Train the Model

```
python train_model.py
```

This will generate:

* `model.pkl`
* `scaler.pkl`
* `imputer.pkl`

---

### 4. Run the Application

```
streamlit run app.py
```

---

## 🌐 Deployment

You can deploy this app for free using:

* Streamlit Community Cloud (Recommended)
* Render
* Railway

---

## 📊 Model Details

* **Algorithm:** Random Forest Regressor
* **Evaluation Metrics:**

  * RMSE (Root Mean Squared Error)
  * R² Score

---

## ⚠️ Important Notes

* Ensure the dataset file is named **HousingData.csv**
* The target column must be **MEDV**
* All `.pkl` files should be in the same directory as `app.py`

---

## 🎯 Future Improvements

* 📈 Add advanced models like XGBoost
* 📊 Interactive visualizations
* 🌍 Map-based predictions
* ☁️ Cloud database integration

---

## 🤝 Contributing

Feel free to fork this repository and improve the project!

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed by **Your Name**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
