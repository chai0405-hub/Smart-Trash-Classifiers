# Smart Trash AI ♻️

An AI-powered waste classification system built using **Machine Learning** and **Flask API**.
The system predicts the **type of waste material** (plastic, metal, organic, cardboard, etc.) based on sensor data.

---

## 📌 Features

* Machine Learning waste classification
* REST API built with Flask
* Predict waste type from sensor data
* Supports batch predictions
* Ready for cloud deployment
* Simple and scalable project structure

---

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** from Scikit-Learn to classify waste materials.

Model input examples:

* Weight
* Moisture level
* Metal detection
* Plastic detection
* Organic level

Model output:

* Plastic
* Metal
* Organic
* Cardboard
* Glass

---

## 📂 Project Structure

```
smart-trash-ai/
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── smart_trash_classifier_dataset.csv
├── trash_classifier_model.pkl
├── label_encoder.pkl
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/smart-trash-ai.git
cd smart-trash-ai
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```
python train_model.py
```

This will generate:

* `trash_classifier_model.pkl`
* `label_encoder.pkl`

---

## 🔎 Test Prediction

```
python predict.py
```

Example Output:

```
Predicted Waste Type: cardboard
```

---

## 🚀 Run the API Server

Start the Flask server:

```
python app.py
```

Server runs at:

```
http://127.0.0.1:5000
```

---

## 📡 API Usage

### Predict Waste Type

Endpoint:

```
POST /predict
```

Example JSON request:

```
{
  "weight": 120,
  "moisture": 20,
  "metal_detected": 0,
  "plastic_detected": 1,
  "organic_level": 10
}
```

Example Response:

```
{
  "predicted_waste_type": "cardboard"
}
```

---

## 🛠 Technologies Used

* Python
* Flask
* Pandas
* Scikit-learn
* Joblib
* NumPy

---

## 🌍 Future Improvements

* Real-time IoT sensor integration
* Smart trash bin dashboard
* Camera-based waste detection
* Mobile app integration
* Cloud deployment

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed as a **Smart Waste Management AI Project**.
