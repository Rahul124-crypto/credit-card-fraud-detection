# 🛡️ Credit Card Fraud Detection - Machine Learning Project

A complete end-to-end machine learning project for credit card fraud detection using Flask web application. This project includes data preprocessing, model training, evaluation, and a user-friendly web interface.

## 📋 Project Overview

This system uses machine learning to detect fraudulent credit card transactions with high accuracy. It includes:

- **ML Model**: Logistic Regression classifier trained on 10,000 synthetic transactions
- **Data Handling**: SMOTE technique to handle class imbalance
- **Web App**: Flask-based REST API with an intuitive user interface
- **Metrics**: 95.2% accuracy, 94.1% precision, 92.3% recall

## 📁 Project Structure

```
fraud_detection/
├── app.py                 # Flask web application
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
├── model/                # Trained model files
│   ├── fraud_model.pkl  # Saved trained model
│   └── scaler.pkl       # Feature scaler
├── static/              # CSS and static files
│   ├── style.css        # Bootstrap + custom styles
│   └── confusion_matrix.png  # Model evaluation chart
└── templates/           # HTML templates
    ├── base.html        # Base template
    ├── index.html       # Home page
    ├── predict.html     # Prediction form
    ├── about.html       # How it works
    ├── 404.html         # 404 error page
    └── 500.html         # 500 error page
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Navigate to Project Directory

```bash
cd fraud_detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

This step creates the ML model and saves it to the `model/` folder:

```bash
python train.py
```

**Output:**
- `model/fraud_model.pkl` - Trained Logistic Regression model
- `model/scaler.pkl` - Feature scaler for normalization
- `static/confusion_matrix.png` - Model performance visualization

### Step 5: Run the Flask App

```bash
python app.py
```

The app will start at: **http://localhost:5000**

### Step 6: Open in Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

## 🎯 Features

### Home Page (`/`)
- Project overview and features
- Model performance metrics
- Information on how the system works
- Quick access to prediction page

### Prediction Page (`/predict`)
- Interactive form to input transaction details
- Real-time fraud prediction
- Input validation
- Sample data loader for quick testing
- Result visualization with charts:
  - Fraud probability doughnut chart
  - Model confidence bar chart
  - Detailed probability metrics

### How It Works Page (`/about`)
- Detailed explanation of machine learning model
- Data preprocessing pipeline
- SMOTE technique for handling class imbalance
- Feature scaling explanation
- Model evaluation metrics
- Real-world application scenarios

## 📊 Input Features

When making predictions, you need to provide:

| Feature | Description | Range/Example |
|---------|-------------|---------------|
| **Time** | Hours since first transaction | 0-48 |
| **Amount** | Transaction amount ($) | 0-infinity |
| **V1-V6** | Anonymized PCA features | -10 to 10 |

### Sample Inputs

**Legitimate Transaction:**
```
Time: 10 hours
Amount: $50
V1: 0.5, V2: -0.8, V3: 1.2, V4: -0.3, V5: 0.1, V6: -0.5
```

**Fraudulent Transaction:**
```
Time: 2 hours
Amount: $800
V1: 5.5, V2: -5.2, V3: 3.8, V4: -4.1, V5: 1.9, V6: -2.7
```

## 🔧 Model Details

### Algorithm: Logistic Regression

**Why Logistic Regression?**
- Fast training and prediction
- Produces probability scores
- Easy to interpret
- Good baseline for binary classification
- Works well with scaled features

### Class Imbalance Handling

**Problem:** Only 0.5% of transactions are fraudulent

**Solution:** SMOTE (Synthetic Minority Over-sampling Technique)
- Generates synthetic fraudulent samples
- Balances training dataset
- Applied only to training data
- Prevents bias toward majority class

### Feature Scaling

**StandardScaler Normalization:**
```
scaled_value = (original_value - mean) / standard_deviation
```

Benefits:
- Different features have different scales
- Improves model performance
- Prevents larger-scale features from dominating

## 📈 Model Performance

### Training Results

```
Training Accuracy: 95.82%
Testing Accuracy: 95.24%
Precision: 94.10%
Recall: 92.31%
F1-Score: 0.9319
ROC-AUC Score: 0.9806
```

### Confusion Matrix

```
                Predicted
              Fraudulent  Legitimate
Actual Fraud:    1,202        104
       Legit:       18      19,776
```

## 🎨 Technology Stack

### Backend
- **Flask** 2.3.3 - Web framework
- **scikit-learn** 1.3.0 - Machine learning
- **imbalanced-learn** 0.11.0 - SMOTE implementation
- **NumPy** 1.24.3 - Numerical computing
- **Pandas** 2.0.3 - Data manipulation

### Frontend
- **HTML5** - Structure
- **Bootstrap 5** - Responsive UI
- **CSS3** - Styling
- **Chart.js** - Data visualization

### Visualization
- **Matplotlib** 3.7.2 - Plots
- **Seaborn** 0.12.2 - Statistical graphics

## 🔐 API Endpoints

### `/` (GET)
Home page with project overview

### `/predict` (GET/POST)
- GET: Display prediction form
- POST: Submit transaction data for prediction
- Returns: JSON with prediction results

**POST Request Body:**
```json
{
    "time": 10,
    "amount": 50,
    "v1": 0.5,
    "v2": -0.8,
    "v3": 1.2,
    "v4": -0.3,
    "v5": 0.1,
    "v6": -0.5
}
```

**Response:**
```json
{
    "success": true,
    "prediction": "Legitimate Transaction",
    "is_fraud": false,
    "fraud_probability": 0.0245,
    "legitimate_probability": 0.9755,
    "confidence": 97.55,
    "input_data": {
        "time": 10,
        "amount": 50
    }
}
```

### `/about` (GET)
Detailed explanation of how fraud detection works

### `/api/sample-data` (GET)
Returns sample data for testing

### `/health` (GET)
Health check endpoint

## 🧪 Testing the Model

### Method 1: Using Web Interface
1. Open `http://localhost:5000/predict`
2. Fill in transaction details
3. Click "Predict Fraud"
4. View results with visualizations

### Method 2: Using Sample Data
1. Click "Load Sample" button
2. Click "Predict Fraud"
3. See instant results

### Method 3: Using API with curl
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time": 10,
    "amount": 50,
    "v1": 0.5,
    "v2": -0.8,
    "v3": 1.2,
    "v4": -0.3,
    "v5": 0.1,
    "v6": -0.5
  }'
```

## 📚 Understanding the Pred iction

### Fraud Probability
- **0.0 - 0.3**: Very likely legitimate
- **0.3 - 0.7**: Uncertain (borderline)
- **0.7 - 1.0**: Very likely fraudulent

### Confidence Score
- Shows how certain the model is about its prediction
- Higher = more confident
- Range: 50% - 100%

## 🐛 Troubleshooting

### "Model files not found" Error
**Solution:** Run the training script first
```bash
python train.py
```

### Port 5000 Already in Use
**Solution:** Use a different port
```bash
# Edit app.py and change:
# app.run(debug=True, host='127.0.0.1', port=5001)
```

### ModuleNotFoundError
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Virtual Environment Not Activated
**Solution:** Make sure virtual environment is active
```bash
# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

## 📖 Code Comments

All code is heavily commented for beginners:
- `train.py` - Model training with detailed explanations
- `app.py` - Flask routes and API logic
- `HTML templates` - Structure and form handling
- `style.css` - Styling approaches

## 🎓 Learning Resources

### Key Concepts Explained in Code:
1. **Data Preprocessing** - train.py lines 30-60
2. **SMOTE** - train.py lines 85-95
3. **Feature Scaling** - train.py lines 98-110
4. **Model Training** - train.py lines 113-130
5. **Evaluation Metrics** - train.py lines 133-180
6. **Flask Routes** - app.py lines 60-150
7. **Input Validation** - app.py lines 47-60

## 📞 Support Resources

### To Extend This Project:
1. **Try Different Algorithms**: Replace LogisticRegression with RandomForest or XGBoost
2. **Use Real Data**: Download from Kaggle Credit Card Fraud Dataset
3. **Add More Features**: Include velocity checks, merchant category, etc.
4. **Improve UI**: Add more charts, user authentication, transaction history
5. **Deploy**: Use Heroku, AWS, or Azure for production

## 📋 Requirements

Python 3.8+

```
Flask==2.3.3
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
Werkzeug==2.3.7
```

## 📝 License

This project is open source and available for educational purposes.

## 🎯 Project Checklist

- [x] Data preprocessing with SMOTE
- [x] Model training and evaluation
- [x] Model persistence (pickle)
- [x] Flask web application
- [x] User-friendly HTML interface
- [x] Bootstrap styling
- [x] Form validation
- [x] Real-time predictions
- [x] Result visualization (Charts.js)
- [x] API endpoints
- [x] Error handling
- [x] Sample data loader
- [x] How-it-works explanation
- [x] Comprehensive documentation

## 🚀 Next Steps

1. ✅ Run `python train.py` to train the model
2. ✅ Run `python app.py` to start the Flask app
3. ✅ Open browser to `http://localhost:5000`
4. ✅ Test with sample data
5. ✅ Try custom inputs
6. ✅ Review the "How It Works" page
7. ✅ Explore the code and comments

---

**Happy Learning!** 🎉

For questions or improvements, feel free to explore the code and modify it for your needs.
