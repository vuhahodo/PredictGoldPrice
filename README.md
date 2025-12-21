# 🪙 Gold Price AI Predictor

**Web application for gold price prediction using AI/ML models with interactive visualization and multiple forecasting algorithms.**

---

## 🌐 Live Demo

- **Web App (Vercel)**: https://predict-gold-price.vercel.app
- **Backend API (Render - Swagger)**: https://predict-gold-backend.onrender.com/docs

> Note: Render free tier may spin down when idle. The first request can take ~30–50 seconds to wake up.

---
## 🎥 Demo Video

[Watch the demo](https://youtu.be/TsfV7JxbE3E)

- Quick highlights:
   - End-to-end prediction flow (upload → configure → predict → visualize)
   - LSTM vs GM(1,1) comparison with MAPE
   - Interactive chart with actual vs predicted prices
   - Clean, responsive UI built with React + Vite


---

## 📌 Overview

Gold Price AI Predictor is a full-stack web application that combines machine learning and time-series forecasting to predict gold prices. The application features two complementary models:
- **GM(1,1)**: Grey Model for baseline predictions
- **LSTM**: Deep Learning model for advanced sequence-to-sequence forecasting

Users can upload historical gold price data, select a prediction model, and visualize results through an interactive dashboard.

---

## ✨ Key Features

- 📊 **Interactive Dashboard**: Real-time visualization of actual vs predicted prices
- 📈 **Multiple Models**: Choose between GM(1,1) and LSTM algorithms
- 📁 **CSV Upload**: Support for custom historical gold price data
- 🎯 **Performance Metrics**: MAPE (Mean Absolute Percentage Error) calculation
- 🎨 **Responsive UI**: Modern React interface with Recharts visualization
- ⚡ **Fast API**: RESTful backend with FastAPI for quick predictions

---

## 📊 Dataset

This project uses the **Gold Price 10 Years (2013-2023)** dataset from Kaggle:
- **Source**: [Kaggle - Gold Price Dataset](https://www.kaggle.com/datasets/farzadnekouei/gold-price-10-years-20132023)
- **Time Period**: 2013-2023 (10 years of historical gold price data)
- **Format**: CSV with historical gold price records
- **Location**: `backend/LSTM/Gold Price (2013-2023).csv`

The dataset contains daily gold price information spanning a decade, providing comprehensive historical context for training the LSTM model and testing predictions with real-world market data.

---

## 🛠 Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **ML/DL**: TensorFlow/Keras, NumPy, Pandas, scikit-learn
- **Model Storage**: Pre-trained LSTM model (lstm_gold.h5)
- **Dependencies**: joblib, plotly

### Frontend
- **Framework**: React 19 + TypeScript
- **Build Tool**: Vite
- **Visualization**: Recharts
- **UI Components**: Lucide React Icons
- **APIs**: Google GenAI SDK

---

## 📁 Project Structure

```
goldpriceai-predictor/
├── backend/
│   ├── main.py                          # FastAPI entry point
│   ├── requirements.txt                 # Python dependencies
│   ├── models/
│   │   ├── lstm_service.py             # LSTM prediction service
│   │   └── gm11_service.py             # Grey model service
│   ├── utils/
│   │   ├── preprocess.py               # Data preprocessing
│   │   └── metrics.py                  # Evaluation metrics
│   ├── artifacts/
│   │   └── lstm_gold.h5                # Pre-trained LSTM model
│   └── LSTM/
│       ├── Gold Price (2013-2023).csv  # Training data
│       └── gold-price-prediction-lstm-96-accuracy.ipynb
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Main application component
│   │   ├── index.tsx                   # Entry point
│   │   ├── types.ts                    # TypeScript types
│   │   ├── services/
│   │   │   ├── predictService.ts       # Backend API client
│   │   │   └── geminiService.ts        # Google Gemini API integration
│   │   └── utils/
│   │       └── greyModel.ts            # GM(1,1) implementation
│   ├── index.html                      # HTML template
│   ├── package.json                    # Node.js dependencies
│   ├── tsconfig.json                   # TypeScript configuration
│   ├── vite.config.ts                  # Vite build configuration
│   └── vite-env.d.ts                   # Vite types
│
└── README.md                           # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

#### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

**Backend Dependencies:**
- fastapi
- pandas
- numpy
- scikit-learn
- joblib
- tensorflow
- plotly

#### 2. Frontend Setup

```bash
cd frontend
npm install
# or
yarn install
```

### Running the Application

#### Start Backend Server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`

#### Start Frontend Development Server
```bash
cd frontend
npm run dev
# or
yarn dev
```
The application will be available at `http://localhost:3000`

### Building for Production

```bash
cd frontend
npm run build
# or
yarn build
```

---

## 🔧 Configuration

### Backend CORS Settings
CORS origins are configured via `FRONTEND_ORIGIN` (comma-separated). If not set, the backend defaults to `http://localhost:3000` for local development.

### Frontend API Endpoint
Update the API base URL in `frontend/services/predictService.ts` if running on different ports.

### Environment Variables

**Frontend (Vercel):**
- `VITE_API_BASE_URL=https://predict-gold-backend.onrender.com`

**Backend (Render):**
- `FRONTEND_ORIGIN=https://predict-gold-price.vercel.app`

---

## 📊 API Endpoints

### LSTM Prediction
**POST** `/predict/lstm`

**Request:**
- `file`: CSV file with historical data
- `date_col`: Column name for dates
- `price_col`: Column name for prices
- `window_size`: Sequence window size (default: 60)
- `test_year`: Year to use for testing (default: 2022)

**Response:**
```json
{
  "dates": ["2022-01-01", "2022-01-02", ...],
  "actual": [1800.5, 1805.2, ...],
  "predicted": [1798.3, 1806.1, ...],
  "mape": 2.34
}
```

---

## 🧠 Models Overview

### GM(1,1) - Grey Model
- **Type**: First-order univariate grey model
- **Advantage**: Simple, requires minimal historical data
- **Use Case**: Quick baseline predictions
- **Location**: `frontend/utils/greyModel.ts`

### LSTM - Long Short-Term Memory
- **Type**: Deep recurrent neural network
- **Training Data**: Gold prices (2013-2023)
- **Accuracy**: ~96%
- **Advantage**: Captures complex temporal patterns
- **Use Case**: Detailed, accurate predictions
- **Model**: `backend/artifacts/lstm_gold.h5`
- **Training Notebook**: `backend/LSTM/gold-price-prediction-lstm-96-accuracy.ipynb`

---

## 📈 Data Format

### Expected CSV Format
```csv
Date,Price
2023-01-01,1800.50
2023-01-02,1805.25
2023-01-03,1802.75
```

**Required Columns:**
- **Date Column**: Should be in format YYYY-MM-DD (or specify custom format)
- **Price Column**: Numerical values representing gold prices

---

## 🔍 Usage Guide

1. **Open the Application**: Navigate to `http://localhost:3000` in your browser
2. **Upload Data**: Click "Upload CSV" and select your historical gold price data
3. **Configure Parameters**:
   - Select date column
   - Select price column
   - Set window size (for LSTM)
   - Specify test year
4. **Choose Model**:
   - Select LSTM or GM(1,1) from dropdown
5. **Run Prediction**: Click "Predict" button
6. **View Results**: Interactive chart shows actual vs predicted prices with MAPE score

---

## 🛡️ Error Handling

The application includes comprehensive error handling:
- Invalid CSV format validation
- Missing column detection
- API connection errors with user feedback
- Data preprocessing exceptions

---

## 📝 Model Performance

- **LSTM Model**: 96% accuracy on test set (2023 data)
- **Training Period**: 2013-2023 (10 years of historical data)
- **Metrics Used**: MAPE (Mean Absolute Percentage Error)

---

## 🔄 Future Enhancements

- [ ] Support for multiple time-series models (ARIMA, Prophet)
- [ ] Real-time data fetching from market APIs
- [ ] Model retraining interface
- [ ] Advanced statistical analysis
- [ ] Export predictions to multiple formats
- [ ] Price alert notifications

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👥 Authors

Gold Price AI Predictor - A specialized forecasting solution for precious metal price prediction.

---

## 📞 Support

For issues, questions, or contributions, please open an issue in the repository or contact the development team.

---

**Last Updated**: December 2025
