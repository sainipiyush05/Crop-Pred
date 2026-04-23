# 🌾 AgroWare Crop Prediction

AgroWare is an AI-powered crop recommendation system designed to help farmers and agricultural stakeholders make data-driven decisions. By analyzing real-time and historical environmental data, AgroWare predicts the most suitable crops for a specific location and time of year.

## 🚀 Key Features

- **Automated Feature Engineering**: Fetches real-time environmental data using just a location name and month.
- **Multi-Source Data Integration**:
  - **Geocoding**: Uses OpenCage API to resolve locations to precise coordinates.
  - **Weather Data**: Integrates NASA POWER (climatology) and OpenWeather (real-time fallback) for temperature, humidity, and rainfall.
  - **Soil Properties**: Fetches Nitrogen and pH from SoilGrids v2, with ICAR (Indian Council of Agricultural Research) state-level averages for Phosphorus and Potassium.
- **High-Performance ML Model**: Powered by an XGBoost classifier trained on a comprehensive dataset of 22 different crop types.
- **RESTful API**: Built with FastAPI for high performance and easy integration with mobile or web frontends.

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Machine Learning**: XGBoost, Scikit-Learn, Pandas, NumPy
- **Data APIs**: OpenCage, NASA POWER, OpenWeather, SoilGrids
- **Notebooks**: Jupyter for model training and EDA

## 📋 Prerequisites

- Python 3.10 or higher
- API Keys for:
  - [OpenCage Geocoding](https://opencagedata.com/)
  - [OpenWeatherMap](https://openweathermap.org/) (Optional fallback)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sainipiyush05/Crop-Pred.git
   cd AgroWare_crop
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENCAGE_API_KEY=your_opencage_key
   OPENWEATHER_API_KEY=your_openweather_key
   ```

## 🚀 Running the API

Start the development server:
```bash
python app.py
```
The API will be available at `http://localhost:8000`. You can access the interactive documentation (Swagger UI) at `http://localhost:8000/docs`.

## 📡 API Endpoints

### 1. Get Environmental Features
Collects N, P, K, temperature, humidity, pH, and rainfall for a given location.

- **URL**: `/predict/features`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "location": "Amritsar, Punjab",
    "month": 10
  }
  ```
- **Response**: Returns the calculated feature vector ready for model input.

## 📊 Model Training

The training pipeline is documented in `models/train_model.ipynb`. It covers:
- Data cleaning and encoding.
- Exploratory Data Analysis (EDA).
- Feature importance analysis.
- Model serialization using `joblib`.

## 📂 Project Structure

- `app.py`: FastAPI entry point.
- `services/`: API integration services (Weather, Soil, GeoLocation).
- `models/`: Trained models and training notebooks.
- `data/`: Datasets used for training.
- `data_pipeline.py`: Data cleaning and preprocessing pipeline.

## 🤝 Acknowledgments

- [NASA POWER](https://power.larc.nasa.gov/) for climate data.
- [ISRIC SoilGrids](https://www.isric.org/) for global soil information.
- [ICAR](https://icar.org.in/) for Indian soil survey data.


## 🚀 How to Run

### 1. Start the API Server
Run the following command in the project root:
```bash
python3 app.py
```
*Alternatively, using uvicorn directly:*
```bash
uvicorn app:app --reload
```

### 2. Interactive API Documentation
Once the server is running, open your browser and navigate to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs) (Recommended for testing)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 3. Test Crop Prediction
You can test the prediction endpoint using `curl`:
```bash
curl -X POST "http://localhost:8000/predict/crop" \
     -H "Content-Type: application/json" \
     -d '{
           "location": "Punjab",
           "month": 6
         }'
```