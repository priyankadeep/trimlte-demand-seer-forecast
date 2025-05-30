# Multi-Model Demand Forecasting System

A comprehensive time series forecasting application that supports **SARIMA**, **Prophet**, and **LSTM** models for demand planning and inventory management.

## Features

- **Three Forecasting Models:**
  - **SARIMA** - Statistical time series model with seasonality (Recommended)
  - **Prophet** - Robust forecasting model
  - **LSTM** - Deep learning neural network model

- **Advanced Analytics:**
  - Interactive dashboards with real-time charts
  - Comprehensive metrics (MAE, MSE, RMSE, AIC, BIC, MAPE)
  - Model performance comparison
  - AI-powered insights with Groq integration

- **Data Processing:**
  - CSV data upload and processing
  - Automatic data validation and cleaning

## Prerequisites

- **Node.js** (v16 or higher)
- **Python** (v3.11 recommended)
- **npm** or **yarn**
- **conda** (recommended for Python environment management)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd trimlte-demand-seer-forecast
```

### 2. Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Install additional UI components if needed
npm install @radix-ui/react-slider @radix-ui/react-tabs
```

### 3. Backend Setup (Python Environment)

#### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n forecast python=3.11 -y

# Activate the environment
conda activate forecast

# Install core packages with conda
conda install -c conda-forge numpy=1.24.3 pandas=2.0.3 -y
conda install -c conda-forge flask flask-cors statsmodels scikit-learn -y
conda install -c conda-forge prophet -y

# Install TensorFlow with pip
pip install tensorflow==2.13.0
```

#### Option B: Using Virtual Environment

```bash
# Create virtual environment
python -m venv forecast_env

# Activate environment
# On macOS/Linux:
source forecast_env/bin/activate
# On Windows:
forecast_env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 4. Requirements.txt

Create a `requirements.txt` file with:

```txt
flask==2.3.2
flask-cors==4.0.0
pandas==2.0.3
numpy==1.24.3
statsmodels==0.14.0
prophet==1.1.4
tensorflow==2.13.0
scikit-learn==1.3.0
```

## Running the Application

### 1. Start the Backend API

```bash
# Make sure you're in the conda environment
conda activate forecast

# Run the Python API
python forecast_api.py
```

You should see:
```
Starting Multi-Model Forecast API on http://localhost:5001
Supported models: SARIMA, Prophet, LSTM
```

### 2. Start the Frontend

Open a new terminal:

```bash
# Start the React development server
npm run dev
```

The application will open at: `http://localhost:8080`

## Usage

### 1. Data Upload
- Navigate to the **Data Upload** tab
- Upload your CSV file with columns:
  - `Order Date` - Date in YYYY-MM-DD format
  - `Item Groups` - Product categories
  - `Qty` - Quantity values
  - `Price` - Unit price
  - `Qty - w/Sub-Total` - Net quantity for revenue calculation
  - `Ext. Price` - Extended price (optional)

### 2. Model Configuration
- Go to the **Model Config** tab
- Select your forecasting model:
  - **SARIMA** (Recommended) - For regular seasonal patterns
  - **Prophet** - For data with trend changes and holidays
  - **LSTM** - For complex non-linear patterns

### 3. Generate Forecasts
- Navigate to the **Forecasting** tab
- Select an item group
- Click "Generate [Model] Forecast"
- View results with confidence intervals and performance metrics

### 4. AI Assistant
- Use the **AI Assistant** tab for insights
- Requires Groq API key for advanced analytics
- Ask questions about trends, patterns, and recommendations

## Project Structure

```
trimlte-demand-seer-forecast/
├── src/
│   ├── components/
│   │   ├── DataUpload.tsx           # Data upload and processing
│   │   ├── ModelConfiguration.tsx   # Model parameter configuration
│   │   ├── ForecastDashboard.tsx    # Main forecasting interface
│   │   ├── AIChatbot.tsx           # AI assistant
│   │   └── ui/                     # UI components
│   ├── services/
│   │   └── forecastAPI.ts          # API communication
│   ├── types/
│   │   └── forecast.ts             # TypeScript type definitions
│   └── pages/
│       └── Index.tsx               # Main application page
├── forecast_api.py                 # Python backend API
├── requirements.txt                # Python dependencies
├── package.json                    # Node.js dependencies
└── README.md                      # This file
```

## Configuration

### Model Parameters

#### SARIMA
- **Order (p,d,q)**: AR, differencing, MA parameters
- **Seasonal Order (P,D,Q,S)**: Seasonal parameters
- **Auto-select**: Automatically finds optimal parameters

#### Prophet
- **Growth**: Linear or logistic trend
- **Seasonality Mode**: Additive or multiplicative
- **Change Point Settings**: Trend change detection
- **Seasonality**: Yearly, weekly, quarterly options

#### LSTM
- **Window Size**: Number of past periods to use
- **Architecture**: LSTM layer configuration
- **Training**: Epochs, batch size, dropout rate

## Troubleshooting

### Common Issues

#### 1. Python Package Conflicts
```bash
# Clear conda cache and reinstall
conda clean --all
conda install --force-reinstall numpy pandas
```

#### 2. Prophet Installation Issues
```bash
# Install Prophet via conda (recommended)
conda install -c conda-forge prophet

# Or install system dependencies first
# macOS:
xcode-select --install
# Then install prophet
```

#### 3. TensorFlow Issues
```bash
# Install specific TensorFlow version
pip install tensorflow==2.13.0

# For Apple Silicon Macs:
pip install tensorflow-macos==2.13.0
```

#### 4. Frontend Build Issues
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### API Errors
- Check that Python API is running on port 5001
- Verify CSV format matches expected columns
- Ensure date formats are consistent

## Model Recommendations

### When to Use Each Model:

1. **SARIMA** (Recommended)
   - Regular seasonal patterns
   - Historical data with clear trends
   - Most reliable for demand planning

2. **Prophet**
   - Data with trend changes
   - Missing data points
   - Holiday effects

3. **LSTM**
   - Complex non-linear patterns
   - Large datasets (100+ data points)
   - Experimental use cases

## API Keys

### Groq AI Assistant (Optional)
- Sign up at [Groq Console](https://console.groq.com)
- Get your API key
- Add it in the AI Assistant tab for enhanced insights

## Sample Data Format

### Main Dataset (Sales Data)
```csv
Order Date,Item Groups,Qty,Price,Qty - w/Sub-Total,Ext. Price
2023-01-01,8433 3-Panel Shaker Door,10,150.00,10,1500.00
2023-01-02,8402 2-Panel Shaker Door,5,120.00,5,600.00
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review console logs for error details
3. Ensure all dependencies are correctly installed
4. Verify data format matches requirements
