from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.eval_measures import rmse, meanabs
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
from itertools import product
import logging

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Set random seed for consistency
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_response(resp: dict) -> dict:
    """Convert any NaN floats in forecast, metrics, diagnostics to 0.0."""
    # Sanitize forecast entries
    if 'forecast' in resp:
        for p in resp['forecast']:
            for key in ('mean', 'lower', 'upper'):
                val = p.get(key)
                if isinstance(val, float) and math.isnan(val):
                    p[key] = 0.0

    # Sanitize metrics and diagnostics
    for section in ('metrics', 'diagnostics'):
        if section in resp and isinstance(resp[section], dict):
            for k, v in resp[section].items():
                if isinstance(v, float) and math.isnan(v):
                    resp[section][k] = 0.0

    return resp

class MultiModelForecaster:
    """Multi-model forecaster supporting SARIMA, Prophet, and LSTM"""
    
    def __init__(self):
        self.item_configs = {
            '8433 3-Panel Shaker Door': {
                'min_data_points': 12,
                'differencing_threshold': 0.05,
                'seasonal': True,
                'max_order': (3, 1, 3),
                'max_seasonal_order': (2, 1, 2)
            },
            # Add other configurations as needed
        }
        self.default_config = {
            'min_data_points': 12,
            'differencing_threshold': 0.05,
            'seasonal': True,
            'max_order': (2, 1, 2),
            'max_seasonal_order': (1, 1, 1)
        }

    def get_item_config(self, item_group):
        """Get configuration for specific item group"""
        if item_group in self.item_configs:
            return self.item_configs[item_group]
        for key, config in self.item_configs.items():
            if key.lower() in item_group.lower():
                return config
        return self.default_config

    def prepare_data(self, ts_data, freq='M'):
        """Prepare time series data"""
        df = pd.DataFrame(ts_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Resample data
        if freq == 'M':
            df_res = df.resample('MS').agg({'value': 'sum', 'revenue': 'sum'})
            seasonal_period = 12
        else:
            df_res = df.resample('W-MON').agg({'value': 'sum', 'revenue': 'sum'})
            seasonal_period = 52
            
        df_res = df_res.fillna(0)
        
        # Trim leading/trailing zeros
        nonzero = df_res['value'] > 0
        if nonzero.any():
            first = nonzero.idxmax()
            last = nonzero[::-1].idxmax()
            df_res = df_res.loc[first:last]

        return df_res, seasonal_period

    def forecast_sarima(self, df_res, config, seasonal_period, periods):
        """SARIMA forecasting logic"""
        try:
            series = df_res['value']
            
            # Parameter selection
            if config.get('autoSelect', True):
                order, seasonal = self.grid_search_sarima(series, seasonal_period)
            else:
                order = tuple(config.get('order', [1, 1, 1]))
                seasonal = tuple(config.get('seasonalOrder', [1, 1, 1, seasonal_period]))

            # Fit model
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization='approximate_diffuse'
            )
            fit = model.fit(disp=False, maxiter=1000)

            # Forecast
            fc_res = fit.get_forecast(steps=periods).summary_frame(alpha=0.05)
            fc_res = self.validate_forecast(fc_res)
            
            # Generate forecast dates
            last_date = df_res.index[-1]
            if config.get('frequency') == 'M':
                dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
            else:
                dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=periods, freq='W-MON')
            
            fc_res.index = dates
            
            forecast_data = [
                {
                    'date': idx.strftime('%Y-%m-%d'),
                    'mean': float(row['mean']),
                    'lower': float(row['mean_ci_lower']),
                    'upper': float(row['mean_ci_upper'])
                }
                for idx, row in fc_res.iterrows()
            ]

            # Calculate metrics
            metrics = {
                'aic': float(fit.aic),
                'bic': float(fit.bic),
                'method': 'sarima'
            }

            # Add test metrics if we have test data
            if len(df_res) > config.get('min_data_points', 12):
                train_size = max(int(len(df_res) * 0.8), config.get('min_data_points', 12))
                if train_size < len(df_res):
                    try:
                        test_data = df_res['value'].iloc[train_size:]
                        test_predictions = fit.forecast(steps=len(test_data))
                        
                        mae_val = mean_absolute_error(test_data, test_predictions)
                        mse_val = mean_squared_error(test_data, test_predictions)
                        rmse_val = np.sqrt(mse_val)
                        
                        metrics.update({
                            'mae': float(mae_val),
                            'mse': float(mse_val),
                            'rmse': float(rmse_val)
                        })
                    except Exception as e:
                        logger.warning(f"Could not calculate test metrics: {e}")

            return forecast_data, metrics, {'order': order, 'seasonal_order': seasonal}

        except Exception as e:
            logger.error(f"SARIMA forecasting error: {e}")
            raise

    def forecast_prophet(self, df_res, config, periods):
        """Prophet forecasting logic"""
        try:
            # Prepare data for Prophet
            prophet_df = df_res.reset_index()
            prophet_df = prophet_df.rename(columns={'date': 'ds', 'value': 'y'})
            
            # Remove weekends if needed
            prophet_df = prophet_df[prophet_df['ds'].dt.weekday < 5]
            
            # Initialize Prophet model
            prophet_config = config.get('prophetConfig', {})
            model = Prophet(
                growth=prophet_config.get('growth', 'linear'),
                yearly_seasonality=prophet_config.get('yearlySeasonality', True),
                weekly_seasonality=prophet_config.get('weeklySeasonality', False),
                daily_seasonality=prophet_config.get('dailySeasonality', False),
                seasonality_mode=prophet_config.get('seasonalityMode', 'additive'),
                changepoint_range=prophet_config.get('changePointRange', 0.9),
                changepoint_prior_scale=prophet_config.get('changePointPriorScale', 0.5)
            )
            
            # Add quarterly seasonality if requested
            if prophet_config.get('addQuarterly', False):
                model.add_seasonality(
                    name='quarterly', 
                    period=91.25, 
                    fourier_order=prophet_config.get('quarterlyFourierOrder', 5)
                )
            
            # Fit model
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            
            # Extract forecast data
            forecast_future = forecast.tail(periods)
            forecast_data = [
                {
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'mean': float(max(0, row['yhat'])),
                    'lower': float(max(0, row['yhat_lower'])),
                    'upper': float(max(0, row['yhat_upper']))
                }
                for _, row in forecast_future.iterrows()
            ]

            # Calculate metrics (basic for Prophet)
            metrics = {
                'method': 'prophet',
                'mae': 0.0,
                'rmse': 0.0
            }

            model_params = {
                'growth': prophet_config.get('growth', 'linear'),
                'seasonality_mode': prophet_config.get('seasonalityMode', 'additive')
            }

            return forecast_data, metrics, model_params

        except Exception as e:
            logger.error(f"Prophet forecasting error: {e}")
            raise

    def forecast_lstm(self, df_res, config, periods):
        """LSTM forecasting logic"""
        try:
            # Get LSTM configuration
            lstm_config = config.get('lstmConfig', {})
            window_size = lstm_config.get('windowSize', 12)
            epochs = lstm_config.get('epochs', 50)
            batch_size = lstm_config.get('batchSize', 16)
            lstm_units1 = lstm_config.get('lstmUnits1', 64)
            lstm_units2 = lstm_config.get('lstmUnits2', 32)
            dropout_rate = lstm_config.get('dropoutRate', 0.2)
            bidirectional = lstm_config.get('bidirectional', False)
            validation_split = lstm_config.get('validationSplit', 0.1)

            # Prepare data
            data = df_res[['value']].values
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences
            X, y = self.create_sequences(scaled_data, window_size)
            
            if len(X) < window_size * 2:
                raise ValueError("Insufficient data for LSTM training")
            
            # Build model
            model = Sequential()
            
            if bidirectional:
                model.add(Bidirectional(LSTM(lstm_units1, return_sequences=True), 
                                    input_shape=(window_size, 1)))
                model.add(Dropout(dropout_rate))
                model.add(Bidirectional(LSTM(lstm_units2)))
            else:
                model.add(LSTM(lstm_units1, return_sequences=True, input_shape=(window_size, 1)))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(lstm_units2))
            
            model.add(Dropout(dropout_rate))
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                    validation_split=validation_split, verbose=0)
            
            # Generate forecast
            last_sequence = scaled_data[-window_size:]
            forecast_scaled = []
            current_input = last_sequence.reshape(1, window_size, 1)
            
            for _ in range(periods):
                next_pred = model.predict(current_input, verbose=0)[0]
                forecast_scaled.append(next_pred[0])
                current_input = np.append(current_input[:, 1:, :], [[next_pred]], axis=1)
            
            # Inverse transform
            forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
            forecast_values = scaler.inverse_transform(forecast_scaled).flatten()
            forecast_values = np.clip(forecast_values, 0, None)  # Ensure non-negative
            
            # Generate forecast dates
            last_date = df_res.index[-1]
            if config.get('frequency', 'M') == 'M':
                dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
            else:
                dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=periods, freq='W-MON')
            
            # Create confidence intervals (simple approach)
            std_dev = np.std(df_res['value'])
            forecast_data = []
            for i, (date, value) in enumerate(zip(dates, forecast_values)):
                margin = std_dev * 1.96 * np.sqrt(1 + i/10)  # Increasing uncertainty
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'mean': float(value),
                    'lower': float(max(0, value - margin)),
                    'upper': float(value + margin)
                })

            # Calculate metrics
            metrics = {
                'method': 'lstm',
                'mae': 0.0,
                'rmse': 0.0
            }

            model_params = {
                'window_size': window_size,
                'epochs': epochs,
                'lstm_units': [lstm_units1, lstm_units2],
                'bidirectional': bidirectional
            }

            return forecast_data, metrics, model_params

        except Exception as e:
            logger.error(f"LSTM forecasting error: {e}")
            raise

    def create_sequences(self, data, window_size):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    def validate_forecast(self, df):
        """Validate forecast results"""
        df['mean'] = df['mean'].clip(lower=0)
        df['mean_ci_lower'] = df['mean_ci_lower'].clip(lower=0)
        df['mean_ci_upper'] = df[['mean_ci_upper', 'mean']].max(axis=1)
        return df

    def grid_search_sarima(self, series, seasonal_period, config=None):
        """Grid search for optimal SARIMA parameters"""
        if config is None:
            config = self.default_config
        
        p_range = range(config['max_order'][0] + 1)
        d = self.determine_differencing(series, config)
        q_range = range(config['max_order'][2] + 1)
        
        if config.get('seasonal', False):
            P_range = range(config['max_seasonal_order'][0] + 1)
            D_range = range(config['max_seasonal_order'][1] + 1)
            Q_range = range(config['max_seasonal_order'][2] + 1)
        else:
            P_range = D_range = Q_range = [0]
            seasonal_period = 0

        best_aic = float('inf')
        best_params = (1, d, 1)
        best_seasonal = (0, 0, 0, seasonal_period)
        
        for p, q in product(p_range, q_range):
            for P, D, Q in product(P_range, D_range, Q_range):
                try:
                    model = SARIMAX(
                        series,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fit = model.fit(disp=False, maxiter=200)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_params = (p, d, q)
                        best_seasonal = (P, D, Q, seasonal_period)
                except:
                    continue
        
        return best_params, best_seasonal

    def determine_differencing(self, series, config):
        """Determine optimal differencing order"""
        for i in range(3):
            diff = series if i == 0 else series.diff(i).dropna()
            try:
                result = adfuller(diff.dropna())
                if result[1] < config['differencing_threshold']:
                    return i
            except:
                continue
        return 0

forecaster = MultiModelForecaster()

@app.route('/api/forecast', methods=['POST'])
def generate_forecast():
    try:
        data = request.json
        item_group = data['itemGroup']
        ts = data['timeSeries']
        model_type = data.get('modelType', 'SARIMA')
        freq = data.get('frequency', 'M')
        periods = data.get('periods', 12)
        config = data  # Full config object

        logger.info(f"Generating {model_type} forecast for: {item_group} ({len(ts)} points)")

        # Prepare data
        df_res, seasonal_period = forecaster.prepare_data(ts, freq)
        
        if len(df_res) < 6:
            return jsonify({
                'error': 'Insufficient data for forecasting',
                'message': 'Need at least 6 data points'
            }), 400

        # Route to appropriate forecaster
        if model_type == 'SARIMA':
            forecast_data, metrics, model_params = forecaster.forecast_sarima(
                df_res, config, seasonal_period, periods
            )
        elif model_type == 'Prophet':
            forecast_data, metrics, model_params = forecaster.forecast_prophet(
                df_res, config, periods
            )
        elif model_type == 'LSTM':
            forecast_data, metrics, model_params = forecaster.forecast_lstm(
                df_res, config, periods
            )
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400

        response = {
            'itemGroup': item_group,
            'forecast': forecast_data,
            'metrics': metrics,
            'model_params': model_params,
            'modelType': model_type
        }

        return jsonify(sanitize_response(response))

    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Multi-Model Forecast API is running',
        'version': '3.0',
        'models': ['SARIMA', 'Prophet', 'LSTM']
    })

if __name__ == '__main__':
    print("Starting Multi-Model Forecast API on http://localhost:5001")
    print("Supported models: SARIMA, Prophet, LSTM")
    app.run(debug=True, port=5001, host='0.0.0.0')