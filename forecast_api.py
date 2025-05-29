# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import pandas as pd
# # import numpy as np
# # from datetime import datetime, timedelta
# # from statsmodels.tsa.statespace.sarimax import SARIMAX
# # import warnings
# # import os

# # warnings.filterwarnings('ignore')

# # app = Flask(__name__)
# # CORS(app)  # Enable CORS for all routes

# # # Set random seed for consistency
# # np.random.seed(42)
# # os.environ['PYTHONHASHSEED'] = '42'

# # @app.route('/api/forecast', methods=['POST'])
# # def generate_forecast():
# #     try:
# #         data = request.json
# #         item_group = data['itemGroup']
# #         time_series = data['timeSeries']
# #         frequency = data.get('frequency', 'M')
# #         periods = data.get('periods', 12)
# #         order = tuple(data.get('order', [1, 1, 1]))
# #         seasonal_order = tuple(data.get('seasonalOrder', [1, 1, 1, 12]))
# #         auto_select = data.get('autoSelect', False)
        
# #         print(f"Generating forecast for: {item_group}")
# #         print(f"Frequency: {frequency}, Periods: {periods}")
# #         print(f"Order: {order}, Seasonal Order: {seasonal_order}")
# #         print(f"Time series length: {len(time_series)}")
        
# #         # Convert time series to DataFrame
# #         df = pd.DataFrame(time_series)
# #         df['date'] = pd.to_datetime(df['date'])
# #         df = df.set_index('date')
# #         df = df.sort_index()
        
# #         # Ensure we have a value column
# #         if 'value' not in df.columns:
# #             return jsonify({'error': 'Time series must have a "value" column'}), 400
        
# #         print(f"DataFrame shape before resampling: {df.shape}")
# #         print(f"Date range: {df.index.min()} to {df.index.max()}")
        
# #         # Resample to proper frequency
# #         if frequency == 'M':
# #             # Resample to month start, using sum for aggregation
# #             df_resampled = df.resample('MS').agg({'value': 'sum'})
# #             df_resampled = df_resampled.fillna(0)  # Fill missing months with 0
# #             print(f"Monthly resampled shape: {df_resampled.shape}")
# #         else:
# #             # Weekly resampling
# #             df_resampled = df.resample('W').agg({'value': 'sum'})
# #             df_resampled = df_resampled.fillna(0)
# #             print(f"Weekly resampled shape: {df_resampled.shape}")
        
# #         # Remove rows where value is 0 (no sales)
# #         df_resampled = df_resampled[df_resampled['value'] > 0]
# #         print(f"Shape after removing zeros: {df_resampled.shape}")
        
# #         # Check if we have enough data
# #         min_required = 12 if frequency == 'M' else 26
# #         if len(df_resampled) < min_required:
# #             print(f"Warning: Only {len(df_resampled)} data points available (minimum recommended: {min_required})")
            
# #             # If we have too little data, use a simpler model
# #             if len(df_resampled) < 4:
# #                 # Use simple average for very small datasets
# #                 avg_value = df_resampled['value'].mean()
                
# #                 # Generate simple forecast
# #                 last_date = df_resampled.index[-1]
# #                 forecast_data = []
                
# #                 for i in range(periods):
# #                     if frequency == 'M':
# #                         forecast_date = last_date + pd.DateOffset(months=i+1)
# #                     else:
# #                         forecast_date = last_date + pd.DateOffset(weeks=i+1)
                    
# #                     # Add some random variation
# #                     variation = avg_value * 0.1 * (np.random.random() - 0.5)
# #                     forecast_value = max(0, avg_value + variation)
                    
# #                     forecast_data.append({
# #                         'date': forecast_date.strftime('%Y-%m-%d'),
# #                         'mean': float(forecast_value),
# #                         'lower': float(max(0, forecast_value * 0.8)),
# #                         'upper': float(forecast_value * 1.2)
# #                     })
                
# #                 return jsonify({
# #                     'itemGroup': item_group,
# #                     'forecast': forecast_data,
# #                     'metrics': {
# #                         'mae': 0,
# #                         'rmse': 0,
# #                         'aic': 0
# #                     },
# #                     'warning': 'Insufficient data for SARIMA model, using simple average'
# #                 })
        
# #         # Fit SARIMA model
# #         try:
# #             # For small datasets, use simpler parameters
# #             if len(df_resampled) < 24:
# #                 order = (1, 0, 1)
# #                 seasonal_order = (0, 0, 0, 12) if frequency == 'M' else (0, 0, 0, 52)
            
# #             model = SARIMAX(
# #                 df_resampled['value'],
# #                 order=order,
# #                 seasonal_order=seasonal_order,
# #                 enforce_stationarity=False,
# #                 enforce_invertibility=False,
# #                 initialization='approximate_diffuse',
# #                 simple_differencing=True
# #             )
            
# #             fitted_model = model.fit(
# #                 disp=False, 
# #                 maxiter=1000,
# #                 method='lbfgs',
# #                 low_memory=True
# #             )
            
# #             print(f"Model AIC: {fitted_model.aic:.2f}")
            
# #             # Generate forecast
# #             forecast_result = fitted_model.get_forecast(steps=periods)
# #             forecast_df = forecast_result.summary_frame(alpha=0.05)
            
# #             # Create forecast dates
# #             last_date = df_resampled.index[-1]
# #             if frequency == 'M':
# #                 forecast_dates = pd.date_range(
# #                     start=last_date + pd.DateOffset(months=1),
# #                     periods=periods,
# #                     freq='MS'
# #                 )
# #             else:
# #                 forecast_dates = pd.date_range(
# #                     start=last_date + pd.DateOffset(weeks=1),
# #                     periods=periods,
# #                     freq='W'
# #                 )
            
# #             forecast_df.index = forecast_dates
            
# #             # Format response
# #             forecast_data = []
# #             for idx, row in forecast_df.iterrows():
# #                 forecast_data.append({
# #                     'date': idx.strftime('%Y-%m-%d'),
# #                     'mean': float(max(0, row['mean'])),  # Ensure non-negative
# #                     'lower': float(max(0, row['mean_ci_lower'])),
# #                     'upper': float(max(0, row['mean_ci_upper']))
# #                 })
            
# #             # Calculate metrics on historical data
# #             train_size = int(len(df_resampled) * 0.8)
# #             mae = 0
# #             rmse = 0
            
# #             if train_size < len(df_resampled) and train_size >= 4:
# #                 train_data = df_resampled.iloc[:train_size]
# #                 test_data = df_resampled.iloc[train_size:]
                
# #                 try:
# #                     # Refit on training data
# #                     train_model = SARIMAX(
# #                         train_data['value'],
# #                         order=order,
# #                         seasonal_order=seasonal_order,
# #                         enforce_stationarity=False,
# #                         enforce_invertibility=False
# #                     ).fit(disp=False)
                    
# #                     predictions = train_model.forecast(steps=len(test_data))
# #                     mae = float(np.mean(np.abs(test_data['value'] - predictions)))
# #                     rmse = float(np.sqrt(np.mean((test_data['value'] - predictions) ** 2)))
# #                 except Exception as e:
# #                     print(f"Could not calculate test metrics: {e}")
            
# #             response = {
# #                 'itemGroup': item_group,
# #                 'forecast': forecast_data,
# #                 'metrics': {
# #                     'mae': mae,
# #                     'rmse': rmse,
# #                     'aic': float(fitted_model.aic)
# #                 }
# #             }
            
# #             print(f"Successfully generated forecast for {item_group}")
# #             return jsonify(response)
            
# #         except Exception as e:
# #             print(f"Model fitting error: {str(e)}")
            
# #             # Fallback to simple forecast
# #             avg_value = df_resampled['value'].mean()
# #             last_date = df_resampled.index[-1]
# #             forecast_data = []
            
# #             for i in range(periods):
# #                 if frequency == 'M':
# #                     forecast_date = last_date + pd.DateOffset(months=i+1)
# #                 else:
# #                     forecast_date = last_date + pd.DateOffset(weeks=i+1)
                
# #                 forecast_value = avg_value
# #                 forecast_data.append({
# #                     'date': forecast_date.strftime('%Y-%m-%d'),
# #                     'mean': float(forecast_value),
# #                     'lower': float(forecast_value * 0.8),
# #                     'upper': float(forecast_value * 1.2)
# #                 })
            
# #             return jsonify({
# #                 'itemGroup': item_group,
# #                 'forecast': forecast_data,
# #                 'metrics': {
# #                     'mae': 0,
# #                     'rmse': 0,
# #                     'aic': 0
# #                 },
# #                 'warning': f'SARIMA failed, using simple average: {str(e)}'
# #             })
            
# #     except Exception as e:
# #         print(f"API error: {str(e)}")
# #         import traceback
# #         traceback.print_exc()
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/api/health', methods=['GET'])
# # def health_check():
# #     return jsonify({'status': 'healthy', 'message': 'Forecast API is running'})

# # if __name__ == '__main__':
# #     print("Starting Forecast API on http://localhost:5001")
# #     app.run(debug=True, port=5001, host='0.0.0.0')


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.stattools import adfuller, acf, pacf
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tools.eval_measures import rmse, meanabs
# import warnings
# import os
# from itertools import product
# import logging
# import math 

# warnings.filterwarnings('ignore')

# app = Flask(__name__)
# CORS(app)

# # Set random seed for consistency
# np.random.seed(42)
# os.environ['PYTHONHASHSEED'] = '42'

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SARIMAForecaster:
#     """Enhanced SARIMA Forecaster with automatic parameter selection and item-specific configurations"""
    
#     def __init__(self):
#         # Item-specific configurations based on demand patterns
#         self.item_configs = {
#             # High-volume, stable products
#             '8433 3-Panel Shaker Door': {
#                 'min_data_points': 12,
#                 'differencing_threshold': 0.05,
#                 'seasonal': True,
#                 'max_order': (3, 1, 3),
#                 'max_seasonal_order': (2, 1, 2)
#             },
#             '8730 Plantation Primed Louver/Louver 6/8 & 7/0 Only': {
#                 'min_data_points': 12,
#                 'differencing_threshold': 0.05,
#                 'seasonal': True,
#                 'max_order': (2, 1, 2),
#                 'max_seasonal_order': (1, 1, 1)
#             },
#             # High volatility products
#             '8402 2-Panel Shaker Door': {
#                 'min_data_points': 18,
#                 'differencing_threshold': 0.01,
#                 'seasonal': True,
#                 'max_order': (2, 1, 2),
#                 'max_seasonal_order': (1, 0, 1),
#                 'transform': 'log'  # Log transform for high volatility
#             },
#             # Low volume products
#             '8401BZ Fir with Z-Bar': {
#                 'min_data_points': 6,
#                 'differencing_threshold': 0.10,
#                 'seasonal': False,
#                 'max_order': (1, 0, 1),
#                 'max_seasonal_order': (0, 0, 0)
#             }
#         }
        
#         # Default configuration for unknown items
#         self.default_config = {
#             'min_data_points': 12,
#             'differencing_threshold': 0.05,
#             'seasonal': True,
#             'max_order': (2, 1, 2),
#             'max_seasonal_order': (1, 1, 1)
#         }
    
#     def get_item_config(self, item_group):
#         """Get configuration for specific item group"""
#         # Check for exact match
#         if item_group in self.item_configs:
#             return self.item_configs[item_group]
        
#         # Check for partial matches
#         for key, config in self.item_configs.items():
#             if key.lower() in item_group.lower() or item_group.lower() in key.lower():
#                 return config
        
#         # Determine config based on item characteristics
#         if 'primed' in item_group.lower():
#             return self.default_config
#         elif any(term in item_group.lower() for term in ['bifold', 'barndoor', 'custom']):
#             # Irregular items need simpler models
#             return {
#                 'min_data_points': 6,
#                 'differencing_threshold': 0.10,
#                 'seasonal': False,
#                 'max_order': (1, 1, 1),
#                 'max_seasonal_order': (0, 0, 0)
#             }
        
#         return self.default_config
    
#     def check_stationarity(self, series, alpha=0.05):
#         """Enhanced stationarity test with multiple methods"""
#         try:
#             # Remove NaN and zero values
#             clean_series = series.dropna()
#             clean_series = clean_series[clean_series > 0]
            
#             if len(clean_series) < 10:
#                 return False, 0
            
#             # ADF test
#             result = adfuller(clean_series, autolag='AIC')
#             is_stationary = result[1] < alpha
            
#             return is_stationary, result[1]
#         except:
#             return False, 1.0
    
#     def determine_differencing(self, series, config):
#         """Determine optimal differencing order"""
#         d = 0
#         max_diff = 2
        
#         for i in range(max_diff + 1):
#             if i == 0:
#                 diff_series = series
#             else:
#                 diff_series = series.diff(i).dropna()
            
#             is_stationary, p_value = self.check_stationarity(diff_series)
            
#             if is_stationary or p_value < config['differencing_threshold']:
#                 d = i
#                 break
        
#         return d
    
#     def grid_search_sarima(self, series, seasonal_period=12, config=None):
#         """Enhanced grid search with item-specific configurations"""
#         if config is None:
#             config = self.default_config
        
#         # Define search ranges based on config
#         p_range = range(0, config['max_order'][0] + 1)
#         d_range = [self.determine_differencing(series, config)]
#         q_range = range(0, config['max_order'][2] + 1)
        
#         if config['seasonal']:
#             P_range = range(0, config['max_seasonal_order'][0] + 1)
#             D_range = range(0, config['max_seasonal_order'][1] + 1)
#             Q_range = range(0, config['max_seasonal_order'][2] + 1)
#         else:
#             P_range = [0]
#             D_range = [0]
#             Q_range = [0]
#             seasonal_period = 0
        
#         best_aic = float('inf')
#         best_params = None
#         best_seasonal_params = None
#         results = []
        
#         # Grid search
#         for p, d, q in product(p_range, d_range, q_range):
#             for P, D, Q in product(P_range, D_range, Q_range):
#                 if p == 0 and q == 0:
#                     continue
                
#                 try:
#                     model = SARIMAX(
#                         series,
#                         order=(p, d, q),
#                         seasonal_order=(P, D, Q, seasonal_period) if seasonal_period > 0 else (0, 0, 0, 0),
#                         enforce_stationarity=False,
#                         enforce_invertibility=False,
#                         initialization='approximate_diffuse'
#                     )
                    
#                     fitted = model.fit(disp=False, maxiter=200)
                    
#                     # Calculate metrics
#                     aic = fitted.aic
#                     bic = fitted.bic
                    
#                     results.append({
#                         'order': (p, d, q),
#                         'seasonal_order': (P, D, Q, seasonal_period),
#                         'aic': aic,
#                         'bic': bic
#                     })
                    
#                     if aic < best_aic:
#                         best_aic = aic
#                         best_params = (p, d, q)
#                         best_seasonal_params = (P, D, Q, seasonal_period)
                    
#                 except Exception as e:
#                     continue
        
#         # If no valid model found, use simple defaults
#         if best_params is None:
#             best_params = (1, d_range[0], 1)
#             best_seasonal_params = (0, 0, 0, seasonal_period) if seasonal_period > 0 else (0, 0, 0, 0)
        
#         logger.info(f"Best parameters - Order: {best_params}, Seasonal: {best_seasonal_params}, AIC: {best_aic:.2f}")
        
#         return best_params, best_seasonal_params, results
    
#     def apply_transformation(self, series, transform_type):
#         """Apply data transformation"""
#         if transform_type == 'log':
#             # Add small constant to avoid log(0)
#             return np.log1p(series)
#         elif transform_type == 'sqrt':
#             return np.sqrt(series)
#         elif transform_type == 'box-cox':
#             from scipy import stats
#             transformed, _ = stats.boxcox(series + 1)
#             return pd.Series(transformed, index=series.index)
#         return series
    
#     def inverse_transformation(self, series, transform_type):
#         """Inverse transformation"""
#         if transform_type == 'log':
#             return np.expm1(series)
#         elif transform_type == 'sqrt':
#             return np.square(series)
#         return series
    
#     def validate_forecast(self, forecast_df):
#         """Ensure forecast values are reasonable"""
#         # Ensure non-negative values
#         forecast_df['mean'] = forecast_df['mean'].clip(lower=0)
#         forecast_df['mean_ci_lower'] = forecast_df['mean_ci_lower'].clip(lower=0)
        
#         # Ensure upper bound is greater than lower bound
#         forecast_df['mean_ci_upper'] = forecast_df[['mean_ci_upper', 'mean']].max(axis=1)
        
#         return forecast_df
    
#     def calculate_advanced_metrics(self, actual, predicted):
#         """Calculate comprehensive forecast metrics"""
#         # Ensure both series have the same length and convert to numpy arrays
#         actual_values = np.array(actual)
#         predicted_values = np.array(predicted)
        
#         # Basic metrics
#         mae = meanabs(actual_values, predicted_values)
#         rmse_val = rmse(actual_values, predicted_values)
        
#         # MAPE - handle zero values
#         non_zero_mask = actual_values != 0
#         if np.any(non_zero_mask):
#             mape = np.mean(np.abs((actual_values[non_zero_mask] - predicted_values[non_zero_mask]) / actual_values[non_zero_mask])) * 100
#         else:
#             mape = 0
        
#         # Directional accuracy
#         if len(actual_values) > 1:
#             actual_diff = np.diff(actual_values)
#             pred_diff = np.diff(predicted_values)
#             direction_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
#         else:
#             direction_accuracy = 0
        
#         return {
#             'mae': float(mae),
#             'rmse': float(rmse_val),
#             'mape': float(mape),
#             'direction_accuracy': float(direction_accuracy)
#         }

# forecaster = SARIMAForecaster()

# @app.route('/api/forecast', methods=['POST'])
# def generate_forecast():
#     try:
#         data = request.json
#         item_group = data['itemGroup']
#         time_series = data['timeSeries']
#         frequency = data.get('frequency', 'M')
#         periods = data.get('periods', 12)
#         auto_select = data.get('autoSelect', True)
        
#         logger.info(f"Generating forecast for: {item_group}")
#         logger.info(f"Time series length: {len(time_series)}")
        
#         # Get item-specific configuration
#         config = forecaster.get_item_config(item_group)
#         logger.info(f"Using configuration: {config}")
        
#         # Convert to DataFrame
#         df = pd.DataFrame(time_series)
#         df['date'] = pd.to_datetime(df['date'])
#         df = df.set_index('date')
#         df = df.sort_index()
        
#         # Aggregate by frequency
#         if frequency == 'M':
#             df_resampled = df.resample('MS').agg({
#                 'value': 'sum',
#                 'revenue': 'sum' if 'revenue' in df.columns else 'first'
#             })
#             seasonal_period = 12
#         else:
#             df_resampled = df.resample('W-MON').agg({
#                 'value': 'sum',
#                 'revenue': 'sum' if 'revenue' in df.columns else 'first'
#             })
#             seasonal_period = 52
        
#         # Fill missing periods with 0
#         df_resampled = df_resampled.fillna(0)
        
#         # Remove leading and trailing zeros
#         first_nonzero = (df_resampled['value'] > 0).idxmax()
#         last_nonzero = (df_resampled['value'] > 0)[::-1].idxmax()
#         df_resampled = df_resampled.loc[first_nonzero:last_nonzero]
        
#         logger.info(f"Resampled data shape: {df_resampled.shape}")
#         logger.info(f"Date range: {df_resampled.index.min()} to {df_resampled.index.max()}")
#         logger.info(f"Value range: {df_resampled['value'].min():.2f} to {df_resampled['value'].max():.2f}")
        
#         # Check minimum data requirements
#         if len(df_resampled) < config['min_data_points']:
#             logger.warning(f"Insufficient data: {len(df_resampled)} points (need {config['min_data_points']})")
            
#             # Use simple forecasting for insufficient data
#             if len(df_resampled) >= 3:
#                 # Use exponential smoothing
#                 from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
#                 try:
#                     model = ExponentialSmoothing(
#                         df_resampled['value'],
#                         seasonal_periods=seasonal_period if len(df_resampled) >= seasonal_period else None,
#                         trend='add' if len(df_resampled) >= 6 else None,
#                         seasonal='add' if len(df_resampled) >= seasonal_period * 2 else None
#                     )
#                     fitted = model.fit()
#                     forecast_values = fitted.forecast(periods)
                    
#                     # Create forecast dates
#                     last_date = df_resampled.index[-1]
#                     if frequency == 'M':
#                         forecast_dates = pd.date_range(
#                             start=last_date + pd.DateOffset(months=1),
#                             periods=periods,
#                             freq='MS'
#                         )
#                     else:
#                         forecast_dates = pd.date_range(
#                             start=last_date + pd.DateOffset(weeks=1),
#                             periods=periods,
#                             freq='W-MON'
#                         )
                    
#                     forecast_data = []
#                     for i, (date, value) in enumerate(zip(forecast_dates, forecast_values)):
#                         # Simple confidence intervals
#                         std = df_resampled['value'].std()
#                         margin = std * 1.96 * np.sqrt(1 + i/10)  # Increasing uncertainty
                        
#                         forecast_data.append({
#                             'date': date.strftime('%Y-%m-%d'),
#                             'mean': float(max(0, value)),
#                             'lower': float(max(0, value - margin)),
#                             'upper': float(value + margin)
#                         })
                    
#                     return jsonify({
#                         'itemGroup': item_group,
#                         'forecast': forecast_data,
#                         'metrics': {
#                             'mae': 0,
#                             'rmse': 0,
#                             'aic': 0,
#                             'method': 'exponential_smoothing'
#                         },
#                         'warning': 'Used exponential smoothing due to limited data'
#                     })
                    
#                 except Exception as e:
#                     logger.error(f"Exponential smoothing failed: {e}")
            
#             # Fallback to naive forecast
#             avg_value = df_resampled['value'].mean()
#             std_value = df_resampled['value'].std()
            
#             forecast_data = []
#             last_date = df_resampled.index[-1]
            
#             for i in range(periods):
#                 if frequency == 'M':
#                     forecast_date = last_date + pd.DateOffset(months=i+1)
#                 else:
#                     forecast_date = last_date + pd.DateOffset(weeks=i+1)
                
#                 # Add slight trend and seasonality
#                 trend_factor = 1 + (i * 0.01)  # 1% growth per period
#                 seasonal_factor = 1.0
                
#                 if config['seasonal'] and len(df_resampled) >= seasonal_period:
#                     # Simple seasonal pattern
#                     month = forecast_date.month if frequency == 'M' else forecast_date.week
#                     historical_month_avg = df_resampled[df_resampled.index.month == month]['value'].mean() if frequency == 'M' else avg_value
#                     if historical_month_avg > 0:
#                         seasonal_factor = historical_month_avg / avg_value
                
#                 forecast_value = avg_value * trend_factor * seasonal_factor
#                 margin = std_value * 1.96 * np.sqrt(1 + i/10)
                
#                 forecast_data.append({
#                     'date': forecast_date.strftime('%Y-%m-%d'),
#                     'mean': float(max(0, forecast_value)),
#                     'lower': float(max(0, forecast_value - margin)),
#                     'upper': float(forecast_value + margin)
#                 })
            
#             return jsonify({
#                 'itemGroup': item_group,
#                 'forecast': forecast_data,
#                 'metrics': {
#                     'mae': 0,
#                     'rmse': 0,
#                     'aic': 0,
#                     'method': 'naive_seasonal'
#                 },
#                 'warning': 'Insufficient data for SARIMA, used naive seasonal method'
#             })
        
#         # Apply transformation if specified
#         transform_type = config.get('transform', None)
#         if transform_type:
#             series = forecaster.apply_transformation(df_resampled['value'], transform_type)
#         else:
#             series = df_resampled['value']
        
#         # Split into train/test for validation
#         train_size = int(len(series) * 0.8)
#         train_size = max(train_size, config['min_data_points'])
        
#         if train_size < len(series):
#             train_data = series.iloc[:train_size]
#             test_data = series.iloc[train_size:]
#         else:
#             train_data = series
#             test_data = pd.Series()
        
#         # Determine best parameters
#         if auto_select and len(train_data) >= config['min_data_points']:
#             try:
#                 order, seasonal_order, search_results = forecaster.grid_search_sarima(
#                     train_data, 
#                     seasonal_period if config['seasonal'] else 0,
#                     config
#                 )
#             except Exception as e:
#                 logger.error(f"Grid search failed: {e}")
#                 order = (1, 1, 1)
#                 seasonal_order = (1, 1, 1, seasonal_period) if config['seasonal'] else (0, 0, 0, 0)
#         else:
#             # Use provided or default parameters
#             order = tuple(data.get('order', [1, 1, 1]))
#             seasonal_order = tuple(data.get('seasonalOrder', [1, 1, 1, seasonal_period]))
        
#         logger.info(f"Using SARIMA{order}x{seasonal_order}")
        
#         # Fit final model on all data
#         try:
#             model = SARIMAX(
#                 series,
#                 order=order,
#                 seasonal_order=seasonal_order,
#                 enforce_stationarity=False,
#                 enforce_invertibility=False,
#                 initialization='approximate_diffuse',
#                 simple_differencing=True
#             )
            
#             fitted_model = model.fit(
#                 disp=False,
#                 maxiter=1000,
#                 method='lbfgs',
#                 low_memory=True
#             )
            
#             logger.info(f"Model converged: {fitted_model.mle_retvals['converged']}")
#             logger.info(f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
            
#             # In-sample fit quality
#             fitted_values = fitted_model.fittedvalues
#             residuals = series - fitted_values
#             logger.info(f"In-sample RMSE: {np.sqrt(np.mean(residuals**2)):.2f}")
            
#             # Generate forecast
#             forecast_result = fitted_model.get_forecast(steps=periods)
#             forecast_df = forecast_result.summary_frame(alpha=0.05)
            
#             # Apply inverse transformation
#             if transform_type:
#                 forecast_df['mean'] = forecaster.inverse_transformation(forecast_df['mean'], transform_type)
#                 forecast_df['mean_ci_lower'] = forecaster.inverse_transformation(forecast_df['mean_ci_lower'], transform_type)
#                 forecast_df['mean_ci_upper'] = forecaster.inverse_transformation(forecast_df['mean_ci_upper'], transform_type)
            
#             # Validate forecast
#             forecast_df = forecaster.validate_forecast(forecast_df)
            
#             # Create forecast dates
#             last_date = df_resampled.index[-1]
#             if frequency == 'M':
#                 forecast_dates = pd.date_range(
#                     start=last_date + pd.DateOffset(months=1),
#                     periods=periods,
#                     freq='MS'
#                 )
#             else:
#                 forecast_dates = pd.date_range(
#                     start=last_date + pd.DateOffset(weeks=1),
#                     periods=periods,
#                     freq='W-MON'
#                 )
            
#             forecast_df.index = forecast_dates
            
#             # Format response
#             forecast_data = []
#             for idx, row in forecast_df.iterrows():
#                 forecast_data.append({
#                     'date': idx.strftime('%Y-%m-%d'),
#                     'mean': float(row['mean']),
#                     'lower': float(row['mean_ci_lower']),
#                     'upper': float(row['mean_ci_upper'])
#                 })
            
#             # Calculate metrics if we have test data
#             metrics = {
#                 'aic': float(fitted_model.aic),
#                 'bic': float(fitted_model.bic),
#                 'method': 'sarima'
#             }
            
#             if len(test_data) > 0:
#                 try:
#                     # Generate predictions for test period
#                     test_forecast = fitted_model.forecast(steps=len(test_data))
                    
#                     # Apply inverse transformation if needed
#                     if transform_type:
#                         test_forecast = forecaster.inverse_transformation(test_forecast, transform_type)
#                         test_actual = forecaster.inverse_transformation(test_data, transform_type)
#                     else:
#                         test_actual = test_data
                    
#                     # Calculate metrics
#                     advanced_metrics = forecaster.calculate_advanced_metrics(test_actual, test_forecast)
#                     metrics.update(advanced_metrics)
#                 except Exception as e:
#                     logger.error(f"Error calculating test metrics: {e}")
#                     metrics.update({'mae': 0, 'rmse': 0, 'mape': 0, 'direction_accuracy': 0})
#             else:
#                 metrics.update({'mae': 0, 'rmse': 0, 'mape': 0, 'direction_accuracy': 0})
            
#             # Add model diagnostics
#             try:
#                 diagnostics = {
#                     'ljung_box_pvalue': float(fitted_model.test_serial_correlation('ljungbox')[0][1][-1]),
#                     'jarque_bera_pvalue': float(fitted_model.test_normality('jarquebera')[0][1]),
#                     'heteroscedasticity_pvalue': float(fitted_model.test_heteroskedasticity('breakvar')[0][1])
#                 }
#             except Exception as e:
#                 logger.error(f"Error calculating diagnostics: {e}")
#                 diagnostics = {
#                     'ljung_box_pvalue': 0,
#                     'jarque_bera_pvalue': 0,
#                     'heteroscedasticity_pvalue': 0
#                 }
            
#             response = {
#                 'itemGroup': item_group,
#                 'forecast': forecast_data,
#                 'metrics': metrics,
#                 'diagnostics': diagnostics,
#                 'model_params': {
#                     'order': order,
#                     'seasonal_order': seasonal_order,
#                     'transform': transform_type
#                 }
#             }
            
#             logger.info(f"Successfully generated forecast for {item_group}")
#             return jsonify(response)
            
#         except Exception as e:
#             logger.error(f"SARIMA fitting error: {str(e)}")
#             import traceback
#             traceback.print_exc()
            
#             # Return error with details
#             return jsonify({
#                 'error': f'Model fitting failed: {str(e)}',
#                 'itemGroup': item_group,
#                 'details': {
#                     'data_points': len(series),
#                     'date_range': f"{df_resampled.index.min()} to {df_resampled.index.max()}",
#                     'value_range': f"{series.min():.2f} to {series.max():.2f}"
#                 }
#             }), 400
            
#     except Exception as e:
#         logger.error(f"API error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({
#         'status': 'healthy',
#         'message': 'Enhanced SARIMA Forecast API is running',
#         'version': '2.0'
#     })

# if __name__ == '__main__':
#     print("Starting Enhanced SARIMA Forecast API on http://localhost:5001")
#     app.run(debug=True, port=5001, host='0.0.0.0')


# forecast_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import rmse, meanabs
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

class SARIMAForecaster:
    """Enhanced SARIMA Forecaster with automatic parameter selection and item-specific configurations"""
    def __init__(self):
        # Item-specific configurations based on demand patterns
        self.item_configs = {
            '8433 3-Panel Shaker Door': {
                'min_data_points': 12,
                'differencing_threshold': 0.05,
                'seasonal': True,
                'max_order': (3, 1, 3),
                'max_seasonal_order': (2, 1, 2)
            },
            '8730 Plantation Primed Louver/Louver 6/8 & 7/0 Only': {
                'min_data_points': 12,
                'differencing_threshold': 0.05,
                'seasonal': True,
                'max_order': (2, 1, 2),
                'max_seasonal_order': (1, 1, 1)
            },
            '8402 2-Panel Shaker Door': {
                'min_data_points': 18,
                'differencing_threshold': 0.01,
                'seasonal': True,
                'max_order': (2, 1, 2),
                'max_seasonal_order': (1, 0, 1),
                'transform': 'log'
            },
            '8401BZ Fir with Z-Bar': {
                'min_data_points': 6,
                'differencing_threshold': 0.10,
                'seasonal': False,
                'max_order': (1, 0, 1),
                'max_seasonal_order': (0, 0, 0)
            }
        }
        # Default configuration
        self.default_config = {
            'min_data_points': 12,
            'differencing_threshold': 0.05,
            'seasonal': True,
            'max_order': (2, 1, 2),
            'max_seasonal_order': (1, 1, 1)
        }

    def get_item_config(self, item_group):
        if item_group in self.item_configs:
            return self.item_configs[item_group]
        for key, config in self.item_configs.items():
            if key.lower() in item_group.lower():
                return config
        return self.default_config

    def check_stationarity(self, series, alpha=0.05):
        try:
            clean = series.dropna()[series > 0]
            if len(clean) < 10:
                return False, 1.0
            result = adfuller(clean, autolag='AIC')
            return (result[1] < alpha), result[1]
        except:
            return False, 1.0

    def determine_differencing(self, series, config):
        for i in range(3):
            diff = series if i == 0 else series.diff(i).dropna()
            stationary, pval = self.check_stationarity(diff, alpha=config['differencing_threshold'])
            if stationary or pval < config['differencing_threshold']:
                return i
        return 0

    def grid_search_sarima(self, series, seasonal_period=12, config=None):
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
                        enforce_invertibility=False,
                        initialization='approximate_diffuse'
                    )
                    fit = model.fit(disp=False, maxiter=200)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_params = (p, d, q)
                        best_seasonal = (P, D, Q, seasonal_period)
                except:
                    continue
        logger.info(f"Best SARIMA params: {best_params}x{best_seasonal}, AIC={best_aic:.2f}")
        return best_params, best_seasonal

    def apply_transformation(self, series, transform):
        if transform == 'log':
            return np.log1p(series)
        return series

    def inverse_transformation(self, series, transform):
        if transform == 'log':
            return np.expm1(series)
        return series

    def validate_forecast(self, df):
        df['mean'] = df['mean'].clip(lower=0)
        df['mean_ci_lower'] = df['mean_ci_lower'].clip(lower=0)
        df['mean_ci_upper'] = df[['mean_ci_upper', 'mean']].max(axis=1)
        return df

    def calculate_advanced_metrics(self, actual, predicted):
        a = np.array(actual)
        p = np.array(predicted)
        mae_val = meanabs(a, p)
        rmse_val = rmse(a, p)
        # MAPE
        nz = a != 0
        mape = np.mean(np.abs((a[nz] - p[nz]) / a[nz]))*100 if nz.any() else 0
        # Directional accuracy
        if len(a) > 1:
            dir_acc = np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100
        else:
            dir_acc = 0
        return {'mae': float(mae_val), 'rmse': float(rmse_val), 'mape': float(mape), 'direction_accuracy': float(dir_acc)}

forecaster = SARIMAForecaster()

@app.route('/api/forecast', methods=['POST'])
def generate_forecast():
    try:
        data = request.json
        item_group = data['itemGroup']
        ts = data['timeSeries']
        freq = data.get('frequency', 'M')
        periods = data.get('periods', 12)
        auto = data.get('autoSelect', True)

        logger.info(f"Generating forecast for: {item_group} ({len(ts)} points)")
        config = forecaster.get_item_config(item_group)

        # Build DataFrame
        df = pd.DataFrame(ts)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Resample
        if freq == 'M':
            df_res = df.resample('MS').agg({'value':'sum', 'revenue':'sum'})
            seasonal_period = 12
        else:
            df_res = df.resample('W-MON').agg({'value':'sum', 'revenue':'sum'})
            seasonal_period = 52
        df_res = df_res.fillna(0)
        # Trim leading/trailing zeros
        nonzero = df_res['value']>0
        if nonzero.any():
            first = nonzero.idxmax()
            last = nonzero[::-1].idxmax()
            df_res = df_res.loc[first:last]

        logger.info(f"Resampled shape: {df_res.shape}")

        # If insufficient data, exponential smoothing branch
        if len(df_res) < config['min_data_points']:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            try:
                model = ExponentialSmoothing(
                    df_res['value'],
                    seasonal_periods=seasonal_period if len(df_res)>=seasonal_period else None,
                    trend='add' if len(df_res)>=6 else None,
                    seasonal='add' if len(df_res)>=seasonal_period*2 else None
                )
                fit = model.fit()
                preds = fit.forecast(periods)
                last_date = df_res.index[-1]
                dates = (
                    pd.date_range(start=last_date+pd.DateOffset(months=1), periods=periods, freq='MS')
                    if freq=='M'
                    else pd.date_range(start=last_date+pd.DateOffset(weeks=1), periods=periods, freq='W-MON')
                )
                forecast_data = []
                std = df_res['value'].std()
                for i,(date,val) in enumerate(zip(dates,preds)):
                    margin = std * 1.96 * np.sqrt(1 + i/10)
                    forecast_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'mean': float(max(0,val)),
                        'lower': float(max(0,val-margin)),
                        'upper': float(val+margin)
                    })
                resp = {
                    'itemGroup': item_group,
                    'forecast': forecast_data,
                    'metrics': {'mae':0.0,'rmse':0.0,'aic':0.0,'method':'exponential_smoothing'},
                    'warning':'Used exponential smoothing due to limited data'
                }
                return jsonify(sanitize_response(resp))
            except Exception as e:
                logger.error(f"Exponential smoothing failed: {e}")

            # fallback naive seasonal
            avg = df_res['value'].mean()
            std = df_res['value'].std()
            last_date = df_res.index[-1]
            forecast_data = []
            for i in range(periods):
                date = (
                    last_date + pd.DateOffset(months=i+1) if freq=='M'
                    else last_date + pd.DateOffset(weeks=i+1)
                )
                trend = 1 + i*0.01
                val = avg * trend
                margin = std * 1.96 * np.sqrt(1 + i/10)
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'mean': float(max(0,val)),
                    'lower': float(max(0,val-margin)),
                    'upper': float(val+margin)
                })
            resp = {
                'itemGroup': item_group,
                'forecast': forecast_data,
                'metrics': {'mae':0.0,'rmse':0.0,'aic':0.0,'method':'naive_seasonal'},
                'warning':'Insufficient data for SARIMA, used naive seasonal'
            }
            return jsonify(sanitize_response(resp))

        # Otherwise SARIMA branch
        # Apply transform if needed
        transform = config.get('transform')
        series = forecaster.apply_transformation(df_res['value'], transform) if transform else df_res['value']

        # Split train/test
        train_size = max(int(len(series)*0.8), config['min_data_points'])
        if train_size < len(series):
            train, test = series.iloc[:train_size], series.iloc[train_size:]
        else:
            train, test = series, pd.Series()

        # Parameter selection
        if auto and len(train) >= config['min_data_points']:
            order, seasonal = forecaster.grid_search_sarima(train, seasonal_period, config)
        else:
            order = tuple(data.get('order',[1,1,1]))
            seasonal = tuple(data.get('seasonalOrder',[1,1,1,seasonal_period]))

        logger.info(f"Fitting SARIMA{order}x{seasonal}")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
            initialization='approximate_diffuse'
        )
        fit = model.fit(disp=False, maxiter=1000, method='lbfgs', low_memory=True)

        # Forecast
        fc_res = fit.get_forecast(steps=periods).summary_frame(alpha=0.05)
        if transform:
            fc_res['mean'] = forecaster.inverse_transformation(fc_res['mean'], transform)
            fc_res['mean_ci_lower'] = forecaster.inverse_transformation(fc_res['mean_ci_lower'], transform)
            fc_res['mean_ci_upper'] = forecaster.inverse_transformation(fc_res['mean_ci_upper'], transform)
        fc_res = forecaster.validate_forecast(fc_res)

        last = df_res.index[-1]
        dates = (
            pd.date_range(start=last+pd.DateOffset(months=1), periods=periods, freq='MS')
            if freq=='M'
            else pd.date_range(start=last+pd.DateOffset(weeks=1), periods=periods, freq='W-MON')
        )
        fc_res.index = dates

        forecast_data = [
            {'date': idx.strftime('%Y-%m-%d'),
            'mean': float(row['mean']),
            'lower': float(row['mean_ci_lower']),
            'upper': float(row['mean_ci_upper'])}
            for idx, row in fc_res.iterrows()
        ]

        # Metrics
        metrics = {'aic':float(fit.aic), 'bic':float(fit.bic), 'method':'sarima'}
        if not test.empty:
            try:
                preds = fit.forecast(steps=len(test))
                if transform:
                    preds = forecaster.inverse_transformation(preds, transform)
                    actual = forecaster.inverse_transformation(test, transform)
                else:
                    actual = test
                adv = forecaster.calculate_advanced_metrics(actual, preds)
                metrics.update(adv)
            except Exception as e:
                logger.error(f"Error computing test metrics: {e}")
                metrics.update({'mae':0.0,'rmse':0.0,'mape':0.0,'direction_accuracy':0.0})

        # Diagnostics
        try:
            diags = {
                'ljung_box_pvalue': float(fit.test_serial_correlation('ljungbox')[0][1][-1]),
                'jarque_bera_pvalue': float(fit.test_normality('jarquebera')[0][1]),
                'heteroscedasticity_pvalue': float(fit.test_heteroskedasticity('breakvar')[0][1])
            }
        except:
            diags = {'ljung_box_pvalue':0.0,'jarque_bera_pvalue':0.0,'heteroscedasticity_pvalue':0.0}

        response = {
            'itemGroup': item_group,
            'forecast': forecast_data,
            'metrics': metrics,
            'diagnostics': diags,
            'model_params': {'order':order,'seasonal_order':seasonal,'transform':transform}
        }

        return jsonify(sanitize_response(response))

    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status':'healthy','message':'Forecast API is running','version':'2.0'})

if __name__ == '__main__':
    print("Starting Forecast API on http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')
