# import pandas as pd
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import itertools
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np
# import mlflow
# import mlflow.sklearn

# class SeasonalARIMAForecaster:
#     def __init__(self, file_path, date_column='Bill Date', value_column='Net Amount', forecast_steps=15):
#         self.file_path = file_path
#         self.date_column = date_column
#         self.value_column = value_column
#         self.forecast_steps = forecast_steps

#         self.df = pd.read_csv(self.file_path, parse_dates=[self.date_column])
#         self.df.set_index(self.date_column, inplace=True)
#         self.df[self.value_column] = pd.to_numeric(self.df[self.value_column], errors='coerce')
#         self.df.dropna(subset=[self.value_column], inplace=True)
#         self.y = self.df[self.value_column]

#         # Set the remote server URI to your EC2 instance where MLflow is hosted
#         self.remote_server_uri = "http://ec2-13-203-206-18.ap-south-1.compute.amazonaws.com:5000/"
#         mlflow.set_tracking_uri(self.remote_server_uri)

#     def generate_parameter_combinations(self):
#         """Generates parameter combinations for SARIMA model."""
#         p = d = q = range(0, 2)
#         pdq = list(itertools.product(p, d, q)) 
#         seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]  # Monthly seasonality
#         return pdq, seasonal_pdq

#     def grid_search(self, pdq, seasonal_pdq):
#         """Performs grid search to find the best SARIMA model."""
#         print('Examples of parameter combinations for Seasonal ARIMA...')
#         for param in pdq:
#             for param_seasonal in seasonal_pdq:
#                 try:
#                     mod = sm.tsa.statespace.SARIMAX(self.y,
#                                                     order=param,
#                                                     seasonal_order=param_seasonal,
#                                                     enforce_stationarity=False,
#                                                     enforce_invertibility=False)
#                     results = mod.fit()
#                     print(f'ARIMA{param}x{param_seasonal}12 - AIC: {results.aic}')
#                 except Exception as e:
#                     print(f'ARIMA{param}x{param_seasonal}12 - Error: {e}')
#                     continue

#     def fit_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
#         """Fit SARIMA model to the data."""
#         mod = sm.tsa.statespace.SARIMAX(self.y,
#                                         order=order,
#                                         seasonal_order=seasonal_order,
#                                         enforce_stationarity=False,
#                                         enforce_invertibility=False)
#         results = mod.fit()
#         return results

#     def plot_diagnostics(self, results):
#         """Plots diagnostics for the SARIMA model."""
#         results.plot_diagnostics(figsize=(16, 8))
#         plt.show()

#     def forecast(self, results):
#         """Generates forecast for the next `forecast_steps` days."""
#         forecast = results.get_forecast(steps=self.forecast_steps)
#         forecast_ci = forecast.conf_int()
#         forecast_index = pd.date_range(start=self.y.index[-1], periods=self.forecast_steps + 1, freq='D')[1:]
#         ax = self.y.plot(label='Observed', figsize=(14, 7))
#         forecast.predicted_mean.plot(ax=ax, label=f'{self.forecast_steps}-day Forecast', color='red', alpha=0.7)

#         ax.fill_between(forecast_index,
#                         forecast_ci.iloc[:, 0],
#                         forecast_ci.iloc[:, 1],
#                         color='red', alpha=0.2)

#         ax.set_xlabel('Date')
#         ax.set_ylabel('Net Amount Forecast')
#         plt.legend()
#         plt.show()

#         return forecast.predicted_mean, forecast_ci

#     def calculate_metrics(self, forecasted_values, forecast_ci):
#         """Calculates performance metrics for the forecasted values."""
#         actual_values = self.y[-self.forecast_steps:]
#         mse = mean_squared_error(actual_values, forecasted_values)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(actual_values, forecasted_values)

#         print(f'Mean Squared Error (MSE): {mse}')
#         print(f'Root Mean Squared Error (RMSE): {rmse}')
#         print(f'R-Squared (R2) Score: {r2}')
        
#         # Log metrics to MLflow
#         mlflow.log_metric("mse", mse)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)

#     def run(self):
#         """Runs the full forecasting process."""
#         # Start an MLflow run
#         with mlflow.start_run() as run:
#             pdq, seasonal_pdq = self.generate_parameter_combinations()

#             # Perform grid search to find the best SARIMA model
#             self.grid_search(pdq, seasonal_pdq)
#             results = self.fit_model(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))

#             # Log SARIMA model parameters to MLflow
#             mlflow.log_param("order", (1, 1, 1))
#             mlflow.log_param("seasonal_order", (1, 1, 0, 12))

#             # Print model summary and diagnostics
#             print(results.summary().tables[1])
#             self.plot_diagnostics(results)

#             # Forecast and log the forecasted values
#             forecasted_values, forecast_ci = self.forecast(results)
#             forecasted_values.to_csv("forecasted_values.csv")  # Save forecasted values
#             mlflow.log_artifact("forecasted_values.csv")

#             # Calculate metrics and log them
#             self.calculate_metrics(forecasted_values, forecast_ci)

# # Run the forecasting process
# forecaster = SeasonalARIMAForecaster('grouped_data.csv')
# forecaster.run()








import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import mlflow
import mlflow.sklearn

class SeasonalARIMAForecaster:
    def __init__(self, file_path, date_column='Bill Date', value_column='Net Amount', forecast_steps=15):
        self.file_path = file_path
        self.date_column = date_column
        self.value_column = value_column
        self.forecast_steps = forecast_steps

        self.df = pd.read_csv(self.file_path, parse_dates=[self.date_column])
        self.df.set_index(self.date_column, inplace=True)
        self.df[self.value_column] = pd.to_numeric(self.df[self.value_column], errors='coerce')
        self.df.dropna(subset=[self.value_column], inplace=True)
        self.y = self.df[self.value_column]

        # Set the remote server URI to your EC2 instance where MLflow is hosted
        self.remote_server_uri = "http://ec2-13-203-206-18.ap-south-1.compute.amazonaws.com:5000/"
        mlflow.set_tracking_uri(self.remote_server_uri)

    def generate_parameter_combinations(self):
        """Generates parameter combinations for SARIMA model."""
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q)) 
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]  # Monthly seasonality
        return pdq, seasonal_pdq

    def grid_search(self, pdq, seasonal_pdq):
        """Performs grid search to find the best SARIMA model."""
        print('Examples of parameter combinations for Seasonal ARIMA...')
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    print(f'ARIMA{param}x{param_seasonal}12 - AIC: {results.aic}')
                except Exception as e:
                    print(f'ARIMA{param}x{param_seasonal}12 - Error: {e}')
                    continue

    def fit_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
        """Fit SARIMA model to the data."""
        mod = sm.tsa.statespace.SARIMAX(self.y,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()
        return results

    def plot_diagnostics(self, results):
        """Plots diagnostics for the SARIMA model."""
        results.plot_diagnostics(figsize=(16, 8))
        plt.show()

    def forecast(self, results):
        """Generates forecast for the next `forecast_steps` days."""
        forecast = results.get_forecast(steps=self.forecast_steps)
        forecast_ci = forecast.conf_int()
        forecast_index = pd.date_range(start=self.y.index[-1], periods=self.forecast_steps + 1, freq='D')[1:]
        ax = self.y.plot(label='Observed', figsize=(14, 7))
        forecast.predicted_mean.plot(ax=ax, label=f'{self.forecast_steps}-day Forecast', color='red', alpha=0.7)

        ax.fill_between(forecast_index,
                        forecast_ci.iloc[:, 0],
                        forecast_ci.iloc[:, 1],
                        color='red', alpha=0.2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Net Amount Forecast')
        plt.legend()
        plt.show()

        return forecast.predicted_mean, forecast_ci

    def calculate_metrics(self, forecasted_values, forecast_ci):
        """Calculates performance metrics for the forecasted values."""
        actual_values = self.y[-self.forecast_steps:]
        mse = mean_squared_error(actual_values, forecasted_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, forecasted_values)

        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R-Squared (R2) Score: {r2}')
        
        # Log metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

    def run(self):
        """Runs the full forecasting process."""
        # Start an MLflow run
        with mlflow.start_run() as run:
            pdq, seasonal_pdq = self.generate_parameter_combinations()

            # Perform grid search to find the best SARIMA model
            self.grid_search(pdq, seasonal_pdq)
            results = self.fit_model(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))

            # Log SARIMA model parameters to MLflow
            mlflow.log_param("order", (1, 1, 1))
            mlflow.log_param("seasonal_order", (1, 1, 0, 12))

            # Print model summary and diagnostics
            print(results.summary().tables[1])
            self.plot_diagnostics(results)

            # Forecast and log the forecasted values
            forecasted_values, forecast_ci = self.forecast(results)
            forecasted_values.to_csv("forecasted_values.csv")  # Save forecasted values
            mlflow.log_artifact("forecasted_values.csv")

            # Calculate metrics and log them
            self.calculate_metrics(forecasted_values, forecast_ci)

# Run the forecasting process
forecaster = SeasonalARIMAForecaster('grouped_data.csv')
forecaster.run()