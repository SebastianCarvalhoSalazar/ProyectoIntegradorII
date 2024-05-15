import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import itertools
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.filterwarnings("ignore")

class ARIMAModel:
    def __init__(self, df, linea=None, hora=None, fecha_columna='Fecha', valor_columna='Qty_passangers', frecuencia='D',
                 train_until=None, p_range=range(1,1), d_range=range(1,1), q_range=range(1,1), n_splits=5):
        self.train_df = df.loc[df.Fecha<=train_until]
        self.test_df = df.loc[df.Fecha>train_until]
        self.train_df = self.compactar_datos(self.train_df, linea, hora, fecha_columna, valor_columna, frecuencia)
        self.test_df = self.compactar_datos(self.test_df, linea, hora, fecha_columna, valor_columna, frecuencia)
        self.train_data = self.train_df[valor_columna]
        self.test_data = self.test_df[valor_columna]
        self.df = self.compactar_datos(df, linea, hora, fecha_columna, valor_columna, frecuencia)
        self.X = self.df[valor_columna]
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.n_splits = n_splits

    def compactar_datos(self, df, linea=None, hora=None, fecha_columna='fecha', valor_columna='afluencia', frecuencia='D'):
        df_copy = df.copy()

        if linea is not None:
            df_copy = df_copy[df_copy['Linea'] == linea]
        if hora is not None:
            df_copy = df_copy[df_copy['Horas'] == hora]

        df_copy[fecha_columna] = pd.to_datetime(df_copy[fecha_columna])
        df_copy.set_index(fecha_columna, inplace=True)

        df_compactado = df_copy.groupby(['Linea', 'Horas', pd.Grouper(freq=frecuencia)]).sum().reset_index()

        return df_compactado

    def test_stationarity(self):
      """ Realizar la prueba de Dickey-Fuller """
      result = adfuller(self.train_data)
      print('Prueba de Dickey-Fuller:')
      print(f'Estadística de prueba: {result[0]}')
      print(f'P-valor: {result[1]}')
      print(f'Valores críticos: {result[4]}')
      return result[1]

    def fit(self, method='manual_search'):
        if method == 'manual_search':
            return self._manual_search()
        elif method == 'cross_val':
            return self._cross_val()
        else:
            print("Método No Definido")

    def _manual_search(self):
        best_aic = float('inf')
        self.best_order = None
        best_model = None
        H = adfuller(self.train_data)
        if H[1] < 0.05:
          self.d_range = range(1)
          print("La serie de Tiempo es Estacionaria, ya que el valor de p es menor que 0.05, lo que indica que hay suficiente evidencia para rechazar la hipótesis nula de no estacionariedad.")
        else:
          pass
        order_combinations = list(itertools.product(self.p_range, self.d_range, self.q_range))
        for order in order_combinations:
            try:
                arima_model = ARIMA(self.train_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
                arima_model_fit = arima_model.fit()
                current_aic = arima_model_fit.aic
                if current_aic < best_aic:
                    best_aic = current_aic
                    self.best_order = order
                    best_model = arima_model_fit
            except ValueError as e:
                print(f'Error ajustando modelo ARIMA para orden {order}: {e}')
                continue
        return best_model, self.best_order, best_aic

    def _cross_val(self):
        best_aic = float('inf')
        self.best_order = None
        best_model = None
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        H = adfuller(self.train_data)
        if H[1] < 0.05:
          self.d_range = range(1)
          print("La serie de Tiempo es Estacionaria, ya que el valor de p es menor que 0.05, lo que indica que hay suficiente evidencia para rechazar la hipótesis nula de no estacionariedad.")
        else:
          pass
        order_combinations = list(itertools.product(self.p_range, self.d_range, self.q_range))
        for order in order_combinations:
            aic_scores = []
            for train_index, test_index in tscv.split(self.train_data):
                train_data, test_data = self.train_data.iloc[train_index], self.train_data.iloc[test_index]
                try:
                    arima_model = ARIMA(train_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
                    arima_model_fit = arima_model.fit()
                    current_aic = arima_model_fit.aic
                    aic_scores.append(current_aic)
                except ValueError as e:
                    print(f'Error ajustando modelo ARIMA para orden {order}: {e}')
                    continue
            avg_aic = np.mean(aic_scores)
            if avg_aic < best_aic:
                best_aic = avg_aic
                self.best_order = order
                best_model = ARIMA(self.train_data, order=self.best_order, enforce_stationarity=False, enforce_invertibility=False).fit()
                # print(f"El mejor orden (p,d,q) es: {self.best_order}")
        return best_model, self.best_order, best_aic

    def apply_arima_to_test(self, model_fit):
        if model_fit is None:
            print("No se proporcionó un modelo ARIMA válido.")
            return None
        predictions = model_fit.forecast(steps=len(self.test_data))
        return predictions

    def forecast(self, steps=None, alpha=0.05):
        if steps is None:
            steps = len(self.test_data)

        static_model = ARIMA(self.train_data, order=self.best_order)
        static_model_fit = static_model.fit()
        static_forecast = static_model_fit.forecast(steps=steps)

        dynamic_forecast = []
        dynamic_conf_int = []
        history = [x for x in self.train_data]
        for _ in range(steps):
            model = ARIMA(history, order=self.best_order)
            model_fit = model.fit()
            output = model_fit.forecast()

            forecast_conf_int = model_fit.get_forecast(steps=1).conf_int(alpha=alpha)
            dynamic_conf_int.append(forecast_conf_int.tolist()[0])

            dynamic_forecast.append(output[0])
            history.append(output[0])

        dynamic_conf_int = pd.DataFrame(dynamic_conf_int)

        return static_forecast, dynamic_forecast, dynamic_conf_int.values

    def plot_predictions_vs_actual(self, static_predictions, dynamic_predictions, static_conf_int=None, dynamic_conf_int=None):
        df_vis = pd.DataFrame({'Actual': self.test_data, 'Static Predicted': static_predictions.values, 'Dynamic Predicted': dynamic_predictions}, index=self.test_data.index)
        fig = plt.figure(figsize=(15, 4.5))
        plt.plot(df_vis['Actual'], label='Real Data', marker='o')
        plt.plot(df_vis['Static Predicted'], label='Static Predicted', marker='o')
        plt.plot(df_vis['Dynamic Predicted'], label='Dynamic Predicted', marker='o')

        # if static_conf_int is not None:
        #     plt.fill_between(df_vis.index, static_conf_int[:,0], static_conf_int[:,1], color='blue', alpha=0.2, label='Static Confidence Interval')

        # if dynamic_conf_int is not None:
        #     plt.fill_between(df_vis.index, dynamic_conf_int[:,0], dynamic_conf_int[:,1], color='orange', alpha=0.2, label='Dynamic Confidence Interval')

        plt.title('Comparison between Real Data, Static Predictions and Dynamic Predictions')
        plt.xlabel('Days')
        plt.ylabel('Influx')
        plt.legend()
        plt.grid(True)
        # plt.show()
        return fig

    def get_confidence_static_intervals(self, model_fit, steps=None, alpha=0.05):
        if model_fit is None:
            print("No se proporcionó un modelo ARIMA válido.")
            return None

        forecast = model_fit.get_forecast(steps=steps)
        conf_int = forecast.conf_int(alpha=alpha)
        return conf_int.values

    def score(self, static_predictions, dynamic_predictions):
        r2_static = r2_score(self.test_data, static_predictions)
        # print(f"Coeficiente de determinación (R²) para predicción estática: {r2_static}")
        mse_static = mean_squared_error(self.test_data, static_predictions)
        print(f"Error cuadrático medio (RMSE) para predicción estática: {np.sqrt(mse_static)}")

        r2_dynamic = r2_score(self.test_data, dynamic_predictions)
        # print(f"Coeficiente de determinación (R²) para predicción dinámica: {r2_dynamic}")
        mse_dynamic = mean_squared_error(self.test_data, dynamic_predictions)
        print(f"Error cuadrático medio (RMSE) para predicción dinámica: {np.sqrt(mse_dynamic)}")

        return {'static_r2': r2_static, 'static_rmse': np.sqrt(mse_static),
                'dynamic_r2': r2_dynamic, 'dynamic_rmse': np.sqrt(mse_dynamic)}