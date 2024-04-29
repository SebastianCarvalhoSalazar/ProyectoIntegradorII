import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

class TimeSeriesXGBoost:
    def __init__(self, df, fecha_columna='Fecha', valor_columna='Qty_passangers', linea_columna='Linea', horas_columna='Horas', train_until='2023-06-23'):
        self.df = df
        self.fecha_columna = fecha_columna
        self.valor_columna = valor_columna
        self.linea_columna = linea_columna
        self.horas_columna = horas_columna
        self.train_until = train_until

    def preprocess_data(self, linea, hora):
        self.df = self.df.set_index(self.fecha_columna)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df[(self.df[self.linea_columna] == linea) & (self.df[self.horas_columna] == hora)]
        self.df = self.df[[self.valor_columna]]
        self.df = self.df.sort_index()
        self.train = self.df.loc[self.df.index < self.train_until]
        self.test = self.df.loc[self.df.index >= self.train_until]

    def create_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        df['rolling_mean_7d'] = df['Qty_passangers'].rolling(window=7,min_periods=1).mean().fillna(method='ffill')
        df['rolling_std_7d'] = df['Qty_passangers'].rolling(window=7,min_periods=1).std().fillna(method='ffill')
        df['rolling_mean_30d'] = df['Qty_passangers'].rolling(window=30,min_periods=1).mean().fillna(method='ffill')
        df['rolling_std_30d'] = df['Qty_passangers'].rolling(window=30,min_periods=1).std().fillna(method='ffill')
        # df['dist_to_rolling_mean_7d'] = df[self.valor_columna] - df['rolling_mean_7d']
        # df['dist_to_rolling_mean_30d'] = df[self.valor_columna] - df['rolling_mean_30d']
        # df['dist_to_rolling_mean_7d'] = np.sqrt((df[self.valor_columna] - df['rolling_mean_7d']) ** 2)
        # df['dist_to_rolling_mean_30d'] = np.sqrt((df[self.valor_columna] - df['rolling_mean_30d']) ** 2)
        return df

    def add_lags(self, df):
        target_map = df['Qty_passangers'].to_dict()
        df['lag1'] = (df.index - pd.Timedelta('7 days')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('14 days')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('21 days')).map(target_map)
        df['lag4'] = (df.index - pd.Timedelta('28 days')).map(target_map)
        return df

    def train_model(self):
        self.train = self.create_features(self.train)
        self.train = self.add_lags(self.train)

        self.FEATURES = self.train.drop([self.valor_columna], axis=1).columns
        TARGET = self.valor_columna

        # Definir los hiperparámetros a ajustar
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [4, 6, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'reg_lambda': [0.01, 0.1, 1]
        }

        # Inicializar el mejor modelo y el mejor puntaje
        all_avg_scores = []
        best_avg_score = float('inf')
        self.best_model = None

        # Dividir los datos utilizando TimeSeriesSplit
        tss = TimeSeriesSplit(n_splits=8)

        # Realizar la búsqueda de hiperparámetros con validación cruzada en serie temporal
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for learning_rate in param_grid['learning_rate']:
                    for reg_lambda in param_grid['reg_lambda']:
                        fold_scores = []  # Almacenar los puntajes RMSE de cada fold

                        for train_idx, val_idx in tss.split(self.train):
                            cross_val_train = self.train.iloc[train_idx]
                            cross_val_test = self.train.iloc[val_idx]

                            X_train = cross_val_train[self.FEATURES]
                            y_train = cross_val_train[TARGET]

                            X_test = cross_val_test[self.FEATURES]
                            y_test = cross_val_test[TARGET]

                            reg = xgb.XGBRegressor(n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                learning_rate=learning_rate,
                                                objective='reg:squarederror',
                                                reg_lambda=reg_lambda)

                            reg.fit(X_train, y_train,
                                    eval_set=[(X_train, y_train), (X_test, y_test)],
                                    verbose=False)

                            y_pred = reg.predict(X_test)
                            score = np.sqrt(mean_squared_error(y_test, y_pred))

                            fold_scores.append(score)  # Agregar el puntaje RMSE del fold actual a la lista

                        avg_score = np.mean(fold_scores)  # Calcular el promedio de los puntajes RMSE del fold
                        all_avg_scores.append(avg_score)
                        if avg_score < best_avg_score:
                            best_avg_score = avg_score
                            self.best_model = reg
                            # print(f"Hiperparámetros: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, reg_lambda={reg_lambda}, RMSE promedio: {best_avg_score}")

        # intervalos de confianza No Parametricos
        alpha = 95
        alpha2 = alpha/2
        self.no_parametric_lower_ic = np.percentile(all_avg_scores, alpha2)
        self.no_parametric_upper_ic = np.percentile(all_avg_scores, 100 - alpha2)

        # Intervalos de confianza Semi-Parametricos
        valor_95 = norm.ppf(alpha/100)
        std_rmse_cv = np.sqrt(np.var(all_avg_scores))
        self.semi_parametric_ic = valor_95 * std_rmse_cv

        # Parametros del modelo
        params_dict = self.best_model.get_params()
        keys = ['n_estimators', 'max_depth', 'learning_rate', 'reg_lambda']
        selected_params = {key : params_dict[key] for key in keys}
        return selected_params['n_estimators'], selected_params['max_depth'], selected_params['learning_rate'], selected_params['reg_lambda'], best_avg_score

    def predict(self):
        self.best_model.fit(self.train.drop([self.valor_columna], axis=1), self.train[[self.valor_columna]])

        future = pd.date_range(str(self.train.index.max().date()), str(self.test.index.max().date()), freq='1D')
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True

        self.train['isFuture'] = False
        df_and_future = pd.concat([self.train, future_df])
        df_and_future = self.create_features(df_and_future)
        df_and_future = self.add_lags(df_and_future)

        future_w_features = df_and_future.query('isFuture').copy()
        future_w_features['pred'] = self.best_model.predict(future_w_features[self.FEATURES])
        self.future_w_features_pred = future_w_features

        return self.future_w_features_pred['pred'].to_list()[2:]


    def plot_predic(self):
            
        predictions = self.future_w_features_pred['pred'].iloc[2:]
        no_parametric_lower_ic_list = predictions - (self.no_parametric_lower_ic - np.mean(predictions.values))
        no_parametric_upper_ic_list = predictions + (self.no_parametric_upper_ic - np.mean(predictions.values))
        semi_parametric_lower_ic_list = predictions - (self.semi_parametric_ic - np.mean(predictions.values))
        semi_parametric_upper_ic_list = predictions + (self.semi_parametric_ic - np.mean(predictions.values))

        fig, ax = plt.subplots(figsize=(15, 4.5))
        predictions.reset_index(drop=True).plot(ax=ax, color='r', ms=6, lw=2, title='Predictions Vs Real Data', grid=True, label='Predictions', marker='o')
        self.test[self.valor_columna].iloc[1:].reset_index(drop=True).plot(ax=ax, color='b', ms=6, lw=2, grid=True, label='Real Data', marker='o')

        # Agregar intervalos de confianza
        plt.fill_between(predictions.reset_index(drop=True).index, no_parametric_lower_ic_list, no_parametric_upper_ic_list, color='orange', alpha=0.3, label='Non-Parametric 95% CI')
        plt.fill_between(predictions.reset_index(drop=True).index, semi_parametric_lower_ic_list, semi_parametric_upper_ic_list, color='blue', alpha=0.3, label='Semi-Parametric 95% CI')

        plt.xlabel('Days')
        plt.ylabel('influx')
        plt.legend()
        return fig

    def score(self):
        rmse = np.sqrt(mean_squared_error(self.future_w_features_pred['pred'].values[2::], self.test[self.valor_columna].values[1::]))
        print(f"Error cuadrático medio (RMSE) para predicción XGBoost es: {rmse}")
        return rmse