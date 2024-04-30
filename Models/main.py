import pandas as pd
import matplotlib.pyplot as plt
import os
import time

import warnings
warnings.filterwarnings("ignore")

from arima_model import ARIMAModel
from time_series_xgboost import TimeSeriesXGBoost


def main(df, output_folder_p, train_until_p):
    resultados_df = pd.DataFrame()
    scores_total = pd.DataFrame()
    train_until = train_until_p
    output_folder = output_folder_p

    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for linea in df.loc[~df.Linea.isin(['T-A', 'M', 'J', 'K']),'Linea'].unique():

            line_folder = os.path.join(output_folder, str(linea))
            if not os.path.exists(line_folder):
                os.makedirs(line_folder)

            for hora in df.Horas.unique():
                print("=" * 100, "\n")
                print("*"*26)
                print(f"*** Linea: {linea} - Hora: {hora} ***")                
                print("*"*26, "\n")

                # Medir el tiempo de inicio del ciclo
                start_time = time.time()

                try:
                    # Instancias
                    scores = pd.DataFrame([[linea, hora]], columns=['Linea', 'Horas'])
                    arima_model_instance = ARIMAModel(df, linea=linea, hora=hora, fecha_columna='Fecha', valor_columna='Qty_passangers', frecuencia='D',
                                    train_until=train_until, p_range=range(5,8), d_range=range(1,3), q_range=range(5,8), n_splits=8)
                    
                    ts_xgb = TimeSeriesXGBoost(df,train_until=train_until)
                    ts_xgb.preprocess_data(linea, hora)

                    prediction_df = arima_model_instance.test_df

                    # Estacionariedad
                    pvalor = arima_model_instance.test_stationarity()
                    scores['p-valor'] = pvalor

                    # Entrenamiento
                    best_model, best_order, best_aic = arima_model_instance.fit(method='cross_val')
                    scores['best_aic'] = best_aic
                    scores['best_order'] = [list(best_order)]

                    print(f"El mejor orden (p,d,q) para la linea {linea} en la hora {hora} es: {best_order}")

                    n_estimators, max_depth, learning_rate, reg_lambda, best_avg_score = ts_xgb.train_model()
                    scores['n_estimators'] = n_estimators
                    scores['max_depth'] = max_depth
                    scores['learning_rate'] = learning_rate
                    scores['reg_lambda'] = reg_lambda
                    
                    print(f"Hiperparámetros: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, reg_lambda={reg_lambda}, RMSE promedio: {best_avg_score}")

                    # Predicciones
                    static_predictions, dynamic_predictions, dynamic_conf_int = arima_model_instance.forecast()
                    static_conf_int = arima_model_instance.get_confidence_static_intervals(best_model, steps=len(arima_model_instance.test_data))

                    xgboost_predict = ts_xgb.predict()
                    prediction_df['static_predictions'] = static_predictions.to_list()
                    prediction_df['dynamic_predictions'] = dynamic_predictions
                    prediction_df['l_dynamic_conf_int'] = dynamic_conf_int[:,0]
                    prediction_df['u_static_predictions'] = dynamic_conf_int[:,1]
                    prediction_df['l_static_conf_int'] = static_conf_int[:,0]
                    prediction_df['u_static_conf_int'] = static_conf_int[:,1]

                    prediction_df['xgboost_predict'] = xgboost_predict

                    # graficar
                    fig = arima_model_instance.plot_predictions_vs_actual(static_predictions, dynamic_predictions, static_conf_int=static_conf_int, dynamic_conf_int=dynamic_conf_int)
                    filename_arima = os.path.join(line_folder, f"Arima_figura_{linea}_{hora}.png")
                    filename_ts_xgb = os.path.join(line_folder, f"XGBoost_figura_{linea}_{hora}.png")
                    fig.savefig(filename_arima)
                    plt.clf()
                    plt.close()

                    
                    fig, no_parametric_lower_ic_list, no_parametric_upper_ic_list, semi_parametric_lower_ic_list, semi_parametric_upper_ic_list = ts_xgb.plot_predic()
                    fig.savefig(filename_ts_xgb)
                    plt.clf()
                    plt.close()            

                    prediction_df['l_xgboost_no_parametric_ic'] = no_parametric_upper_ic_list
                    prediction_df['u_xgboost_no_parametric_ic'] = no_parametric_lower_ic_list
                    prediction_df['l_xgboost_semi_parametric_ic'] = semi_parametric_upper_ic_list
                    prediction_df['u_xgboost_semi_parametric_ic'] = semi_parametric_lower_ic_list
                    
                    # Metricas
                    scores_arima = arima_model_instance.score(static_predictions, dynamic_predictions)
                    ts_xgb_rmse = ts_xgb.score()

                    # scores['r2_static'] = scores_arima['static_r2']
                    scores['rmse_static'] = scores_arima['static_rmse']
                    # scores['r2_dynamic'] = scores_arima['dynamic_r2']
                    scores['rmse_dynamic'] = scores_arima['dynamic_rmse']
                    scores['rmse_xgboost'] = ts_xgb_rmse

                    # Llenando la salida
                    resultados_df = pd.concat([resultados_df, prediction_df])
                    scores_total = pd.concat([scores_total, scores])

                    # Guardando la salida
                    resultados_df.to_excel(os.path.join(output_folder, "resultados_df.xlsx"), index=False)
                    scores_total.to_excel(os.path.join(output_folder, "scores_total.xlsx"), index=False)

                    # Medir el tiempo de finalización del ciclo y calcular la duración
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Tiempo de ejecución del ciclo: {duration} segundos")
                except Exception as e:
                    print(f"*** ERROR EN LA LINEA {linea} - HORA {hora} ***")
                    print(e)
                    continue

        resultados_df['Fecha'] = resultados_df['Fecha'].astype(str)
        resultados_df.to_excel(os.path.join(output_folder, "resultados_df.xlsx"), index=False)
        scores_total.to_excel(os.path.join(output_folder, "scores_total.xlsx"), index=False)
    except Exception as e:
        print("Error general:")
        print(e)

if __name__ == "__main__":

    input_path = 'DataImputed2024.csv'
    output_folder = "output_data"
    train_until = '2024-01-24'
    df = pd.read_csv(input_path).drop('Unnamed: 0', axis=1)

    main(df, output_folder, train_until)