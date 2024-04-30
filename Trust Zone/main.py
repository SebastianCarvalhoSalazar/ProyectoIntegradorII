# %%
import holidays_co
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, date
from upload_file import upload_file
import openpyxl
from sklearn.preprocessing import LabelEncoder
import statistics

# Definicion de funciones necesarias para la transformación de los datos

def load_data(Link, header):
  data = pd.read_excel(Link, header=header)
  data_columns = data.columns
  data.fillna(0, inplace=True)
  # Convirtiendo los nombres de las columnas en string
  data_columns = list(map(lambda index: str(index).replace(':00:00', ''), data_columns))

  # Asignando el verdadero valor de las columnas 0 y 1
  data_columns[0] = 'Fecha'
  data_columns[1] = 'Linea'
  data = data.set_axis(data_columns, axis=1).drop(data.columns[-1], axis=1)

  # Convirtiendo las horas en columnas como observaciones del DF
  data = data.set_index(['Fecha', 'Linea']).stack()
  data = data.reset_index().rename(columns={'level_2': 'Horas', 0: 'Qty_passangers'})

  # Organizando la columna de Lineas
  data['Linea'] = data.Linea.apply(lambda x: x.replace("LÍNEA ", ""))
  return data

def convert_to_date_format(day_of_year, year):
  date_convert = datetime.fromordinal(date(2021, 1, 1).toordinal() + day_of_year)
  return date_convert.strftime("%Y-%m-%d")

def custom_window(group, name, qty_observations_up, qty_observations_down):
    # Crea una nueva columna con una lista de valores de 'Qty_passangers' con una ventana de 5 días hacia adelante y hacia atrás
    window_values = []
    for i in range(len(group)):
        start_index = max(0, i - qty_observations_down)
        end_index = min(len(group), i + qty_observations_up)
        window_values.append(group['Qty_passangers'].iloc[start_index:end_index].tolist())
    
    group[name] = window_values

    return group

# Modelos para la transformación de los datos
le = LabelEncoder()

afluencia_2021 = load_data("https://www.arcgis.com/sharing/rest/content/items/1e5e1bb004944971a57fa2847137d9f0/data", header=2)
afluencia_2022 = load_data("https://www.arcgis.com/sharing/rest/content/items/4c66112ec6d045f29f7ad2cbffe06cc2/data", header=1)
afluencia_2023 = load_data("https://www.arcgis.com/sharing/rest/content/items/569c4b4c1ad54c3da95aa5f195637db2/data", header=1)
afluencia_2023['Fecha'] = pd.to_datetime(afluencia_2023['Fecha'], format='%d/%m/%Y')
afluencia_2024 = load_data("https://www.arcgis.com/sharing/rest/content/items/ba2a0bdefaa04a39adcfccab11e76198/data", header=1)
# Uniendo todos los datasets
all_data = pd.concat([afluencia_2021, afluencia_2022, afluencia_2023, afluencia_2024])
afluencia_2021['Fecha'] = le.fit_transform(afluencia_2021['Fecha'])
afluencia_2021['Fecha'] = afluencia_2021.Fecha.apply(lambda x: convert_to_date_format(x, 2021))
all_data = pd.concat([afluencia_2021, afluencia_2022, afluencia_2023, afluencia_2024])
all_data['Fecha'] = pd.to_datetime(all_data['Fecha'], format='%Y-%m-%d')
all_data['uploaded'] = datetime.now()

festivos = []

# Agregando los festivos al dataset
for anio in all_data.Fecha.dt.year.unique():
  festivos = festivos + list(map(lambda x: str(x.date), holidays_co.get_colombia_holidays_by_year(anio)))

all_data['festivos'] = all_data.Fecha.astype(str).apply(lambda x: 1 if x in festivos else 0)

# Eliminando horas extremo
all_data = all_data.loc[~all_data.Horas.isin(['04', '23'])]
all_data = all_data.loc[~all_data.Linea.isin(['L', 'P'])]

# # Ordenando el dataset
all_data = all_data.sort_values(by=['Linea', 'Horas'])

# # Agrupando los datos por 'Linea' y 'Horas'
grouped = all_data.groupby(['Linea', 'Horas'])

# # Aplica la función a cada grupo
name='window_daily'
all_data = grouped.apply(lambda x: custom_window(x, name=name, qty_observations_up=11, qty_observations_down=10))

# # Mediana
all_data['Mediana'] = all_data[name].apply(lambda lista: statistics.median(lista))

# Promedio
all_data['promedio'] = all_data[name].apply(lambda lista: statistics.mean(lista))

all_data['relacion'] = all_data.Mediana / all_data.promedio


# # Llenando Se-manal
all_data['dia_semana'] = all_data.Fecha.dt.day_name()
all_data = all_data.drop(['Linea', 'Horas'], axis=1)

grouping_by = ['Linea', 'Horas', 'dia_semana']
# Ordenando el dataset
all_data = all_data.reset_index(drop=False).sort_values(by=grouping_by)

# Agrupando los datos por 'Linea' y 'Horas'
grouped = all_data.groupby(grouping_by)

# Aplica la función a cada grupo
name='window_weekly'
all_data = grouped.apply(lambda x: custom_window(x, name=name, qty_observations_up=6, qty_observations_down=5))

# Mediana
all_data['Mediana_semanal'] = all_data[name].apply(lambda lista: statistics.median(lista))

# Promedio
all_data['promedio_semanal'] = all_data[name].apply(lambda lista: statistics.mean(lista))

all_data['relacion_semanal'] = all_data.Mediana_semanal / all_data.promedio_semanal

all_data = all_data.drop(grouping_by+['level_2'], axis=1)
all_data.reset_index(inplace=True)
all_data.drop(['level_3'], axis=1, inplace=True)
all_data.to_csv('data_completa_abril_2024.csv')

