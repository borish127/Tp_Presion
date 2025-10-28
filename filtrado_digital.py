#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:40:14 2025

@author: borish
"""

# SCRIPT PARA EL FILTRADO DIGITAL DE UNA SEÑAL DESDE UN ARCHIVO TXT
# -----------------------------------------------------------------
# Este script carga coeficientes de filtro desde un archivo CSV y datos
# de señal desde un archivo TXT. Aplica el filtro y guarda el resultado.

import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

#%% --- 1. Definición de Nombres de Archivos ---

# Nombres de los archivos a utilizar.
coeffs_filename = 'coeficientes.csv'
input_filename = 'datos.txt'
output_filename = 'resultado_filtrado.txt'

#%% --- 2. Carga de los Coeficientes del Filtro desde CSV ---

try:
    # Se leen los coeficientes desde el archivo CSV.
    # header=None indica que el archivo no tiene fila de encabezado.
    coeffs_df = pd.read_csv(coeffs_filename, header=None)
    
    # Se extraen las columnas para obtener los coeficientes B (primera columna) y A (segunda columna).
    # .dropna() elimina cualquier valor no numérico (NaN) que pueda resultar de columnas desiguales.
    B = coeffs_df[0].dropna().values
    A = coeffs_df[1].dropna().values
    
    print("Coeficientes cargados correctamente.")
    print(f"B = {B}")
    print(f"A = {A}")

except FileNotFoundError:
    print(f"Error: El archivo de coeficientes '{coeffs_filename}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de coeficientes: {e}")
    exit()


#%% --- 3. Carga de la Señal desde el Archivo TXT ---

try:
    # Se realiza la lectura del archivo TXT de datos.
    df = pd.read_csv(input_filename, sep=r'\s+', header=None, names=['Tiempo', 'Voltaje'])
    tiempo = df['Tiempo'].values
    senal_original = df['Voltaje'].values

except FileNotFoundError:
    print(f"Error: El archivo de datos '{input_filename}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de datos: {e}")
    exit()


#%% --- 4. Aplicación del Filtro Digital ---

# Se utiliza la función signal.filtfilt para aplicar el filtro de fase cero.
senal_filtrada = signal.filtfilt(B, A, senal_original)


#%% --- 5. Visualización de los Resultados ---

# Se genera un gráfico que superpone la señal original y la señal filtrada.
plt.figure(figsize=(15, 7))
plt.plot(tiempo, senal_original, label='Señal Original', color='blue', alpha=0.6)
plt.plot(tiempo, senal_filtrada, label='Señal Filtrada', color='red', linewidth=2)
plt.title('Comparación de la Señal Original vs. Filtrada')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()


#%% --- 6. Guardado de los Datos Filtrados en TXT ---

# Se combinan los arrays de resultados en una sola matriz para guardarla.
datos_para_guardar = np.c_[tiempo, senal_original, senal_filtrada]

# Se guarda la matriz en un archivo de texto con formato.
np.savetxt(output_filename,
           datos_para_guardar,
           fmt='%-18.8f',
           header='Tiempo               Voltaje_Original       Voltaje_Filtrado',
           comments='')

print(f"Proceso finalizado. Los datos filtrados han sido guardados en '{output_filename}'.")