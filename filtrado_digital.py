#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:40:14 2025

@author: borish
"""
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

#%% --- 1. Definición de Nombres de Archivos y Parámetros de Muestreo ---

coeffs_filename = 'coeficientes.csv'
input_filename = 'datos.txt'
output_filename = 'resultado_filtrado.txt'

# --- ¡MODIFICACIÓN CLAVE! ---
# Se define la Frecuencia de Muestreo (Fs) en Hertz.
# Este valor DEBE coincidir con la configuración del Arduino (1,000,000 / delayMicroseconds).
FS = 128.0 # Hertz
# -------------------------

#%% --- 2. Carga de los Coeficientes del Filtro desde CSV ---

try:
    # Se leen los coeficientes desde el archivo CSV, que ahora tiene una sola columna.
    coeffs_df = pd.read_csv(coeffs_filename, header=None)
    
    # Se extrae la primera columna para obtener los coeficientes B.
    B = coeffs_df[0].dropna().values
    
    # Para un filtro FIR, el denominador A es siempre [1.0].
    A = [1.0]
    # -------------------------
    
    print("Coeficientes FIR cargados correctamente.")
    print(f"B = {B}")
    print(f"A = {A}")

except FileNotFoundError:
    print(f"Error: El archivo de coeficientes '{coeffs_filename}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de coeficientes: {e}")
    exit()
    
#%% --- 3. Carga de Señal y CONSTRUCCIÓN del Vector de Tiempo ---

try:
    # Se utiliza np.loadtxt para cargar directamente el archivo de texto
    # en un array de NumPy.
    senal_original = np.loadtxt(input_filename)
    # --------------------------------------------------------

    # Se construye el vector de tiempo desde cero usando la frecuencia de muestreo (FS).
    n_muestras = len(senal_original)
    tiempo_s = np.arange(n_muestras) / FS
    
    # Verificación de la duración.
    print(f"\nSeñal cargada con {n_muestras} muestras.")
    print(f"La duración total calculada de la señal es: {tiempo_s[-1]:.2f} segundos.")


except FileNotFoundError:
    print(f"Error: El archivo de datos '{input_filename}' no fue encontrado.")
    exit()
except Exception as e:
    # Este error ahora será mucho más específico si hay una línea que no es un número.
    print(f"Ocurrió un error al leer el archivo de datos: {e}")
    exit()
    
#%% --- 4. Preparación y Aplicación del Filtro ---
# (Esta sección no cambia)

senal_filtrada = signal.filtfilt(B, A, senal_original)

#%% --- 5. Visualización de los Resultados ---
# (Esta sección no cambia, ya que utiliza la variable 'tiempo_s' que acabamos de crear)
# --- Gráfica 1: Señal Original ---
plt.figure(figsize=(15, 5))
plt.plot(tiempo_s, senal_original, label='Señal Original', color='blue')
plt.title('Señal Original (Sin Filtrar)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

# --- Gráfica 2: Señal Filtrada ---
plt.figure(figsize=(15, 5))
plt.plot(tiempo_s, senal_filtrada, label='Señal Filtrada', color='red')
plt.title('Señal Filtrada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

# --- Gráfica 3: Comparación ---
plt.figure(figsize=(15, 7))
plt.plot(tiempo_s, senal_original, label='Señal Original', color='blue', alpha=0.6)
plt.plot(tiempo_s, senal_filtrada, label='Señal Filtrada', color='red', linewidth=1.5)
plt.title('Comparación de la Señal Original vs. Filtrada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

#%% --- 6. Guardado de los Datos ---
# (Esta sección no cambia)
datos_para_guardar = np.c_[tiempo_s, senal_original, senal_filtrada]
np.savetxt(output_filename,
           datos_para_guardar,
           fmt='%-18.8f',
           header='Tiempo_s             Voltaje_Original       Voltaje_Filtrado',
           comments='')

print(f"Proceso finalizado. Los datos filtrados han sido guardados en '{output_filename}'.")