#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:40:14 2025

@author: borish
"""
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks # <--- Se importa la función para encontrar picos
import matplotlib.pyplot as plt
import numpy as np

#%% --- 1. Definición de Nombres de Archivos y Parámetros de Muestreo ---

coeffs_filename = 'coeficientes_1-10.csv'
input_filename = 'datos.txt'
output_filename = 'resultado_filtrado.txt'

FS = 50.0 # Hertz

#%% --- 2. Carga de los Coeficientes del Filtro desde CSV ---

try:
    coeffs_df = pd.read_csv(coeffs_filename, header=None)
    B = coeffs_df[0].dropna().values
    A = [1.0]
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
    senal_original = np.loadtxt(input_filename)
    n_muestras = len(senal_original)
    tiempo_s = np.arange(n_muestras) / FS
    print(f"\nSeñal cargada con {n_muestras} muestras.")
    print(f"La duración total calculada de la señal es: {tiempo_s[-1]:.2f} segundos.")
except FileNotFoundError:
    print(f"Error: El archivo de datos '{input_filename}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de datos: {e}")
    exit()
    
#%% --- 4. Preparación y Aplicación del Filtro ---

senal_filtrada = signal.filtfilt(B, A, senal_original)

#%% --- 5. Visualización de los Resultados ---

# --- Gráfica 1: Señal Original (Línea más fina) ---
plt.figure(figsize=(15, 5))
plt.plot(tiempo_s, senal_original, label='Señal Original', color='blue', linewidth=0.8)
plt.title('Señal Original (Sin Filtrar)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

# --- Gráfica 2: Señal Filtrada (Línea más fina) ---
plt.figure(figsize=(15, 5))
plt.plot(tiempo_s, senal_filtrada, label='Señal Filtrada', color='red', linewidth=0.8)
plt.title('Señal Filtrada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

# --- Gráfica 3: Comparación (Líneas más finas) ---
plt.figure(figsize=(15, 7))
plt.plot(tiempo_s, senal_original, label='Señal Original', color='blue', alpha=0.6, linewidth=0.8)
plt.plot(tiempo_s, senal_filtrada, label='Señal Filtrada', color='red', linewidth=1.0)
plt.title('Comparación de la Señal Original vs. Filtrada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

#%% --- 6. NUEVO: ZOOM Y ANÁLISIS DE PICOS EN ZONA DE INTERÉS ---

# --- Definición de la zona de interés ---
t_inicio_zoom = 24.0
t_fin_zoom = 30.0

# --- Encontrar los índices de las muestras dentro de este rango de tiempo ---
indices_zoom = np.where((tiempo_s >= t_inicio_zoom) & (tiempo_s <= t_fin_zoom))[0]

# --- Extraer los datos ("rebanada") de esa zona ---
tiempo_zoom = tiempo_s[indices_zoom]
senal_filtrada_zoom = senal_filtrada[indices_zoom]

# --- Detección de picos en la señal de la zona de interés ---
# ¡PARÁMETROS A AJUSTAR!
# height: Amplitud mínima que debe tener un pico para ser detectado. Es útil para ignorar ruido.
# distance: Distancia horizontal mínima (en número de muestras) entre picos consecutivos.
distancia_min_s = 0.1 # segundos. Evita detectar picos muy juntos.
distancia_muestras = int(distancia_min_s * FS)

picos_indices, _ = find_peaks(senal_filtrada_zoom, height=0.01, distance=distancia_muestras)

# --- Gráfica 4: Zoom de la Señal con Picos Marcados ---
plt.figure(figsize=(15, 7))
plt.plot(tiempo_zoom, senal_filtrada_zoom, label='Señal Filtrada (Zoom)', color='red', linewidth=1.2)
plt.plot(tiempo_zoom[picos_indices], senal_filtrada_zoom[picos_indices], "x", color='green', markersize=10, label='Picos Detectados')
plt.title(f'Zoom y Detección de Picos ({t_inicio_zoom}s - {t_fin_zoom}s)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

# --- Cálculo e impresión de la frecuencia ---
if len(picos_indices) > 1:
    # Obtener los tiempos exactos en los que ocurrieron los picos
    tiempos_de_picos = tiempo_zoom[picos_indices]
    
    # Calcular la diferencia de tiempo entre picos consecutivos (el período T)
    periodos = np.diff(tiempos_de_picos)
    
    # Calcular las frecuencias instantáneas (F = 1/T)
    frecuencias_instantaneas = 1 / periodos
    
    # Calcular la frecuencia promedio en la zona
    frecuencia_promedio = np.mean(frecuencias_instantaneas)
    
    print("\n--- ANÁLISIS DE FRECUENCIA EN ZONA DE INTERÉS ---")
    print(f"Se encontraron {len(picos_indices)} picos entre {t_inicio_zoom}s y {t_fin_zoom}s.")
    print(f"Tiempos de los picos (s): {np.round(tiempos_de_picos, 2)}")
    print(f"Frecuencias instantáneas (Hz): {np.round(frecuencias_instantaneas, 2)}")
    print("--------------------------------------------------")
    print(f"==> Frecuencia Promedio: {frecuencia_promedio:.2f} Hz <==")
    print("--------------------------------------------------")
else:
    print("\nNo se encontraron suficientes picos en la zona de interés para calcular la frecuencia.")


#%% --- 7. Guardado de los Datos ---
datos_para_guardar = np.c_[tiempo_s, senal_original, senal_filtrada]
np.savetxt(output_filename,
           datos_para_guardar,
           fmt='%-18.8f',
           header='Tiempo_s             Voltaje_Original     Voltaje_Filtrado',
           comments='')

print(f"\nProceso finalizado. Los datos filtrados han sido guardados en '{output_filename}'.")