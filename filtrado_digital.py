#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:40:14 2025

@author: borish
"""
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np

# %% --- 1. Definición de Parámetros ---

coeffs_filename = 'coeficientes_1-10.csv'
input_filename = 'datos_enzo2.txt'
output_filename = 'resultado_filtrado.txt'
FS = 128.0  # Frecuencia de muestreo en Hertz

# %% --- 2. Carga de Coeficientes del Filtro ---

try:
    coeffs_df = pd.read_csv(coeffs_filename, header=None)
    B = coeffs_df[0].dropna().values
    A = [1.0]
    print("Coeficientes FIR cargados correctamente.")
except FileNotFoundError:
    print(f"Error: El archivo de coeficientes '{coeffs_filename}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de coeficientes: {e}")
    exit()

# %% --- 3. Carga de la Señal y Creación del Vector de Tiempo ---

try:
    senal_original = np.loadtxt(input_filename)
    n_muestras = len(senal_original)
    tiempo_s = np.arange(n_muestras) / FS
    print(f"\nSeñal cargada con {n_muestras} muestras ({tiempo_s[-1]:.2f} segundos).")
except FileNotFoundError:
    print(f"Error: El archivo de datos '{input_filename}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de datos: {e}")
    exit()

# %% --- 4. Aplicación del Filtro ---

senal_filtrada = signal.filtfilt(B, A, senal_original)

# %% --- 5. Visualización de Señales Completas ---

plt.figure(figsize=(15, 5))
plt.plot(tiempo_s, senal_original, label='Señal Original', color='blue', linewidth=0.8)
plt.title('Señal Original (Sin Filtrar)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(tiempo_s, senal_filtrada, label='Señal Filtrada', color='red', linewidth=0.8)
plt.title('Señal Filtrada Completa')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

# %% --- 6. Detección Automática de la Zona de Interés ---

umbral_silencio = 0.10 * np.max(np.abs(senal_filtrada))
es_silencioso = np.abs(senal_filtrada) < umbral_silencio
cambios = np.diff(np.concatenate(([False], es_silencioso, [False])).astype(int))
indices_inicio_silencio = np.where(cambios == 1)[0]
indices_fin_silencio = np.where(cambios == -1)[0]
duraciones_silencio = indices_fin_silencio - indices_inicio_silencio

if len(duraciones_silencio) == 0:
    indice_inicio_busqueda = 0
else:
    indice_gap_largo = np.argmax(duraciones_silencio)
    indice_fin_silencio_largo = indices_fin_silencio[indice_gap_largo]
    
    # Se define un margen para incluir el inicio de la señal de interés
    margen_seguridad_s = 10 # Valor ajustable en segundos
    margen_seguridad_muestras = int(margen_seguridad_s * FS)
    
    indice_inicio_busqueda = indice_fin_silencio_largo - margen_seguridad_muestras
    if indice_inicio_busqueda < 0:
        indice_inicio_busqueda = 0

zona_busqueda_senal = senal_filtrada[indice_inicio_busqueda:]
envolvente = np.abs(signal.hilbert(zona_busqueda_senal))
longitud_ventana_s = 0.5
longitud_ventana_muestras = int(longitud_ventana_s * FS)
if longitud_ventana_muestras < 1: longitud_ventana_muestras = 1
kernel_suavizado = np.ones(longitud_ventana_muestras) / longitud_ventana_muestras
envolvente_suavizada = np.convolve(envolvente, kernel_suavizado, mode='same')

amplitud_pico_suavizado = np.max(envolvente_suavizada)
indice_local_pico_suavizado = np.argmax(envolvente_suavizada)
umbral_actividad = 0.25 * amplitud_pico_suavizado

try:
    envolvente_antes_del_pico = envolvente_suavizada[:indice_local_pico_suavizado]
    indices_inicio_candidatos = np.where(envolvente_antes_del_pico <= umbral_actividad)[0]
    indice_local_inicio = indices_inicio_candidatos[-1] if len(indices_inicio_candidatos) > 0 else 0

    envolvente_despues_del_pico = envolvente_suavizada[indice_local_pico_suavizado:]
    indices_fin_candidatos = np.where(envolvente_despues_del_pico <= umbral_actividad)[0]
    indice_caida_relativo = indices_fin_candidatos[0] if len(indices_fin_candidatos) > 0 else len(envolvente_despues_del_pico) - 1
    indice_local_fin = indice_local_pico_suavizado + indice_caida_relativo
except IndexError:
    indice_local_inicio = 0
    indice_local_fin = len(zona_busqueda_senal) - 1

indice_inicio_auto = indice_inicio_busqueda + indice_local_inicio
indice_fin_auto = indice_inicio_busqueda + indice_local_fin
t_inicio_zoom = tiempo_s[indice_inicio_auto]
t_fin_zoom = tiempo_s[indice_fin_auto]

print(f"\nZona de interés detectada: {t_inicio_zoom:.2f}s a {t_fin_zoom:.2f}s")

# %% --- 7. Análisis y Visualización de la Zona de Interés ---

indices_zoom = np.where((tiempo_s >= t_inicio_zoom) & (tiempo_s <= t_fin_zoom))[0]
tiempo_zoom = tiempo_s[indices_zoom]
senal_filtrada_zoom = senal_filtrada[indices_zoom]

distancia_min_s = 0.1
distancia_muestras = int(distancia_min_s * FS)
picos_indices, _ = find_peaks(senal_filtrada_zoom, height=0.01, distance=distancia_muestras)

plt.figure(figsize=(15, 7))
plt.plot(tiempo_zoom, senal_filtrada_zoom, label='Señal Filtrada (Zoom)', color='red', alpha=0.6)
envolvente_suavizada_zoom = envolvente_suavizada[indices_zoom - indice_inicio_busqueda]
plt.plot(tiempo_zoom, envolvente_suavizada_zoom, label='Envolvente Suavizada', color='purple', linewidth=2.5)
plt.axhline(y=umbral_actividad, color='cyan', linestyle=':', label=f'Umbral Actividad ({umbral_actividad:.3f}V)')
plt.plot(tiempo_zoom[picos_indices], senal_filtrada_zoom[picos_indices], "x", color='green', markersize=10, label='Picos Detectados')
plt.title(f'Zoom y Detección de Picos ({t_inicio_zoom:.2f}s - {t_fin_zoom:.2f}s)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.grid(True)
plt.show()

if len(picos_indices) > 1:
    tiempos_de_picos = tiempo_zoom[picos_indices]
    periodos = np.diff(tiempos_de_picos)
    frecuencias_instantaneas = 1 / periodos
    frecuencia_promedio = np.mean(frecuencias_instantaneas)
    
    print(f"\nSe encontraron {len(picos_indices)} picos.")
    print(f"Frecuencia Promedio en la zona: {frecuencia_promedio:.2f} Hz")
else:
    print("\nNo se encontraron suficientes picos para calcular la frecuencia.")

# %% --- 8. Guardado de Datos ---

datos_para_guardar = np.c_[tiempo_s, senal_original, senal_filtrada]
np.savetxt(output_filename,
           datos_para_guardar,
           fmt='%-18.8f',
           header='Tiempo_s             Voltaje_Original     Voltaje_Filtrado',
           comments='')

print(f"\nProceso finalizado. Datos guardados en '{output_filename}'.")