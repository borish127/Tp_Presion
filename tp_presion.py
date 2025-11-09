#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script modificado para analizar la presión arterial mediante el método oscilométrico.

Determina los puntos de presión sistólica, diastólica y media (PAM)
basándose en la envolvente de la señal de oscilación filtrada.
"""
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# %% --- 1. Definición de Parámetros ---

# --- Parámetros del Script ---
# (Puedes cambiar estos valores para probar con otros archivos)
COEFFS_FILENAME = 'coeficientes_1-10.csv'
INPUT_FILENAME = 'datos_rodri2.txt'
OUTPUT_FILENAME = 'resultado_filtrado.txt'
FS = 128.0  # Frecuencia de muestreo en Hertz (del archivo .ino)

# --- Parámetros de Calibración ---
# Según la bitácora: S_final = 2.025V / 280mmHg = 0.00723 V/mmHg
# Factor = 1 / S_final = 1 / 0.00723 = 138.31
FACTOR_CONVERSION_V_A_MMHG = 138.31

# --- Parámetros del Método Oscilométrico ---
# Estos ratios son valores empíricos estándar. Pueden ajustarse.
# Ratio de amplitud para la presión sistólica (respecto al pico de la PAM)
RATIO_SISTOLICA = 0.25
# Ratio de amplitud para la presión diastólica (respecto al pico de la PAM)
RATIO_DIASTOLICA = 0.85

# --- Parámetros de Suavizado y Detección ---
LONGITUD_VENTANA_SUAVIZADO_S = 0.5 # Segundos
# Segundos a ignorar después del pico de inflado para evitar artefactos
SEGUNDOS_POST_PICO_IGNORAR = 2.0 
# Presión mínima para detener el análisis (evita artefactos al final)
PRESION_MINIMA_ANALISIS_MMHG = 30.0

# %% --- 2. Carga de Coeficientes del Filtro ---

try:
    coeffs_df = pd.read_csv(COEFFS_FILENAME, header=None)
    B = coeffs_df[0].dropna().values
    A = [1.0]
    print(f"Coeficientes FIR cargados desde '{COEFFS_FILENAME}'.")
except FileNotFoundError:
    print(f"Error: El archivo de coeficientes '{COEFFS_FILENAME}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de coeficientes: {e}")
    exit()

# %% --- 3. Carga de la Señal y Creación del Vector de Tiempo ---

try:
    senal_original_v = np.loadtxt(INPUT_FILENAME) # Señal en Voltios
    n_muestras = len(senal_original_v)
    tiempo_s = np.arange(n_muestras) / FS
    print(f"Señal cargada desde '{INPUT_FILENAME}' con {n_muestras} muestras ({tiempo_s[-1]:.2f} segundos).")
except FileNotFoundError:
    print(f"Error: El archivo de datos '{INPUT_FILENAME}' no fue encontrado.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo de datos: {e}")
    exit()

# %% --- 4. Aplicación del Filtro y Conversión de Unidades ---

# Usamos filtfilt para una respuesta de fase cero
senal_filtrada = signal.filtfilt(B, A, senal_original_v)
print("Señal filtrada correctamente.")

# Convertimos la señal original de Voltios a mmHg
senal_presion_mmhg = senal_original_v * FACTOR_CONVERSION_V_A_MMHG
print(f"Señal convertida a mmHg usando factor {FACTOR_CONVERSION_V_A_MMHG}.")

# %% --- 5. Detección de la Zona de Interés (Fase de Desinflado) ---

indice_pico_presion = 0
indice_inicio_analisis = 0
indice_fin_analisis = n_muestras -1

# El método oscilométrico mide durante el desinflado.
# Buscamos el pico de presión en la señal convertida para encontrar el inicio del desinflado.
try:
    indice_pico_presion = np.argmax(senal_presion_mmhg)
    
    # Si el pico está al final, algo salió mal (no hay desinflado)
    if indice_pico_presion > len(senal_presion_mmhg) - int(5 * FS): # Margen de 5 seg
        print("Advertencia: El pico de presión está al final de la señal. No se detectó desinflado.")
        # Como fallback, usamos la última mitad de la señal
        indice_pico_presion = n_muestras // 2

    tiempo_pico_presion = tiempo_s[indice_pico_presion]
    print(f"Pico de inflado detectado en {tiempo_pico_presion:.2f}s (Muestra {indice_pico_presion}).")

    # Calculamos el inicio real del análisis, saltando el artefacto post-pico
    muestras_a_ignorar = int(SEGUNDOS_POST_PICO_IGNORAR * FS)
    indice_inicio_analisis = indice_pico_presion + muestras_a_ignorar

    # Validamos que el índice de inicio no se vaya del arreglo
    if indice_inicio_analisis >= n_muestras:
        print("Error: El tiempo a ignorar supera la duración de la señal post-pico.")
        indice_inicio_analisis = indice_pico_presion
    
    tiempo_inicio_analisis = tiempo_s[indice_inicio_analisis]
    print(f"Inicio de análisis (post-artefacto) en {tiempo_inicio_analisis:.2f}s (Muestra {indice_inicio_analisis}).")

    # Buscamos el fin del análisis (cuando la presión baja del umbral mínimo)
    indices_fin_candidatos = np.where(senal_presion_mmhg[indice_inicio_analisis:] < PRESION_MINIMA_ANALISIS_MMHG)[0]
    
    if len(indices_fin_candidatos) > 0:
        indice_fin_analisis = indice_inicio_analisis + indices_fin_candidatos[0]
    else:
        print(f"Advertencia: La presión nunca cayó por debajo de {PRESION_MINIMA_ANALISIS_MMHG} mmHg. Analizando hasta el final.")
        indice_fin_analisis = n_muestras - 1
        
    tiempo_fin_analisis = tiempo_s[indice_fin_analisis]
    print(f"Fin de análisis (presión < {PRESION_MINIMA_ANALISIS_MMHG} mmHg) en {tiempo_fin_analisis:.2f}s (Muestra {indice_fin_analisis}).")


    # Recortamos las señales a la zona de análisis
    tiempo_analisis = tiempo_s[indice_inicio_analisis:indice_fin_analisis]
    senal_presion_analisis = senal_presion_mmhg[indice_inicio_analisis:indice_fin_analisis]
    senal_filtrada_analisis = senal_filtrada[indice_inicio_analisis:indice_fin_analisis]

except Exception as e:
    print(f"Error al detectar la zona de desinflado: {e}")
    # Fallback: usar toda la señal (menos probable que funcione bien)
    indice_pico_presion = 0
    indice_inicio_analisis = 0
    indice_fin_analisis = n_muestras - 1
    tiempo_analisis = tiempo_s
    senal_presion_analisis = senal_presion_mmhg
    senal_filtrada_analisis = senal_filtrada

# %% --- 6. Cálculo de la Envolvente y Detección de PAM, Sistólica y Diastólica ---

print("Calculando envolvente y presiones...")
# 1. Calcular la envolvente de la señal filtrada (oscilaciones)
# NOTA: La envolvente se calcula sobre la señal filtrada (en Voltios),
# ya que solo nos importa su forma relativa.
envolvente_abs = np.abs(signal.hilbert(senal_filtrada_analisis))

# 2. Suavizar la envolvente para encontrar el pico (PAM) de forma robusta
longitud_ventana_muestras = int(LONGITUD_VENTANA_SUAVIZADO_S * FS)
if longitud_ventana_muestras < 1: 
    longitud_ventana_muestras = 1
# Asegurarse de que la ventana no sea más grande que la propia señal
if longitud_ventana_muestras > len(envolvente_abs):
    longitud_ventana_muestras = len(envolvente_abs)
    
kernel_suavizado = np.ones(longitud_ventana_muestras) / longitud_ventana_muestras
envolvente_suavizada = np.convolve(envolvente_abs, kernel_suavizado, mode='same')

# --- Inicializar variables de resultados con valores de fallback ---
# Esto asegura que las variables existan para los gráficos incluso si el 'try' falla
idx_sys_global = indice_inicio_analisis
idx_map_global = indice_inicio_analisis
idx_dias_global = indice_fin_analisis # El final de la zona de análisis

# Asegurarse de que los índices de fallback sean válidos
if idx_sys_global >= n_muestras: idx_sys_global = n_muestras - 1
if idx_map_global >= n_muestras: idx_map_global = n_muestras - 1
if idx_dias_global >= n_muestras: idx_dias_global = n_muestras - 1
    
presion_sys_mmhg = senal_presion_mmhg[idx_sys_global]
presion_map_mmhg = senal_presion_mmhg[idx_map_global]
presion_dias_mmhg = senal_presion_mmhg[idx_dias_global]

t_sys = tiempo_s[idx_sys_global]
t_map = tiempo_s[idx_map_global]
t_dias = tiempo_s[idx_dias_global]

amp_map_pico = 0
amp_sys_target = 0
amp_dias_target = 0
idx_map_local = 0


# 3. Encontrar el pico de la envolvente (corresponde a la PAM)
try:
    if len(envolvente_suavizada) == 0:
        raise ValueError("La zona de análisis está vacía. Ajusta los parámetros.")

    idx_map_local = np.argmax(envolvente_suavizada)
    amp_map_pico = envolvente_suavizada[idx_map_local]
    idx_map_global = indice_inicio_analisis + idx_map_local
    
    # Validar índice
    if idx_map_global >= n_muestras: idx_map_global = n_muestras - 1
        
    t_map = tiempo_s[idx_map_global]
    # Obtener el valor de presión de la señal convertida a mmHg
    presion_map_mmhg = senal_presion_mmhg[idx_map_global]
    
    print(f"  - Pico de PAM detectado en t={t_map:.2f}s")

    # 4. Calcular umbrales para Sistólica y Diastólica
    amp_sys_target = RATIO_SISTOLICA * amp_map_pico
    amp_dias_target = RATIO_DIASTOLICA * amp_map_pico

    # 5. Encontrar Presión Sistólica (antes del pico de PAM)
    envolvente_pre_map = envolvente_suavizada[:idx_map_local]
    # Buscamos el último punto donde la envolvente es MENOR o IGUAL al umbral sistólico
    indices_sys_candidatos = np.where(envolvente_pre_map <= amp_sys_target)[0]
    
    if len(indices_sys_candidatos) > 0:
        idx_sys_local = indices_sys_candidatos[-1]
    else:
        print("  - Advertencia: No se encontró cruce sistólico. Usando inicio de zona.")
        idx_sys_local = 0
        
    idx_sys_global = indice_inicio_analisis + idx_sys_local
    if idx_sys_global >= n_muestras: idx_sys_global = n_muestras - 1
        
    t_sys = tiempo_s[idx_sys_global]
    # Obtener el valor de presión de la señal convertida a mmHg
    presion_sys_mmhg = senal_presion_mmhg[idx_sys_global]
    
    # 6. Encontrar Presión Diastólica (después del pico de PAM)
    envolvente_post_map = envolvente_suavizada[idx_map_local:]
    # Buscamos el primer punto donde la envolvente es MENOR o IGUAL al umbral diastólico
    indices_dias_candidatos = np.where(envolvente_post_map <= amp_dias_target)[0]
    
    if len(indices_dias_candidatos) > 0:
        idx_dias_local_relativo = indices_dias_candidatos[0]
    else:
        print("  - Advertencia: No se encontró cruce diastólico. Usando fin de zona.")
        idx_dias_local_relativo = len(envolvente_post_map) - 1

    # El índice global es el índice del pico MAP (idx_map_global) + el índice relativo post-map
    idx_dias_global = idx_map_global + idx_dias_local_relativo
    
    # Asegurarse de que el índice no se salga de los límites
    if idx_dias_global >= n_muestras:
        idx_dias_global = n_muestras - 1
        
    t_dias = tiempo_s[idx_dias_global]
    # Obtener el valor de presión de la señal convertida a mmHg
    presion_dias_mmhg = senal_presion_mmhg[idx_dias_global]

    # --- 7. Impresión de Resultados ---
    print("\n--- Resultados del Análisis (en mmHg) ---")
    print(f"  Presión Sistólica (t={t_sys:.2f}s):   {presion_sys_mmhg:.1f} mmHg")
    print(f"  Presión Diastólica (t={t_dias:.2f}s): {presion_dias_mmhg:.1f} mmHg")
    print(f"  Presión Media (PAM) (t={t_map:.2f}s):  {presion_map_mmhg:.1f} mmHg")

except Exception as e:
    print(f"Error durante el cálculo de la envolvente o picos: {e}")
    # Los valores de fallback ya fueron asignados al inicio de la sección 6,
    # por lo que las variables existirán para los gráficos.
    # Solo imprimimos un mensaje de error.
    print("\nNo se pudieron calcular las presiones, se usarán valores de fallback para graficar.")


# %% --- 8. Visualización de Resultados ---

print("Generando gráficos...")

# Gráfico 1: Señales completas (Original y Filtrada)
fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.set_title(f'Análisis Completo de la Señal - {INPUT_FILENAME}')
ax1.plot(tiempo_s, senal_presion_mmhg, label='Señal Original (Presión)', color='blue', linewidth=1.5)
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Presión (mmHg)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.6)

# Eje Y secundario para la señal filtrada
ax2 = ax1.twinx()
ax2.plot(tiempo_s, senal_filtrada, label='Señal Filtrada (Oscilaciones)', color='red', linewidth=0.8, alpha=0.7)
ax2.set_ylabel('Amplitud Filtrada (V)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Líneas de eventos
ax1.axvline(x=tiempo_s[indice_pico_presion], color='gray', linestyle='--', label=f'Pico Inflado ({tiempo_s[indice_pico_presion]:.2f}s)')
ax1.axvline(x=tiempo_s[indice_inicio_analisis], color='cyan', linestyle='--', label=f'Inicio Análisis ({tiempo_s[indice_inicio_analisis]:.2f}s)')
ax1.axvline(x=tiempo_s[indice_fin_analisis], color='magenta', linestyle='--', label=f'Fin Análisis ({tiempo_s[indice_fin_analisis]:.2f}s)')
ax1.axvline(x=t_sys, color='green', linestyle=':', linewidth=2, label=f'Sistólica ({presion_sys_mmhg:.1f} mmHg)')
ax1.axvline(x=t_map, color='purple', linestyle='-', linewidth=2, label=f'PAM ({presion_map_mmhg:.1f} mmHg)')
ax1.axvline(x=t_dias, color='orange', linestyle=':', linewidth=2, label=f'Diastólica ({presion_dias_mmhg:.1f} mmHg)')

fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
fig.tight_layout()
plt.show()


# Gráfico 2: Detalle de la fase de desinflado
fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.set_title('Detalle de la Fase de Desinflado y Envolvente')

# Señal original (presión)
ax1.plot(tiempo_analisis, senal_presion_analisis, label='Presión Original (mmHg)', color='blue', linewidth=2)
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Presión (mmHg)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Eje Y secundario para la envolvente
ax2 = ax1.twinx()
ax2.plot(tiempo_analisis, envolvente_suavizada, label='Envolvente Suavizada', color='purple', linewidth=2.5)
ax2.set_ylabel('Amplitud de Oscilación (V)', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Umbrales y Picos
# Línea de pico MAP
if idx_map_local < len(tiempo_analisis):
    ax2.plot(tiempo_analisis[idx_map_local], amp_map_pico, 'x', color='black', markersize=10, label='Pico PAM (Envolvente)')

# Líneas de umbral
ax2.axhline(y=amp_sys_target, color='green', linestyle=':', label=f'Umbral Sistólica ({RATIO_SISTOLICA*100:.0f}%)')
ax2.axhline(y=amp_dias_target, color='orange', linestyle=':', label=f'Umbral Diastólica ({RATIO_DIASTOLICA*100:.0f}%)')

# Líneas verticales que conectan todo
ax1.axvline(x=t_sys, color='green', linestyle=':', linewidth=2, label=f'Sistólica ({presion_sys_mmhg:.1f} mmHg)')
ax1.axvline(x=t_map, color='purple', linestyle='-', linewidth=2, label=f'PAM ({presion_map_mmhg:.1f} mmHg)')
ax1.axvline(x=t_dias, color='orange', linestyle=':', linewidth=2, label=f'Diastólica ({presion_dias_mmhg:.1f} mmHg)')

# Puntos de cruce en la envolvente
idx_sys_local_plot = idx_sys_global - indice_inicio_analisis
idx_dias_local_plot = idx_dias_global - indice_inicio_analisis

# Añadir comprobación de límites para los índices de ploteo
if 0 <= idx_sys_local_plot < len(envolvente_suavizada):
    ax2.plot(t_sys, envolvente_suavizada[idx_sys_local_plot], 'o', color='green', markersize=8)
if 0 <= idx_dias_local_plot < len(envolvente_suavizada):
    ax2.plot(t_dias, envolvente_suavizada[idx_dias_local_plot], 'o', color='orange', markersize=8)


fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
fig.tight_layout()
plt.grid(True)
plt.show()

# %% --- 9. Guardado de Datos ---

try:
    # Guardamos Tiempo, Voltaje Original, Presión en mmHg, y Voltaje Filtrado
    datos_para_guardar = np.c_[tiempo_s, senal_original_v, senal_presion_mmhg, senal_filtrada]
    np.savetxt(OUTPUT_FILENAME,
               datos_para_guardar,
               fmt='%-20.8f', # Formato unificado para todas las columnas
               header='Tiempo_s              Voltaje_Original      Presion_mmHg          Voltaje_Filtrado',
               comments='')
    print(f"\nProceso finalizado. Datos de filtrado guardados en '{OUTPUT_FILENAME}'.")
except Exception as e:
    print(f"\nError al guardar el archivo de salida: {e}")