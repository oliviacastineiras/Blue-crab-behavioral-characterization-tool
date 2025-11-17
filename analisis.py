# analisis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2  
from scipy.stats import mannwhitneyu, kruskal
import os

def cargar_y_preparar_datos(
    ruta_archivo_csv,
    fps,
    interpolation_method='linear',
    interpolation_order=2,
    max_gap_frames=5,
    outlier_percentile=100,
    smoothing_window_size=1,
    puntos_calibracion_px=None,
    dimensiones_reales_mm=None,
    pixeles_por_mm=None
):
    """
    Carga datos, calibra (con manejo de errores mejorado) y aplica post-procesado.
    Pipeline: Interpolate -> Filter Outliers -> Interpolate Again -> Smooth
    """
    try:
        # Usar os.path.basename para mensajes de error más limpios
        nombre_csv = os.path.basename(ruta_archivo_csv)
        # Validar FPS primero
        if not isinstance(fps, (int, float)) or fps <= 0:
             print(f"Error FATAL ({nombre_csv}): FPS inválido proporcionado ({fps}). Abortando.")
             return None # Abortar si FPS no es válido
        TIEMPO_POR_FOTOGRAMA_S = 1.0 / fps
        df = pd.read_csv(ruta_archivo_csv)
        if df.empty:
            print(f"Advertencia: El archivo CSV '{nombre_csv}' está vacío.")
            return None

        # Cálculo de centroides:
        if not all(col in df.columns for col in ['x1', 'y1', 'x2', 'y2']):
             print(f"Error: Faltan columnas de coordenadas ('x1', 'y1', 'x2', 'y2') en {nombre_csv}")
             return None
        df['center_x_px'] = (df['x1'] + df['x2']) / 2
        df['center_y_px'] = (df['y1'] + df['y2']) / 2

        # Calibración
        calibrated_ok = False # Flag para saber si se crearon las columnas _mm
        if puntos_calibracion_px and dimensiones_reales_mm:
            print(f"DEBUG ({nombre_csv}): Intentando calibración visual (perspectiva)...")
            try:
                ancho_mm, alto_mm = dimensiones_reales_mm
                # Asegurar formato correcto para puntos de origen
                puntos_origen = np.array(puntos_calibracion_px, dtype="float32")
                if puntos_origen.shape != (4, 2):
                    raise ValueError(f"Formato incorrecto de puntos_origen. Esperado (4, 2), recibido {puntos_origen.shape}")

                puntos_destino = np.array([[0, 0], [ancho_mm, 0], [ancho_mm, alto_mm], [0, alto_mm]], dtype="float32")

                matriz_transformacion = cv2.getPerspectiveTransform(puntos_origen, puntos_destino)
                if matriz_transformacion is None:
                    raise RuntimeError("cv2.getPerspectiveTransform devolvió None. Revisa los 4 puntos marcados.")

                # Seleccionar puntos válidos del DataFrame para transformar
                validos = df.dropna(subset=['center_x_px', 'center_y_px'])
                if validos.empty:
                    print(f"Advertencia ({nombre_csv}): No hay coordenadas (px) válidas para transformar.")
                    df['center_x_mm'] = np.nan
                    df['center_y_mm'] = np.nan
                    # Consideramos que se "intentó" calibrar, aunque no hubiera datos
                    calibrated_ok = True # O False si quiero que falle si no hay puntos válidos
                else:
                    puntos_a_transformar_px = validos[['center_x_px', 'center_y_px']].values.astype('float32')
                    # Reshape a (N, 1, 2) como espera cv2.perspectiveTransform
                    puntos_a_transformar_reshaped = puntos_a_transformar_px.reshape(-1, 1, 2)

                    puntos_transformados_mm = cv2.perspectiveTransform(puntos_a_transformar_reshaped, matriz_transformacion)

                    if puntos_transformados_mm is None:
                        raise RuntimeError("cv2.perspectiveTransform devolvió None.")

                    # Crear columnas _mm inicializadas con NaN
                    df['center_x_mm'] = np.nan
                    df['center_y_mm'] = np.nan

                    # Asignar los resultados usando el índice de los puntos válidos
                    # El resultado tiene shape (N, 1, 2), accedemos a x:[..., 0, 0], y:[..., 0, 1]
                    df.loc[validos.index, 'center_x_mm'] = puntos_transformados_mm[:, 0, 0]
                    df.loc[validos.index, 'center_y_mm'] = puntos_transformados_mm[:, 0, 1]
                    calibrated_ok = True
                    print(f"DEBUG ({nombre_csv}): Calibración visual completada con éxito.")

            except (ValueError, RuntimeError, TypeError, cv2.error) as e_calib:
                print(f"Error CRÍTICO durante la calibración visual para {nombre_csv}: {e_calib}")
                # No creamos las columnas o las dejamos como NaN si ya existen
                df['center_x_mm'] = np.nan
                df['center_y_mm'] = np.nan
                

        elif pixeles_por_mm is not None and pixeles_por_mm > 0:
            print(f"DEBUG ({nombre_csv}): Aplicando calibración manual (escala)...")
            df['center_x_mm'] = df['center_x_px'] / pixeles_por_mm
            df['center_y_mm'] = df['center_y_px'] / pixeles_por_mm
            calibrated_ok = True
        else:
            print(f"Advertencia ({nombre_csv}): No se proporcionaron datos de calibración válidos. Las métricas en mm no estarán disponibles.")
            # Crear columnas NaN para evitar errores posteriores, pero estarán vacías
            df['center_x_mm'] = np.nan
            df['center_y_mm'] = np.nan
            # calibrated_ok sigue siendo False

        # Post-Procesado (Solo si la calibración produjo columnas _mm válidas)
        # Añadimos la comprobación de calibrated_ok y si hay algún dato no-NaN en _mm
        if calibrated_ok and df['center_x_mm'].notna().any():
            print(f"DEBUG ({nombre_csv}): Iniciando pipeline de post-procesado...")
            TIEMPO_POR_FOTOGRAMA_S = 1.0 / fps if fps > 0 else 0 # Calcular una sola vez

            # 1. Primera Interpolación
            if interpolation_method != 'none' and max_gap_frames > 0:
                print(f"DEBUG ({nombre_csv}): Interpolando huecos iniciales (método={interpolation_method}, max_gap={max_gap_frames})...")
                if interpolation_method == 'polynomial':
                    df['center_x_mm'] = df['center_x_mm'].interpolate(method=interpolation_method, order=interpolation_order, limit=max_gap_frames, limit_direction='both')
                    df['center_y_mm'] = df['center_y_mm'].interpolate(method=interpolation_method, order=interpolation_order, limit=max_gap_frames, limit_direction='both')
                else: # linear u otros métodos
                    df['center_x_mm'] = df['center_x_mm'].interpolate(method=interpolation_method, limit=max_gap_frames, limit_direction='both')
                    df['center_y_mm'] = df['center_y_mm'].interpolate(method=interpolation_method, limit=max_gap_frames, limit_direction='both')

            # 2. Filtrado de Atípicos
            # Solo si hay tiempo por fotograma válido
            if TIEMPO_POR_FOTOGRAMA_S > 0:
                delta_x_temp = df['center_x_mm'].diff()
                delta_y_temp = df['center_y_mm'].diff()
                # Usamos fillna(0) temporalmente para el cálculo de velocidad_temp
                velocidad_temp = np.sqrt(delta_x_temp.fillna(0)**2 + delta_y_temp.fillna(0)**2) / TIEMPO_POR_FOTOGRAMA_S

                if 0 < outlier_percentile < 100:
                    # Calcular umbral solo sobre valores válidos de velocidad_temp
                    valid_velocities = velocidad_temp.dropna()
                    if not valid_velocities.empty:
                        velocity_threshold = valid_velocities.quantile(outlier_percentile / 100.0)
                        # Identificar índices donde la velocidad (no NaN) supera el umbral
                        outlier_indices = df[velocidad_temp.notna() & (velocidad_temp > velocity_threshold)].index

                        if not outlier_indices.empty:
                            print(f"DEBUG ({nombre_csv}): Eliminando {len(outlier_indices)} outliers (velocidad > {velocity_threshold:.2f} mm/s, percentil={outlier_percentile}%)")
                            cols_to_null = ['center_x_mm', 'center_y_mm', 'center_x_px', 'center_y_px'] # También NaN en _px
                            df.loc[outlier_indices, cols_to_null] = np.nan

                            # 3. Segunda Interpolación (si se crearon huecos)
                            if interpolation_method != 'none' and max_gap_frames > 0:
                                print(f"DEBUG ({nombre_csv}): Re-interpolando después de filtrar outliers...")
                                if interpolation_method == 'polynomial':
                                     df['center_x_mm'] = df['center_x_mm'].interpolate(method=interpolation_method, order=interpolation_order, limit=max_gap_frames, limit_direction='both')
                                     df['center_y_mm'] = df['center_y_mm'].interpolate(method=interpolation_method, order=interpolation_order, limit=max_gap_frames, limit_direction='both')
                                else:
                                     df['center_x_mm'] = df['center_x_mm'].interpolate(method=interpolation_method, limit=max_gap_frames, limit_direction='both')
                                     df['center_y_mm'] = df['center_y_mm'].interpolate(method=interpolation_method, limit=max_gap_frames, limit_direction='both')
                        else:
                             print(f"DEBUG ({nombre_csv}): No se encontraron outliers con percentil {outlier_percentile}%.")
                    else:
                         print(f"DEBUG ({nombre_csv}): No hay velocidades válidas para calcular el umbral de outliers.")

                # Limpiar variables temporales
                del delta_x_temp, delta_y_temp, velocidad_temp
            else:
                 print(f"DEBUG ({nombre_csv}): Omitiendo filtro de outliers debido a FPS inválido ({fps}).")


            # 4. Suavizado
            if smoothing_window_size > 1:
                 print(f"DEBUG ({nombre_csv}): Aplicando suavizado (ventana={smoothing_window_size})...")
                 df['center_x_mm'] = df['center_x_mm'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()
                 df['center_y_mm'] = df['center_y_mm'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()

            print(f"DEBUG ({nombre_csv}): Pipeline de post-procesado completado.")
        elif not calibrated_ok:
            print(f"DEBUG ({nombre_csv}): Omitiendo post-procesado porque la calibración falló o no se realizó.")
        else: # calibrated_ok is True but no valid _mm data after calibration
             print(f"DEBUG ({nombre_csv}): Omitiendo post-procesado porque no hay datos válidos en mm después de la calibración.")

        # Cálculo de Métricas Finales:
        # Solo si la calibración funcionó, hay tiempo por fotograma válido y hay datos en mm
        if calibrated_ok and TIEMPO_POR_FOTOGRAMA_S > 0 and df['center_x_mm'].notna().any():
            print(f"DEBUG ({nombre_csv}): Calculando métricas finales (velocidad, aceleración)...")
            df['delta_x_mm'] = df['center_x_mm'].diff()
            df['delta_y_mm'] = df['center_y_mm'].diff()
            # Calcular distancia, rellenando NaN con 0 (e.g., para el primer frame)
            df['distancia_recorrida_mm'] = np.sqrt(df['delta_x_mm']**2 + df['delta_y_mm']**2).fillna(0)

            df['velocidad_mms'] = df['distancia_recorrida_mm'] / TIEMPO_POR_FOTOGRAMA_S
            # Rellenar NaNs de velocidad (el primero y posibles NaNs si delta fue NaN)
            df['velocidad_mms'] = df['velocidad_mms'].bfill()
            df['velocidad_mms'] = df['velocidad_mms'].ffill()
            df['velocidad_mms'] = df['velocidad_mms'].fillna(0)

            df['aceleracion_mms2'] = df['velocidad_mms'].diff() / TIEMPO_POR_FOTOGRAMA_S
            # Rellenar NaNs de aceleración (los dos primeros y otros)
            df['aceleracion_mms2'] = df['aceleracion_mms2'].bfill()
            df['aceleracion_mms2'] = df['aceleracion_mms2'].ffill()
            df['aceleracion_mms2'] = df['aceleracion_mms2'].fillna(0)
            print(f"DEBUG ({nombre_csv}): Métricas finales calculadas.")
        else:
            # Si no hubo calibración o FPS inválido, crear columnas vacías o con 0
            if not calibrated_ok:
                 print(f"DEBUG ({nombre_csv}): Creando columnas de métricas vacías (0) porque no hubo calibración.")
            elif TIEMPO_POR_FOTOGRAMA_S <= 0:
                 print(f"DEBUG ({nombre_csv}): Creando columnas de métricas vacías (0) debido a FPS inválido ({fps}).")
            else: # calibrated_ok pero sin datos válidos en mm
                 print(f"DEBUG ({nombre_csv}): Creando columnas de métricas vacías (0) porque no hay datos válidos en mm.")

            #si no se detecto al cangrejo en ningun momento del video, el resultado devuelto será None para que no se tengan en cuenta estos datos a la hora de calcular los resultados
            df['distancia_recorrida_mm'] = np.nan
            df['velocidad_mms'] = np.nan
            df['aceleracion_mms2'] = np.nan

        return df

    except FileNotFoundError:
        print(f"Error FATAL: No se encontró el archivo CSV: {ruta_archivo_csv}")
        return None
    except KeyError as e:
        print(f"Error FATAL: Falta la columna esperada '{e}' en {os.path.basename(ruta_archivo_csv)}")
        return None
    except Exception as e:
        print(f"Error FATAL inesperado al procesar {os.path.basename(ruta_archivo_csv)}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generar_grafico_velocidad(df):
    """Genera la figura del gráfico de velocidad."""
    fig_velocidad = plt.figure(figsize=(6, 4))
    plt.plot(df['frame'], df['velocidad_mms'])
    plt.title('Velocidad vs. Tiempo')
    plt.xlabel('Fotograma')
    plt.ylabel('Velocidad (mm/s)')
    plt.grid(True)
    return fig_velocidad

def generar_grafico_aceleracion(df):
    """Genera la figura del gráfico de aceleración."""
    fig_aceleracion = plt.figure(figsize=(6, 4))
    plt.plot(df['frame'], df['aceleracion_mms2'])
    plt.title('Aceleración vs. Tiempo')
    plt.xlabel('Fotograma')
    plt.ylabel('Aceleración (mm/s²)')
    plt.grid(True)
    return fig_aceleracion

def generar_grafico_movilidad(df, umbral_velocidad, fps):
    """Genera la figura del gráfico de tarta de movilidad."""
    df['estado'] = np.where(df['velocidad_mms'] < umbral_velocidad, 'inmóvil', 'en movimiento')
    
    conteo_estados = df['estado'].value_counts()
    TIEMPO_POR_FOTOGRAMA_S = 1 / fps
    tiempo_inmovil_s = conteo_estados.get('inmóvil', 0) * TIEMPO_POR_FOTOGRAMA_S
    tiempo_movimiento_s = conteo_estados.get('en movimiento', 0) * TIEMPO_POR_FOTOGRAMA_S

    fig_tarta = plt.figure(figsize=(5, 5))
    plt.pie([tiempo_movimiento_s, tiempo_inmovil_s], labels=['En Movimiento', 'Inmóvil'], colors=['#4CAF50', '#FFC107'], autopct='%1.1f%%', startangle=90)
    plt.title(f'Movilidad (Umbral: {umbral_velocidad} mm/s)')
    plt.axis('equal')
    return fig_tarta

def generar_mapa_calor(df, bins_x=50, bins_y=30):
    """Genera la figura del mapa de calor."""
    fig_mapa_calor = plt.figure(figsize=(6, 5))
    
    # Primero, eliminamos las filas donde las coordenadas son NaN para no causar un error.
    df_validos = df.dropna(subset=['center_x_mm', 'center_y_mm'])
    
    # Si después de eliminar los NaN no queda ningún dato, no podemos dibujar el mapa.
    if df_validos.empty:
        print("Advertencia: No hay datos de posición válidos para generar el mapa de calor (posiblemente todos fueron filtrados).")
        # Creamos un gráfico vacío con un mensaje para que el usuario sepa qué ha pasado.
        plt.text(0.5, 0.5, 'No hay datos válidos para mostrar', horizontalalignment='center', verticalalignment='center')
        plt.title('Mapa de Calor de Posición')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        return fig_mapa_calor # Devuelve la figura vacía pero sin crash

    # Si hay datos válidos, procedemos a dibujar el mapa de calor como antes.
    plt.hist2d(x=df_validos['center_x_mm'], y=df_validos['center_y_mm'], bins=(bins_x, bins_y), cmap='inferno', norm=LogNorm())    
    plt.colorbar(label='Frecuencia')
    plt.title('Mapa de Calor de Posición')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.axis('equal')

    plt.gca().invert_yaxis()
    return fig_mapa_calor

def generar_grafico_trayectoria(df, background_frame=None):
    """
    Genera la figura que muestra la trayectoria.
    Si se proporciona un 'background_frame', lo usa como fondo y dibuja en píxeles.
    Si no, dibuja en milímetros sobre un fondo blanco.
    """
    fig_trayectoria = plt.figure(figsize=(8, 6))
    ax = fig_trayectoria.add_subplot(1, 1, 1)

    if background_frame is not None:
        # MODO CON IMAGEN DE FONDO
        x_coords, y_coords = df['center_x_px'], df['center_y_px']
        x_label, y_label = 'Posición X (píxeles)', 'Posición Y (píxeles)'
        height, width, _ = background_frame.shape
        ax.imshow(background_frame, extent=[0, width, height, 0])
        
    else:
        # MODO NORMAL (SIN FONDO)
        x_coords, y_coords = df['center_x_mm'], df['center_y_mm']
        x_label, y_label = 'Posición X (mm)', 'Posición Y (mm)'

    # Dibujar la trayectoria (común para ambos modos)
    ax.plot(x_coords, y_coords, color='cyan', linewidth=2, label='Trayectoria')
    if not x_coords.empty and not y_coords.empty:
        # Usamos dropna() para encontrar el primer y último punto válido
        valid_coords = df.dropna(subset=['center_x_px', 'center_y_px']) if background_frame is not None else df.dropna(subset=['center_x_mm', 'center_y_mm'])
        if not valid_coords.empty:
            ax.plot(valid_coords.iloc[0][x_coords.name], valid_coords.iloc[0][y_coords.name], 
                     marker='o', color='#32CD32', markersize=10, markeredgecolor='black', label='Inicio')
            ax.plot(valid_coords.iloc[-1][x_coords.name], valid_coords.iloc[-1][y_coords.name], 
                     marker='X', color='#FF4500', markersize=12, markeredgecolor='black', label='Fin')
    
    ax.set_title('Trayectoria Completa')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if background_frame is None:
        ax.set_aspect('equal', adjustable='box')
    
    return fig_trayectoria

def calcular_estadisticas_generales(df):
    """Calcula un resumen de métricas clave."""
    if df is None or df.empty:
        return {}
    
    distancia_total_m = df['distancia_recorrida_mm'].sum() / 1000
    velocidad_media = df['velocidad_mms'].mean()
    velocidad_max = df['velocidad_mms'].max()
    
    stats = {
        "Distancia Total (m)": f"{distancia_total_m:.2f}",
        "Velocidad Media (mm/s)": f"{velocidad_media:.2f}",
        "Velocidad Máxima (mm/s)": f"{velocidad_max:.2f}"
    }
    return stats

def generar_histograma(df, columna, titulo, unidad):
    """Genera un histograma para una columna de datos."""
    fig = plt.figure(figsize=(6, 4))
    plt.hist(df[columna].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title(titulo)
    plt.xlabel(f'{columna.replace("_", " ").capitalize()} ({unidad})')
    plt.ylabel('Frecuencia de Muestras')
    plt.grid(axis='y', alpha=0.75)
    return fig

def generar_grafico_distancia_punto(df, punto_x_mm, punto_y_mm):
    """Genera un gráfico de la distancia a un punto a lo largo del tiempo."""
    distancia = np.sqrt(
        (df['center_x_mm'] - punto_x_mm)**2 + 
        (df['center_y_mm'] - punto_y_mm)**2
    )
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(df['frame'], distancia)
    plt.title(f'Distancia al Punto ({punto_x_mm}, {punto_y_mm})')
    plt.xlabel('Fotograma')
    plt.ylabel('Distancia (mm)')
    plt.grid(True)
    return fig


def analizar_region_de_interes(df, roi_coords, fps, in_pixels=False):
    """
    Calcula el tiempo que el animal pasó dentro de una región de interés.
    Ahora puede trabajar con coordenadas en milímetros (por defecto) o en píxeles.
    """
    if df is None or df.empty:
        return {}

    xmin, ymin, xmax, ymax = roi_coords

    #  Elegimos las columnas correctas
    if in_pixels:
        # Si se indica, usamos las columnas de píxeles
        x_col, y_col = 'center_x_px', 'center_y_px'
    else:
        # Si no, usamos las de milímetros como antes
        x_col, y_col = 'center_x_mm', 'center_y_mm'

    # Filtramos el DataFrame usando las columnas seleccionadas
    df_dentro_roi = df[
        (df[x_col] >= xmin) & (df[x_col] <= xmax) &
        (df[y_col] >= ymin) & (df[y_col] <= ymax)
    ]

    frames_dentro = len(df_dentro_roi)
    frames_totales = len(df)

    tiempo_por_frame = 1 / fps
    tiempo_dentro_s = frames_dentro * tiempo_por_frame
    porcentaje_dentro = (frames_dentro / frames_totales) * 100 if frames_totales > 0 else 0

    return {
        "tiempo_s": f"{tiempo_dentro_s:.2f}",
        "porcentaje": f"{porcentaje_dentro:.2f}%"
    }


def analizar_zonas_caja(df, dimensiones_caja, fps):
    """
    Analiza el tiempo que pasa el cangrejo en diferentes zonas de la caja.

    Parameters:
    df (DataFrame): DataFrame con las columnas 'center_x_mm' y 'center_y_mm'
    dimensiones_caja (tuple): (ancho, alto) de la caja en mm
    fps (int): Fotogramas por segundo del video

    Returns:
    dict: Diccionario con los resultados del análisis por zona
    """
    ancho, alto = dimensiones_caja
    
    # Definir zonas de interés 
    zonas = {
        'esquina_superior_izquierda': (0, 0, ancho/3, alto/3),
        'centro_superior': (ancho/3, 0, 2*ancho/3, alto/3),
        'esquina_superior_derecha': (2*ancho/3, 0, ancho, alto/3),
        'centro_izquierda': (0, alto/3, ancho/3, 2*alto/3),
        'centro': (ancho/3, alto/3, 2*ancho/3, 2*alto/3),
        'centro_derecha': (2*ancho/3, alto/3, ancho, 2*alto/3),
        'esquina_inferior_izquierda': (0, 2*alto/3, ancho/3, alto),
        'centro_inferior': (ancho/3, 2*alto/3, 2*ancho/3, alto),
        'esquina_inferior_derecha': (2*ancho/3, 2*alto/3, ancho, alto)
    }
    
    resultados = {}
    for nombre, (x_min, y_min, x_max, y_max) in zonas.items():
        # Calcular tiempo en esta zona
        en_zona = ((df['center_x_mm'] >= x_min) & (df['center_x_mm'] <= x_max) &
                   (df['center_y_mm'] >= y_min) & (df['center_y_mm'] <= y_max))
        
        frames_en_zona = en_zona.sum()
        tiempo_en_zona = frames_en_zona / fps
        
        resultados[nombre] = {
            'frames': frames_en_zona,
            'tiempo_segundos': round(tiempo_en_zona, 2),
            'porcentaje_total': round((frames_en_zona / len(df)) * 100, 2)
        }
    
    return resultados


def analizar_preferencia_espacial(df, dimensiones_caja, fps):
    """
    Analiza si el cangrejo prefiere bordes o centro.

    Parameters:
    df (DataFrame): DataFrame con las columnas 'center_x_mm' y 'center_y_mm'
    dimensiones_caja (tuple): (ancho, alto) de la caja en mm
    fps (int): Fotogramas por segundo del video

    Returns:
    dict: Diccionario con los resultados del análisis de preferencia
    """
    ancho, alto = dimensiones_caja
    
    # # Definir zona central (50% del área)
    # centro_x_min = ancho * 0.25
    # centro_x_max = ancho * 0.75
    # centro_y_min = alto * 0.25
    # centro_y_max = alto * 0.75
    lado_central = np.sqrt(0.5) # Aprox 0.707
    margen = (1 - lado_central) / 2  # Aprox 0.146
    
    centro_x_min = ancho * margen
    centro_x_max = ancho * (1 - margen)
    centro_y_min = alto * margen
    centro_y_max = alto * (1 - margen)
    
    en_centro = ((df['center_x_mm'] >= centro_x_min) & (df['center_x_mm'] <= centro_x_max) &
                 (df['center_y_mm'] >= centro_y_min) & (df['center_y_mm'] <= centro_y_max))
    
    frames_centro = en_centro.sum()
    frames_bordes = len(df) - frames_centro
    
    return {
        'centro': {
            'frames': frames_centro,
            'tiempo_segundos': round(frames_centro / fps, 2),
            'porcentaje': round((frames_centro / len(df)) * 100, 2)
        },
        'bordes': {
            'frames': frames_bordes,
            'tiempo_segundos': round(frames_bordes / fps, 2),
            'porcentaje': round((frames_bordes / len(df)) * 100, 2)
        }
    }


def generar_visualizacion_zonas(df, dimensiones_caja):
    """
    Genera una visualización gráfica de las zonas de la caja y el tiempo pasado en cada una.

    Parameters:
    df (DataFrame): DataFrame con las columnas 'center_x_mm' y 'center_y_mm'
    dimensiones_caja (tuple): (ancho, alto) de la caja en mm

    Returns:
    Figure: Figura de matplotlib con la visualización
    """
    ancho, alto = dimensiones_caja
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Dibujar la caja
    ax.add_patch(plt.Rectangle((0, 0), ancho, alto, fill=False, edgecolor='black', linewidth=2))
    
    # Dibujar las divisiones de zonas
    # Líneas verticales
    ax.axvline(x=ancho/3, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=2*ancho/3, color='gray', linestyle='--', alpha=0.7)
    
    # Líneas horizontales
    ax.axhline(y=alto/3, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=2*alto/3, color='gray', linestyle='--', alpha=0.7)
    
    # Dibujar la trayectoria
    ax.plot(df['center_x_mm'], df['center_y_mm'], 'b-', alpha=0.5, linewidth=1)
    ax.plot(df['center_x_mm'], df['center_y_mm'], 'bo', alpha=0.5, markersize=2)
    
    # Etiquetar zonas
    zonas = {
        'ESQ\nSUP\nIZQ': (ancho/6, 5*alto/6),
        'CENTRO\nSUPERIOR': (ancho/2, 5*alto/6),
        'ESQ\nSUP\nDER': (5*ancho/6, 5*alto/6),
        'CENTRO\nIZQUIERDA': (ancho/6, alto/2),
        'CENTRO': (ancho/2, alto/2),
        'CENTRO\nDERECHA': (5*ancho/6, alto/2),
        'ESQ\nINF\nIZQ': (ancho/6, alto/6),
        'CENTRO\nINFERIOR': (ancho/2, alto/6),
        'ESQ\nINF\nDER': (5*ancho/6, alto/6)
    }
    
    for texto, (x, y) in zonas.items():
        ax.text(x, y, texto, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.set_aspect('equal')
    ax.set_xlabel('Posición X (mm)')
    ax.set_ylabel('Posición Y (mm)')
    ax.set_title('Análisis de Zonas de la Caja')
    
    return fig


def analizar_conjunto_videos(datos_videos):
    """
    Realiza análisis estadístico sobre un conjunto de videos
    
    Parameters:
    datos_videos (list): Lista de diccionarios con datos de cada video
    
    Returns:
    dict: Resultados del análisis conjunto
    """
    if not datos_videos:
        return {}
    
    # Calcular promedios y desviaciones
    metricas = ['centro', 'bordes', 'distancia', 'velocidad_media', 'velocidad_max']
    resultados = {}
    
    for metrica in metricas:
        valores = [d[metrica] for d in datos_videos if metrica in d]
        if valores:
            resultados[f'promedio_{metrica}'] = sum(valores) / len(valores)
            resultados[f'desviacion_{metrica}'] = np.std(valores)
            resultados[f'min_{metrica}'] = min(valores)
            resultados[f'max_{metrica}'] = max(valores)
    
    return resultados


def generar_reporte_conjunto(datos_videos, nombre_proyecto):
    """
    Genera un reporte textual del análisis conjunto
    
    Parameters:
    datos_videos (list): Lista de diccionarios con datos de cada video
    nombre_proyecto (str): Nombre del proyecto
    
    Returns:
    str: Reporte textual del análisis
    """
    if not datos_videos:
        return "No hay datos para generar el reporte."
    
    # Calcular estadísticas conjuntas
    stats = analizar_conjunto_videos(datos_videos)
    
    reporte = f"REPORTE DE ANÁLISIS CONJUNTO - {nombre_proyecto}\n"
    reporte += "=" * 50 + "\n\n"
    reporte += f"Total de videos analizados: {len(datos_videos)}\n\n"
    
    reporte += "ESTADÍSTICAS DE PREFERENCIA ESPACIAL:\n"
    reporte += f"- Tiempo promedio en centro: {stats.get('promedio_centro', 0):.2f}%\n"
    reporte += f"- Tiempo promedio en bordes: {stats.get('promedio_bordes', 0):.2f}%\n"
    reporte += f"- Rango de tiempo en centro: {stats.get('min_centro', 0):.2f}% - {stats.get('max_centro', 0):.2f}%\n\n"
    
    reporte += "ESTADÍSTICAS DE MOVIMIENTO:\n"
    reporte += f"- Distancia total promedio: {stats.get('promedio_distancia', 0):.2f} m\n"
    reporte += f"- Velocidad media promedio: {stats.get('promedio_velocidad_media', 0):.2f} mm/s\n"
    reporte += f"- Velocidad máxima promedio: {stats.get('promedio_velocidad_max', 0):.2f} mm/s\n\n"
    
    reporte += "ANÁLISIS INDIVIDUAL POR VIDEO:\n"
    for i, datos in enumerate(datos_videos, 1):
        reporte += f"{i}. {datos['nombre']}:\n"
        reporte += f"   - Centro: {datos.get('centro', 0):.2f}%\n"
        reporte += f"   - Bordes: {datos.get('bordes', 0):.2f}%\n"
        reporte += f"   - Distancia: {datos.get('distancia', 0):.2f} m\n"
        reporte += f"   - Velocidad media: {datos.get('velocidad_media', 0):.2f} mm/s\n\n"
    
    return reporte

def calcular_metricas_avanzadas(df, dimensiones_caja, fps):
    """
    Calcula métricas avanzadas de comportamiento
    """
    ancho, alto = dimensiones_caja
    
    # Calcular distancia al centro
    centro_caja = (ancho / 2, alto / 2)
    if 'center_x_mm' in df.columns and 'center_y_mm' in df.columns:
        df['distancia_centro'] = np.sqrt(
            (df['center_x_mm'] - centro_caja[0])**2 + 
            (df['center_y_mm'] - centro_caja[1])**2
        )
        
        # Tiempo cerca del centro (radio de 1/4 del ancho)
        radio_centro = ancho / 4
        cerca_centro = df['distancia_centro'] < radio_centro
        tiempo_centro = cerca_centro.sum() / fps if fps > 0 else 0
    else:
        tiempo_centro = 0
    
    # Para otras métricas avanzadas, por ahora devolvemos valores por defecto
    # si me da tiempo: implementar detección de movimientos circulares y otros patrones después
    return {
        'tiempo_cerca_centro': tiempo_centro,
        'num_movimientos_circulares': 0,  # Placeholder
        'ratio_actividad_inactividad': 0   # Placeholder
    }


def generar_mapa_calor_conjunto(lista_de_dataframes, dimensiones_caja, bins_x=50, bins_y=30):
    """
    Genera un único mapa de calor a partir de una lista de DataFrames de varios vídeos.
    """
    fig_mapa_calor = plt.figure(figsize=(8, 6))

    # Concatenar las coordenadas de todos los dataframes en dos grandes series de Pandas
    todos_x = pd.concat([df['center_x_mm'] for df in lista_de_dataframes], ignore_index=True)
    todos_y = pd.concat([df['center_y_mm'] for df in lista_de_dataframes], ignore_index=True)

    # Eliminar NaNs que puedan existir
    todos_x.dropna(inplace=True)
    todos_y.dropna(inplace=True)

    if todos_x.empty or todos_y.empty:
        print("Advertencia: No hay datos de posición válidos para generar el mapa de calor conjunto.")
        plt.text(0.5, 0.5, 'No hay datos válidos para mostrar', ha='center', va='center')
        plt.title('Mapa de Calor de Posición Conjunto')
        return fig_mapa_calor

    ancho_caja, alto_caja = dimensiones_caja

    # Crear el histograma 2D con todos los datos
    plt.hist2d(x=todos_x, y=todos_y, bins=(bins_x, bins_y), cmap='inferno', norm=LogNorm(), range=[[0, ancho_caja], [0, alto_caja]])
    
    plt.colorbar(label='Frecuencia de Ocupación (Total Frames)')
    plt.title('Mapa de Calor de Posición Conjunto')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.axis('equal') 
    plt.xlim(0, ancho_caja)
    plt.ylim(0, alto_caja)

    plt.gca().invert_yaxis()
    
    return fig_mapa_calor



def calcular_estadisticas_completas(df, umbral_actividad_mms=2.0, dimensiones_caja_mm_para_exploracion=None):
    """
    Calcula un diccionario completo de métricas devolviendo NÚMEROS estándar de Python o None.
    """
    if df is None or df.empty:
        return {
            "Distancia Total (m)": None,
        "Velocidad Media (mm/s)": None,
        "Velocidad Máxima (mm/s)": None,
        "Aceleración Media (mm/s²)": None,
        "Aceleración Máxima (mm/s²)": None,
        "Tiempo Activo (%)": None,
        "Índice de Sinuosidad": None,
        "Distancia Media al Borde (mm)": None,
        "Área Explorada (%)": None,
        "Distancia Media al Centro (mm)": None
        }

    # Función auxiliar para convertir de forma segura a float o devolver None
    def safe_float(value):
        if pd.isna(value) or np.isinf(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Cálculos (con comprobación de existencia de columna)
    # distancia_total_m = df['distancia_recorrida_mm'].sum() / 1000 if 'distancia_recorrida_mm' in df.columns else np.nan
    distancia_total_m = np.nan # Empezar como NaN
    if 'distancia_recorrida_mm' in df.columns:
        if df['distancia_recorrida_mm'].notna().any(): # Si hay al menos UN dato no-NaN
            distancia_total_m = df['distancia_recorrida_mm'].sum() / 1000
        elif df['distancia_recorrida_mm'].isna().all(): # Si TODOS son NaN
            distancia_total_m = np.nan # Mantener como NaN
        else: # Si no hay NaN y la suma es 0 (datos válidos 0.0)
            distancia_total_m = 0.0

    velocidad_media = df['velocidad_mms'].mean() if 'velocidad_mms' in df.columns else np.nan
    # Para max, asegurarse que hay datos no-NaN antes de llamar a max()
    velocidad_max = df['velocidad_mms'].max() if 'velocidad_mms' in df.columns and df['velocidad_mms'].notna().any() else np.nan
    aceleracion_media = df['aceleracion_mms2'].abs().mean() if 'aceleracion_mms2' in df.columns else np.nan
    aceleracion_max = df['aceleracion_mms2'].abs().max() if 'aceleracion_mms2' in df.columns and df['aceleracion_mms2'].notna().any() else np.nan
    
    frames_totales = len(df)

    #PORCENTAJE TIEMPO ACTIVO
    # porcentaje_activo = np.nan
    # if 'velocidad_mms' in df.columns and frames_totales > 0:
    #     frames_activos = df[df['velocidad_mms'] > umbral_actividad_mms].shape[0]
    #     porcentaje_activo = (frames_activos / frames_totales) * 100

    porcentaje_activo = np.nan # Empezar como NaN
    if 'velocidad_mms' in df.columns and frames_totales > 0:
        if df['velocidad_mms'].notna().any(): # Si hay al menos UNA velocidad válida
            frames_activos = df[df['velocidad_mms'] > umbral_actividad_mms].shape[0]
            porcentaje_activo = (frames_activos / frames_totales) * 100
        elif df['velocidad_mms'].isna().all(): # Si TODAS las velocidades son NaN
            porcentaje_activo = np.nan # Mantener como NaN
        else: # Si las velocidades son válidas pero 0.0
            porcentaje_activo = 0.0

    # Sinuosidad
    df_validos = df.dropna(subset=['center_x_mm', 'center_y_mm']) if 'center_x_mm' in df.columns else pd.DataFrame() # DataFrame vacío si no hay _mm
    sinuosidad = np.nan
    if not df_validos.empty and len(df_validos) > 1 and 'distancia_recorrida_mm' in df.columns:
        punto_inicio = df_validos[['center_x_mm', 'center_y_mm']].iloc[0]
        punto_fin = df_validos[['center_x_mm', 'center_y_mm']].iloc[-1]
        distancia_linea_recta_mm = np.linalg.norm(punto_fin.values - punto_inicio.values)
        distancia_total_mm = df['distancia_recorrida_mm'].sum()
        if distancia_linea_recta_mm > 1e-6: # Evitar división por cero o casi cero
            sinuosidad = distancia_total_mm / distancia_linea_recta_mm
        elif distancia_total_mm < 1e-6: # Si no se movió apreciablemente
            sinuosidad = 1.0
        # Si recorrió distancia pero acabó donde empezó, sinuosidad queda como NaN 

    distancia_media_al_borde = df['distancia_al_borde_mm'].mean() if 'distancia_al_borde_mm' in df.columns else np.nan

    #calcular exploraxion:
    porcentaje_exploracion = np.nan # Iniciar como NaN
    if dimensiones_caja_mm_para_exploracion:
        # Llama a la nueva función pasándole el df y las dimensiones
        # La función interna ya maneja si df no tiene datos _mm válidos
        porcentaje_exploracion = calcular_porcentaje_exploracion(df, dimensiones_caja_mm_para_exploracion)

    #calculo distancia media
    distancia_media_centro = np.nan
    if dimensiones_caja_mm_para_exploracion and \
       'center_x_mm' in df.columns and 'center_y_mm' in df.columns and \
       df['center_x_mm'].notna().any():
        try:
            ancho_mm, alto_mm = dimensiones_caja_mm_para_exploracion
            if ancho_mm > 0 and alto_mm > 0: # Asegurar dimensiones positivas
                centro_x_mm = ancho_mm / 2.0
                centro_y_mm = alto_mm / 2.0
                # Calcular la columna de distancias (ignora NaNs en _mm implícitamente)
                distancia_centro_col = np.sqrt(
                    (df['center_x_mm'] - centro_x_mm)**2 +
                    (df['center_y_mm'] - centro_y_mm)**2
                )
                # Calcular la media (mean() ignora NaNs por defecto)
                distancia_media_centro = distancia_centro_col.mean()
        except Exception as e_dist_centro:
            # Capturar posibles errores en el cálculo
            print(f"Advertencia: Error calculando distancia media al centro: {e_dist_centro}")
            distancia_media_centro = np.nan # Asegurar que es NaN si falla

    return {
        "Distancia Total (m)": safe_float(distancia_total_m),
        "Velocidad Media (mm/s)": safe_float(velocidad_media),
        "Velocidad Máxima (mm/s)": safe_float(velocidad_max),
        "Aceleración Media (mm/s²)": safe_float(aceleracion_media),
        "Aceleración Máxima (mm/s²)": safe_float(aceleracion_max),
        "Tiempo Activo (%)": safe_float(porcentaje_activo),
        "Índice de Sinuosidad": safe_float(sinuosidad),
        "Distancia Media al Borde (mm)": safe_float(distancia_media_al_borde),
        "Área Explorada (%)": safe_float(porcentaje_exploracion),
        "Distancia Media al Centro (mm)": safe_float(distancia_media_centro)
    }



def comparar_grupos_estadisticamente(datos_por_grupo, metrica):
    """
    Compara los grupos para una métrica específica, ignorando valores nulos.
    """
    # 1. Obtenemos las listas de datos, incluyendo los valores None/nan
    listas_de_datos_con_nones = [datos[metrica] for datos in datos_por_grupo.values()]
    
    # 2. Creamos nuevas listas limpias, quitando los valores None y nan
    listas_de_datos = [
        [valor for valor in sublista if valor is not None and not np.isnan(valor)] 
        for sublista in listas_de_datos_con_nones
    ]
    
    # 3. Eliminamos cualquier grupo que se haya quedado sin datos después de la limpieza
    listas_de_datos = [sublista for sublista in listas_de_datos if sublista]

    if len(listas_de_datos) < 2:
        return "N/A", None

    try:
        if len(listas_de_datos) == 2:
            stat, p_valor = mannwhitneyu(listas_de_datos[0], listas_de_datos[1], alternative='two-sided')
            return "U de Mann-Whitney", p_valor
        else:
            stat, p_valor = kruskal(*listas_de_datos)
            return "Kruskal-Wallis", p_valor
    except ValueError:
        return "N/A", 1.0
    
def calcular_distancia_al_borde(df, dimensiones_caja):
    """
    Calcula la distancia mínima al borde del hábitat para cada punto de la trayectoria.
    Añade una nueva columna 'distancia_al_borde_mm' al DataFrame.
    """
    if df is None or df.empty or 'center_x_mm' not in df.columns:
        return df

    ancho_caja, alto_caja = dimensiones_caja
    
    # Coordenadas del cangrejo
    x = df['center_x_mm']
    y = df['center_y_mm']

    # Distancias a las 4 paredes
    dist_izquierda = x
    dist_derecha = ancho_caja - x
    dist_arriba = y
    dist_abajo = alto_caja - y

    # Nos quedamos con la distancia mínima para cada punto
    distancias = pd.concat([dist_izquierda, dist_derecha, dist_arriba, dist_abajo], axis=1)
    df['distancia_al_borde_mm'] = distancias.min(axis=1)
    
    return df

def generar_grafico_distancia_al_borde(df):
    """Genera un gráfico de línea de la distancia al borde a lo largo del tiempo."""
    fig = plt.figure(figsize=(6, 4))
    if 'distancia_al_borde_mm' in df.columns and not df['distancia_al_borde_mm'].dropna().empty:
        plt.plot(df['frame'], df['distancia_al_borde_mm'])
        plt.title('Distancia al Borde vs. Tiempo')
        plt.xlabel('Fotograma')
        plt.ylabel('Distancia al Borde (mm)')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Datos de distancia al borde no disponibles.', ha='center', va='center')
        plt.title('Distancia al Borde vs. Tiempo')
    
    return fig


def calcular_porcentaje_exploracion(df, dimensiones_caja_mm, tamaño_celda_mm=10):
    """
    Calcula el porcentaje del área total de la caja que ha sido visitada.

    Args:
        df (pd.DataFrame): DataFrame con 'center_x_mm' y 'center_y_mm'.
        dimensiones_caja_mm (tuple): (ancho_real_mm, alto_real_mm).
        tamaño_celda_mm (int, optional): Lado de cada celda de la rejilla en mm. Defaults to 10.

    Returns:
        float: Porcentaje del área explorada (0-100), o np.nan si no se puede calcular.
    """
    # Comprobaciones iniciales de los datos de entrada
    if df is None or df.empty or 'center_x_mm' not in df.columns or df['center_x_mm'].isna().all():
        #print(f"DEBUG Exploracion: No hay datos mm válidos.")
        return np.nan # Devolver NaN si no hay datos de coordenadas válidos

    if not dimensiones_caja_mm or len(dimensiones_caja_mm) != 2:
        #print(f"DEBUG Exploracion: Dimensiones de caja inválidas: {dimensiones_caja_mm}")
        return np.nan # Devolver NaN si las dimensiones son inválidas

    ancho_mm, alto_mm = dimensiones_caja_mm
    if ancho_mm <= 0 or alto_mm <= 0 or tamaño_celda_mm <= 0:
        #print(f"DEBUG Exploracion: Dimensiones/tamaño celda <= 0.")
        return np.nan # Devolver NaN si las dimensiones no son positivas

    # Calcular el número total de celdas en la rejilla
    try:
        num_celdas_x = int(np.ceil(ancho_mm / tamaño_celda_mm))
        num_celdas_y = int(np.ceil(alto_mm / tamaño_celda_mm))
        total_celdas = num_celdas_x * num_celdas_y
    except (ValueError, TypeError):
        #print(f"DEBUG Exploracion: Error al calcular número de celdas.")
        return np.nan # Error en cálculo

    if total_celdas == 0:
        #print(f"DEBUG Exploracion: Total celdas es 0.")
        return 0.0 # Si el área es 0 (o tamaño celda infinito), la exploración es 0%

    # Usar un conjunto para almacenar las celdas visitadas (evita duplicados)
    celdas_visitadas = set()
    # Filtrar coordenadas NaN
    coords_validas = df.dropna(subset=['center_x_mm', 'center_y_mm'])

    if coords_validas.empty:
        #print(f"DEBUG Exploracion: No hay coordenadas válidas tras dropna.")
        return 0.0 # Si no hay puntos válidos después de filtrar, exploración es 0%

    # Calcular los índices de celda para cada punto válido
    # Usamos // para división entera
    indices_x = (coords_validas['center_x_mm'] // tamaño_celda_mm).astype(int)
    indices_y = (coords_validas['center_y_mm'] // tamaño_celda_mm).astype(int)

    # Asegurar que los índices estén dentro de los límites de la rejilla
    # np.clip(array, min_val, max_val)
    indices_x = np.clip(indices_x, 0, num_celdas_x - 1)
    indices_y = np.clip(indices_y, 0, num_celdas_y - 1)

    # Añadir cada celda visitada (par de índices) al conjunto
    # zip combina los índices x e y para cada punto
    for idx_x, idx_y in zip(indices_x, indices_y):
        celdas_visitadas.add((idx_x, idx_y))

    # Calcular el porcentaje final
    porcentaje_explorado = (len(celdas_visitadas) / total_celdas) * 100.0
    #print(f"DEBUG Exploracion: {len(celdas_visitadas)} / {total_celdas} celdas visitadas = {porcentaje_explorado:.2f}%")

    return porcentaje_explorado


def generar_grafico_distancia_centro(df, dimensiones_caja_mm):
    """
    Genera un gráfico de la distancia al centro geométrico de la caja vs. tiempo.

    Args:
        df (pd.DataFrame): DataFrame con 'center_x_mm', 'center_y_mm' y 'frame'.
        dimensiones_caja_mm (tuple): (ancho_real_mm, alto_real_mm).

    Returns:
        matplotlib.figure.Figure: La figura del gráfico, o una figura con mensaje de error.
    """
    fig = plt.figure(figsize=(6, 4)) # Tamaño estándar para los gráficos pequeños
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Distancia al Centro vs. Tiempo')
    # Comprobaciones de los datos de entrada
    if df is None or df.empty or 'center_x_mm' not in df.columns or 'center_y_mm' not in df.columns or df['center_x_mm'].isna().all():
        ax.text(0.5, 0.5, 'Datos de trayectoria\ninsuficientes.', ha='center', va='center', transform=ax.transAxes)
        return fig
    if not dimensiones_caja_mm or len(dimensiones_caja_mm) != 2 or not all(isinstance(d, (int, float)) and d > 0 for d in dimensiones_caja_mm):
        ax.text(0.5, 0.5, 'Dimensiones de caja\ninválidas o faltantes.', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Calcular centro
    ancho_mm, alto_mm = dimensiones_caja_mm
    centro_x_mm = ancho_mm / 2.0
    centro_y_mm = alto_mm / 2.0

    # Calcular distancia euclidiana al centro (vectorizado)
    # Solo para filas con coordenadas válidas
    distancia_centro = np.sqrt(
        (df['center_x_mm'] - centro_x_mm)**2 +
        (df['center_y_mm'] - centro_y_mm)**2
    ) # Esto mantendrá NaNs donde las coordenadas eran NaN

    # Graficar contra el número de fotograma (índice o columna 'frame')
    if 'frame' in df.columns:
        x_axis = df['frame']
        x_label = 'Fotograma'
    else:
        x_axis = df.index
        x_label = 'Índice (Fotograma)'

    ax.plot(x_axis, distancia_centro, linewidth=1.5) # Línea un poco más gruesa
    ax.set_xlabel(x_label)
    ax.set_ylabel('Distancia al Centro (mm)')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Añadir línea de distancia máxima (centro a esquina) como referencia
    try:
        dist_max = np.sqrt(centro_x_mm**2 + centro_y_mm**2)
        ax.axhline(dist_max, color='red', linestyle=':', linewidth=1, alpha=0.6, label=f'Max ({dist_max:.0f} mm)')
        ax.legend(fontsize='small', loc='best') # Mejor ubicación automática
    except Exception:
        pass # Ignorar si falla el cálculo de dist_max

    ax.set_ylim(bottom=0) # Asegurar que el eje Y empieza en 0
    # Ajustar límites X si hay columna 'frame'
    if 'frame' in df.columns:
         ax.set_xlim(left=df['frame'].min(), right=df['frame'].max())



    return fig
