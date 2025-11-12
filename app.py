# app.py
import customtkinter as ctk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd 
import threading
import subprocess
import os
import shutil
import yaml
import queue
import cv2  
import json
import numpy as np
import matplotlib.gridspec as gridspec
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import datetime





from analisis import (cargar_y_preparar_datos, generar_grafico_velocidad, calcular_estadisticas_completas, 
                      generar_grafico_movilidad, generar_mapa_calor, 
                      generar_grafico_aceleracion, generar_grafico_trayectoria,
                      calcular_estadisticas_generales, generar_histograma,
                      generar_grafico_distancia_punto, analizar_region_de_interes, 
                      analizar_zonas_caja, analizar_preferencia_espacial,  # Nuevas importaciones
                      generar_visualizacion_zonas, calcular_metricas_avanzadas, generar_mapa_calor_conjunto, 
                      comparar_grupos_estadisticamente,calcular_distancia_al_borde, generar_grafico_distancia_al_borde, generar_grafico_distancia_centro)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Herramienta de Análisis de Comportamiento de Cangrejos")
        self.geometry("1500x850")

        # Variables de Estado 
        self.df_procesado = None
        self.ruta_csv_cargado = None
        self.ruta_video_cargado = None # Variable para el vídeo de fondo
        self.single_analysis_calibration = None 
        self.ruta_ultimo_resultado = None
        self.plot_components = {} 
        self.is_processing = False
        self.output_queue = queue.Queue()
        self.project_data = None
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.analysis_window = None

        self.last_project_analysis_kwargs = None # Almacenará los parámetros del último análisis

        # ESTRUCTURA PRINCIPAL CON GRID (2 COLUMNAS)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        

        # COLUMNA 0: PANEL LATERAL (SIDEBAR)
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        # self.sidebar_frame.grid_columnconfigure(0, weight=1)
        # self.sidebar_frame.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="Acciones Principales", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 10), padx=20, fill="x", anchor="w")
        self.procesar_button = ctk.CTkButton(self.sidebar_frame, text="Procesar Vídeo Único", command=self.iniciar_procesamiento_video)
        self.procesar_button.pack(pady=5, padx=20, fill="x")
        self.analizar_button = ctk.CTkButton(self.sidebar_frame, text="Cargar CSV Único", command=self.cargar_nuevo_csv)
        self.analizar_button.pack(pady=5, padx=20, fill="x")

        #  ESTRUCTURA DEL PANEL DE PARÁMETROS 
        self.params_title_label = ctk.CTkLabel(self.sidebar_frame, text="Parámetros de Análisis", font=ctk.CTkFont(size=16, weight="bold"))
        self.params_title_label.pack(pady=(20, 5), padx=20, fill="x", anchor="w")

        self.params_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.params_frame.pack(pady=0, padx=15, fill="x") # params_frame se añade con pack
        self.params_frame.grid_columnconfigure(1, weight=1)

        #Subsección Calibración:
        ctk.CTkLabel(self.params_frame, text="Calibración", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,5))
        ctk.CTkLabel(self.params_frame, text="Escala (px/mm):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.escala_entry = ctk.CTkEntry(self.params_frame, width=70); self.escala_entry.insert(0, "2.75"); self.escala_entry.grid(row=1, column=1, sticky="e", padx=5, pady=2)
        ctk.CTkLabel(self.params_frame, text="FPS Video:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.fps_entry = ctk.CTkEntry(self.params_frame, width=70); self.fps_entry.insert(0, "20"); self.fps_entry.grid(row=2, column=1, sticky="e", padx=5, pady=2)

        # Subsección Post-Procesado:
        ctk.CTkLabel(self.params_frame, text="Post-Procesado", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10,5))

        ctk.CTkLabel(self.params_frame, text="Interpolación:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.interp_menu = ctk.CTkOptionMenu(self.params_frame, width=90, values=['linear', 'polynomial', 'none'], anchor="center")
        self.interp_menu.set('linear') # Valor por defecto
        self.interp_menu.grid(row=4, column=1, sticky="e", padx=5, pady=2)

        ctk.CTkLabel(self.params_frame, text="Max Gap (frames):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        self.interp_gap_entry = ctk.CTkEntry(self.params_frame, width=70); self.interp_gap_entry.insert(0, "5"); self.interp_gap_entry.grid(row=5, column=1, sticky="e", padx=5, pady=2)

        ctk.CTkLabel(self.params_frame, text="Eliminar Atípicos (%):").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        self.outlier_entry = ctk.CTkEntry(self.params_frame, width=70); self.outlier_entry.insert(0, "100"); self.outlier_entry.grid(row=6, column=1, sticky="e", padx=5, pady=2)

        self.smoothing_label = ctk.CTkLabel(self.params_frame, text="Suavizado (ventana): 1"); self.smoothing_label.grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=(5,0))
        self.smoothing_slider = ctk.CTkSlider(self.params_frame, from_=1, to=25, number_of_steps=24, command=lambda v: self.smoothing_label.configure(text=f"Suavizado (ventana): {int(v)}"));
        self.smoothing_slider.set(1); # Valor por defecto
        self.smoothing_slider.grid(row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=(0,10))
    

        #boton aplicar cambios:
        self.apply_button = ctk.CTkButton(self.sidebar_frame, text="Aplicar Cambios y Refrescar", command=self.refrescar_analisis)
        self.apply_button.pack(pady=15, padx=20, fill="x")

        #boton gestion de proyecto:
        self.view_button = ctk.CTkButton(self.sidebar_frame, text="Gestión de Proyecto", command=self.mostrar_vista_proyecto)
        self.view_button.pack(pady=5, padx=20, fill="x")

        # Checkbox global para renderizado de vídeo único
        self.render_video_checkbox = ctk.CTkCheckBox(self.sidebar_frame, text="Renderizar vídeo al procesar")
        self.render_video_checkbox.pack(pady=5, padx=20, anchor="w")
        self.render_video_checkbox.select()

        self.status_container = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.status_container.pack(pady=(10, 10), padx=10, fill="x", side="bottom")
        self.status_label = ctk.CTkLabel(self.status_container, text="Bienvenido", wraplength=230)
        self.status_label.pack(fill="x", pady=5)
        self.progress_bar = ctk.CTkProgressBar(self.status_container)
        
        self.notification_frame = ctk.CTkFrame(self.status_container, fg_color="transparent")

        self.notification_frame.grid_columnconfigure(0, weight=1)
        self.notification_frame.grid_columnconfigure(1, weight=0)
        self.notification_frame.grid_rowconfigure(0, weight=1)

        self.notification_label = ctk.CTkLabel(self.notification_frame, text="", anchor="w")
        self.notification_label.grid(row=0, column=0, padx=(5, 10), pady=5, sticky="ew")

        self.notification_button = ctk.CTkButton(self.notification_frame, text="Cargar", command=self.cargar_resultados_notificados, width=70)
        # Colocar el botón en la columna 1, alineado a la derecha (este)
        self.notification_button.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="e")
        
        # COLUMNA 1: ÁREA DE CONTENIDO PRINCIPAL
        self.main_frame = ctk.CTkScrollableFrame(self, label_text="Resultados del Análisis de Archivo Único")
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        self.project_frame = ctk.CTkFrame(self)
        
        self.after(100, self.procesar_cola_mensajes)




    def _get_directorio_trabajo(self):
        """Función centralizada para obtener la ruta raíz del proyecto de forma robusta."""
        return os.path.dirname(os.path.abspath(__file__))
    def obtener_primer_frame(self, ruta_video):
        """Extrae el primer frame de un video para calibración"""
        if not os.path.exists(ruta_video):
            self.status_label.configure(text=f"Error: No se encontró el vídeo {os.path.basename(ruta_video)}", text_color="red")
            return None
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            self.status_label.configure(text="Error: No se pudo abrir el vídeo.", text_color="red")
            return None
        ret, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
    

    def solicitar_opcion_calibracion(self):
        """Muestra un diálogo para que el usuario elija el método de calibración."""
        if not self.ruta_video_cargado:
            # Si no se encontró vídeo, se usa la escala manual por defecto
            self.status_label.configure(text="No se encontró vídeo asociado. Usando escala manual.", text_color="orange")
            self.refrescar_analisis()
            return

        # Crear la ventana de diálogo
        dialog = ctk.CTkToplevel(self)
        dialog.title("Método de Calibración")
        dialog.geometry("450x200")
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="¿Cómo deseas calibrar este vídeo?", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)

        def on_select(method):
            dialog.destroy()
            if method == "visual":
                self.iniciar_calibracion_unica()
            elif method == "manual":
                self.refrescar_analisis() # Procede con la escala del panel

        ctk.CTkButton(dialog, text="Calibrar Visualmente (Recomendado)", command=lambda: on_select("visual")).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(dialog, text="Usar Escala Manual del Panel", command=lambda: on_select("manual"), fg_color="gray").pack(pady=10, padx=20, fill="x")

    def iniciar_calibracion_unica(self):
        """Inicia el proceso de calibración visual para un vídeo único."""
        frame = self.obtener_primer_frame(self.ruta_video_cargado)
        if frame is None:
            self.status_label.configure(text="Error: No se pudo obtener el fotograma para calibrar.", text_color="red")
            return
            
        self.calib_window = ctk.CTkToplevel(self)
        self.calib_window.title(f"Calibrando Vídeo Único")
        self.calib_window.geometry("1000x700")
        self.calib_window.grab_set()
        
        self.esquinas_calibracion = []
        # Usamos self.video_calibracion_actual para saber que estamos en este modo
        self.video_calibracion_actual = "__SINGLE_VIDEO__" 
                
        ctk.CTkLabel(self.calib_window, text="Marca las 4 esquinas de la pecera en orden. Pulsa 'ESC' para cancelar.").pack(pady=10)
                
        fig, self.ax_calib = plt.subplots(1, 1); self.ax_calib.imshow(frame)
        self.canvas_calib = FigureCanvasTkAgg(fig, master=self.calib_window)
        self.canvas_calib.draw(); self.canvas_calib.get_tk_widget().pack(fill='both', expand=True)
                
        self.status_label.configure(text=f"Calibrando. Marca las 4 esquinas.")
        
        self.cid_click = self.canvas_calib.mpl_connect('button_press_event', self.on_calib_click)
        self.cid_key = self.canvas_calib.mpl_connect('key_press_event', self.on_calib_key)

    def confirmar_medidas_unica(self):
        """Confirma las medidas y guarda la calibración para el análisis único."""
        try:
            ancho_real = float(self.ancho_entry.get())
            alto_real = float(self.alto_entry.get())
            if ancho_real <= 0 or alto_real <= 0: raise ValueError("Las dimensiones deben ser positivas")

            # Usamos list() o .copy() para crear una nueva lista independiente
            puntos_guardados = list(self.esquinas_calibracion)

            # Guardamos los datos en la variable de estado del análisis único
            self.single_analysis_calibration = {
                # --- Usa la copia ---
                "puntos_px": puntos_guardados,
                # --------------------
                "dimensiones_mm": (ancho_real, alto_real)
            }
            print(f"DEBUG: Calibración única guardada: {self.single_analysis_calibration}") #debujprint

            self.status_label.configure(text="¡Calibración completada! Aplicando análisis...")

            # Ahora podemos cerrar las ventanas sin problema
            if hasattr(self, 'medida_window'): # Comprobar si existe antes de destruir
                self.medida_window.destroy()
            self.cerrar_ventana_calibracion() # Esto borrará self.esquinas_calibracion original

            # Una vez calibrado, refrescamos el análisis automáticamente
            self.refrescar_analisis()

        except ValueError as e:
            self.status_label.configure(text=f"Error al confirmar medidas: {str(e)}", text_color="red")
        except AttributeError:
            # Añadir manejo por si self.esquinas_calibracion no existe por alguna razón
            self.status_label.configure(text="Error: No se encontraron los puntos de calibración.", text_color="red")

    def iniciar_calibracion(self, video_name):
        """Inicia el proceso de calibración para un video específico"""
        if self.project_data is None or video_name not in self.project_data["videos"]:
            self.status_label.configure(text="Error: Video no encontrado en el proyecto", text_color="red")
            return
                
        datos_video = self.project_data["videos"][video_name]
        ruta_video = datos_video.get("ruta_original")
            
        # Si no tenemos ruta original, buscar en la estructura de carpetas
        if not ruta_video or not os.path.exists(ruta_video):
            self.status_label.configure(text=f"Error: No se encuentra el fichero de vídeo original para '{video_name}'", text_color="red")
            return
                
        frame = self.obtener_primer_frame(ruta_video)
        if frame is None:
            self.status_label.configure(text="Error: No se pudo obtener frame para calibración", text_color="red")
            return
                
        # Crear ventana de calibración
        self.calib_window = ctk.CTkToplevel(self)
        self.calib_window.title(f"Calibrando: {video_name}")
        self.calib_window.geometry("1000x700")
        self.calib_window.grab_set()
        
        # Variables para almacenar el estado de calibración
        self.esquinas_calibracion = []
        self.video_calibracion_actual = video_name
            
        ctk.CTkLabel(self.calib_window, text="Marca las 4 esquinas de la pecera en orden (en sentido de las agujas del reloj empezando por la esquina superior izquierda). Pulsa 'ESC' para cancelar.").pack(pady=10)
            
        # Crear figura de matplotlib
        fig, self.ax_calib = plt.subplots(1, 1)
        self.ax_calib.imshow(frame)
        self.canvas_calib = FigureCanvasTkAgg(fig, master=self.calib_window)
        self.canvas_calib.draw()
        self.canvas_calib.get_tk_widget().pack(fill='both', expand=True)
            
        self.status_label.configure(text=f"Calibrando {video_name}. Marca las 4 esquinas.")
        
        # Conectar eventos
        self.cid_click = self.canvas_calib.mpl_connect('button_press_event', self.on_calib_click)
        self.cid_key = self.canvas_calib.mpl_connect('key_press_event', self.on_calib_key)


    def on_calib_click(self, event):
        """Maneja el clic del mouse durante la calibración"""
        if event.inaxes != self.ax_calib:
            return
            
        # Solo procesar si estamos en una calibración activa
        if not hasattr(self, 'esquinas_calibracion') or not hasattr(self, 'calib_window'):
            return
            
        # Agregar punto
        self.esquinas_calibracion.append((event.xdata, event.ydata))
        self.ax_calib.plot(event.xdata, event.ydata, 'r+', markersize=12, markeredgewidth=2)
        self.canvas_calib.draw()
        
        # Actualizar mensaje
        puntos_restantes = 4 - len(self.esquinas_calibracion)
        self.status_label.configure(text=f"Calibrando {self.video_calibracion_actual}. Puntos restantes: {puntos_restantes}")
        
        # Si tenemos 4 puntos, proceder con las medidas
        if len(self.esquinas_calibracion) == 4:
            self.mostrar_ventana_medidas()

    def on_calib_key(self, event):
        """Maneja teclas durante la calibración"""
        if event.key == 'escape':
            self.cerrar_ventana_calibracion()
            self.status_label.configure(text="Calibración cancelada.")

    def mostrar_ventana_medidas(self):
        """Muestra la ventana para ingresar medidas reales"""
        # Calcular dimensiones en píxeles
        x_coords = [p[0] for p in self.esquinas_calibracion]
        y_coords = [p[1] for p in self.esquinas_calibracion]
        ancho_px = max(x_coords) - min(x_coords)
        alto_px = max(y_coords) - min(y_coords)
        
        # Crear ventana de medidas
        self.medida_window = ctk.CTkToplevel(self.calib_window)
        self.medida_window.title("Dimensiones reales")
        self.medida_window.geometry("400x250")
        self.medida_window.grab_set()
        
        ctk.CTkLabel(self.medida_window, text="Ingrese las dimensiones reales de la pecera:").pack(pady=10)
        
        frame_medidas = ctk.CTkFrame(self.medida_window)
        frame_medidas.pack(pady=10)
        
        ctk.CTkLabel(frame_medidas, text="Ancho (mm):").grid(row=0, column=0, padx=5, pady=10)
        self.ancho_entry = ctk.CTkEntry(frame_medidas, width=100)
        self.ancho_entry.grid(row=0, column=1, padx=5, pady=10)
        
        ctk.CTkLabel(frame_medidas, text="Alto (mm):").grid(row=1, column=0, padx=5, pady=10)
        self.alto_entry = ctk.CTkEntry(frame_medidas, width=100)
        self.alto_entry.grid(row=1, column=1, padx=5, pady=10)
        
        # Mostrar dimensiones en píxeles para referencia
        ctk.CTkLabel(frame_medidas, text=f"Ancho en píxeles: {ancho_px:.1f}").grid(row=2, column=0, columnspan=2, pady=5)
        ctk.CTkLabel(frame_medidas, text=f"Alto en píxeles: {alto_px:.1f}").grid(row=3, column=0, columnspan=2, pady=5)
        
        ctk.CTkButton(self.medida_window, text="Confirmar", command=self.confirmar_medidas).pack(pady=10)

    def confirmar_medidas(self):
        """Confirma las medidas ingresadas y calcula la escala"""
        if hasattr(self, 'video_calibracion_actual') and self.video_calibracion_actual == "__SINGLE_VIDEO__":
            self.confirmar_medidas_unica()
            return
        try:
            ancho_real = float(self.ancho_entry.get())
            alto_real = float(self.alto_entry.get())
            
            if ancho_real <= 0 or alto_real <= 0:
                raise ValueError("Las dimensiones deben ser positivas")
            
            # Calcular dimensiones en píxeles
            x_coords = [p[0] for p in self.esquinas_calibracion]
            y_coords = [p[1] for p in self.esquinas_calibracion]
            ancho_px = max(x_coords) - min(x_coords)
            alto_px = max(y_coords) - min(y_coords)
            
            # Calcular escala
            escala_x = ancho_px / ancho_real
            escala_y = alto_px / alto_real
            escala_promedio = (escala_x + escala_y) / 2
            
            # Guardar calibración - AÑADIR DIMENSIONES REALES
            self.project_data["videos"][self.video_calibracion_actual]["calibracion_px"] = self.esquinas_calibracion
            self.project_data["videos"][self.video_calibracion_actual]["escala_px_mm"] = escala_promedio
            self.project_data["videos"][self.video_calibracion_actual]["dimensiones_caja_mm"] = (ancho_real, alto_real)
            
            # Actualizar estado a "Calibrado" si no estaba procesado
            if self.project_data["videos"][self.video_calibracion_actual]["estado"] != "Procesado":
                self.project_data["videos"][self.video_calibracion_actual]["estado"] = "Calibrado"
            
            # Actualizar campo de escala en la interfaz
            self.escala_entry.delete(0, "end")
            self.escala_entry.insert(0, f"{escala_promedio:.2f}")
            
            self.refrescar_lista_videos_proyecto()
            self.status_label.configure(text=f"¡Calibración completada! Escala: {escala_promedio:.2f} px/mm")
            
            # Cerrar ventanas
            self.medida_window.destroy()
            self.cerrar_ventana_calibracion()
            
        except ValueError as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="red")

    def cerrar_ventana_calibracion(self):
        """Cierra la ventana de calibración y limpia los recursos"""
        if hasattr(self, 'canvas_calib') and hasattr(self, 'cid_click') and hasattr(self, 'cid_key'):
            self.canvas_calib.mpl_disconnect(self.cid_click)
            self.canvas_calib.mpl_disconnect(self.cid_key)
        
        if hasattr(self, 'calib_window'):
            self.calib_window.destroy()
        
        # Limpiar variables de calibración
        if hasattr(self, 'esquinas_calibracion'):
            del self.esquinas_calibracion
        if hasattr(self, 'video_calibracion_actual'):
            del self.video_calibracion_actual



    def refrescar_analisis(self):
        if not self.ruta_csv_cargado:
            self.status_label.configure(text="No hay archivo CSV cargado.", text_color="orange")
            return
        
        print(f"DEBUG: Entrando en refrescar_analisis. self.single_analysis_calibration = {self.single_analysis_calibration}")

        self.limpiar_vistas_principales()

        try:
            # Recoger parámetros de Calibración
            fps = int(self.fps_entry.get())
            # (La escala solo se usa si no hay calibración visual)

            # Recoger parámetros de Post-Procesado 
            interp_method = self.interp_menu.get()
            interp_gap = int(self.interp_gap_entry.get())
            outlier_text = self.outlier_entry.get()
            outlier_perc = float(outlier_text) if outlier_text else 100.0
            smooth_window = int(self.smoothing_slider.get())

            # Validaciones básicas (puedes añadir más)
            if fps <= 0: raise ValueError("FPS debe ser positivo")
            if interp_gap < 0: raise ValueError("Max Gap no puede ser negativo")
            if not (0 <= outlier_perc <= 100): raise ValueError("Percentil de atípicos debe estar entre 0 y 100")
            if smooth_window < 1: raise ValueError("Ventana de suavizado debe ser >= 1")

        except ValueError as e:
            self.status_label.configure(text=f"Error en parámetros: {e}", text_color="orange")
            return

        # Preparar argumentos para cargar_y_preparar_datos
        kwargs_carga = {
            "ruta_archivo_csv": self.ruta_csv_cargado,
            "fps": fps,
            "interpolation_method": interp_method,
            "max_gap_frames": interp_gap,
            "outlier_percentile": outlier_perc,
            "smoothing_window_size": smooth_window,
            # Añadimos orden por si se usa 'polynomial'
            "interpolation_order": 2
        }

        # Añadir argumentos de Calibración
        dimensiones_caja = None
        if self.single_analysis_calibration:
            kwargs_carga["puntos_calibracion_px"] = self.single_analysis_calibration["puntos_px"]
            kwargs_carga["dimensiones_reales_mm"] = self.single_analysis_calibration["dimensiones_mm"]
            dimensiones_caja = self.single_analysis_calibration["dimensiones_mm"]
            # Si hay calibración visual, actualizamos el entry de escala como referencia
            # (Cálculo aproximado, la transformación perspectiva es más compleja)
            try:
                puntos = self.single_analysis_calibration["puntos_px"]
                dims = self.single_analysis_calibration["dimensiones_mm"]
                ancho_px = np.linalg.norm(np.array(puntos[1]) - np.array(puntos[0]))
                alto_px = np.linalg.norm(np.array(puntos[3]) - np.array(puntos[0]))
                esc_x = ancho_px / dims[0]
                esc_y = alto_px / dims[1]
                self.escala_entry.delete(0, 'end')
                self.escala_entry.insert(0, f"{(esc_x + esc_y) / 2:.2f}")
            except Exception: pass # Ignorar si falla el cálculo
        else:
            # Usar escala manual del panel
            try:
                escala_manual = float(self.escala_entry.get())
                if escala_manual <= 0: raise ValueError("Escala debe ser positiva")
                kwargs_carga["pixeles_por_mm"] = escala_manual
                if self.ruta_video_cargado:
                    try:
                        frame = self.obtener_primer_frame(self.ruta_video_cargado)
                        if frame is not None:
                            h, w, _ = frame.shape
                            dimensiones_caja = (w / escala_manual, h / escala_manual) # Guardar dimensiones estimadas
                    except Exception: pass # Ignorar si falla la estimación
            except ValueError as e:
                self.status_label.configure(text=f"Error en Escala (px/mm): {e}", text_color="orange")
                return

        # Llamada a cargar_y_preparar_datos
        self.status_label.configure(text="Aplicando análisis...")
        try:
            # Usamos **kwargs_carga para pasar todos los parámetros
            self.df_procesado = cargar_y_preparar_datos(**kwargs_carga)

            if self.df_procesado is None or self.df_procesado.empty:
                self.status_label.configure(text="Error: El procesado resultó en datos vacíos.", text_color="orange")
                return

            # Calcular distancia al borde (si hay calibración visual y dimensiones)
            if self.single_analysis_calibration and "dimensiones_mm" in self.single_analysis_calibration:
                 self.df_procesado = calcular_distancia_al_borde(self.df_procesado, self.single_analysis_calibration["dimensiones_mm"])
            elif "pixeles_por_mm" in kwargs_carga and self.ruta_video_cargado:
                 # Intentar estimar dimensiones si hay escala y video
                 try:
                     frame = self.obtener_primer_frame(self.ruta_video_cargado)
                     if frame is not None:
                         h, w, _ = frame.shape
                         dims_estimadas = (w / kwargs_carga["pixeles_por_mm"], h / kwargs_carga["pixeles_por_mm"])
                         self.df_procesado = calcular_distancia_al_borde(self.df_procesado, dims_estimadas)
                 except Exception: pass # Ignorar si falla

        except Exception as e:
             self.status_label.configure(text=f"Error durante el análisis: {e}", text_color="red")
             import traceback
             traceback.print_exc()
             return

        # Construir y Rellenar la GUI
        self.construir_interfaz_resultados()
        self.actualizar_estadisticas()
        self.actualizar_grafico_trayectoria()
        self.actualizar_mapa_calor()
        self.actualizar_grafico_movilidad()
        self.actualizar_graficos_simples() # Actualiza borde si existe la columna

        self.actualizar_grafico_distancia_centro(dimensiones_caja)

        if not self.is_processing:
            self.status_label.configure(text=f"Análisis completado: {os.path.basename(self.ruta_csv_cargado)}")

    def construir_interfaz_resultados(self):
        self._crear_componente_estadisticas(row=0, column=0, columnspan=2)#estadisticas
        self._crear_componente_principal(row=1, column=0, columnspan=2)#tray interactiva
        self._crear_componente_movilidad(row=2, column=0) #movilidad
        self._crear_componente_mapa_calor(row=2, column=1)# mapa calor
        #velocidad y aceleracion:
        self.plot_components['velocidad_line'] = {'parent': self._crear_contenedor_grafico("Velocidad vs. Tiempo", row=3, column=0)}
        self.plot_components['aceleracion_line'] = {'parent': self._crear_contenedor_grafico("Aceleración vs. Tiempo", row=3, column=1)}
        self.plot_components['velocidad_hist'] = {'parent': self._crear_contenedor_grafico("Distribución de Velocidad", row=4, column=0)}
        self.plot_components['aceleracion_hist'] = {'parent': self._crear_contenedor_grafico("Distribución de Aceleración", row=4, column=1)}
        #si hay calibracion hacemos distancia al borde:
        # if self.single_analysis_calibration:
        self.plot_components['distancia_borde'] = {'parent': self._crear_contenedor_grafico("Distancia al Borde vs. Tiempo", row=5, column=0)}
    #y distancia al centro:
        self.plot_components['distancia_centro_container'] = {'parent': self._crear_contenedor_grafico("Distancia al Centro vs. Tiempo", row=5, column=1)}

    def _crear_contenedor_grafico(self, titulo, row, column):
        container = ctk.CTkFrame(self.main_frame)
        container.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(container, text=titulo, font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
        plot_frame = ctk.CTkFrame(container, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        return plot_frame
        
    def _crear_componente_estadisticas(self, row, column, columnspan):
        """Crea el frame superior para mostrar las métricas clave."""
        frame = ctk.CTkFrame(self.main_frame)
        frame.grid(row=row, column=column, columnspan=columnspan, padx=5, pady=5, sticky="ew")

        # AÑADIR el nuevo título a la lista
        stats_titulos = [
            "Distancia Total (m)",
            "Velocidad Media (mm/s)",
            "Velocidad Máxima (mm/s)",
            "Área Explorada (%)"
        ]

        component_dict = {}
        num_stats = len(stats_titulos)
        # Crear un grid dentro del frame para que se ajusten bien
        frame.grid_columnconfigure(list(range(num_stats)), weight=1)

        for i, titulo in enumerate(stats_titulos):
            sub_frame = ctk.CTkFrame(frame)
            sub_frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew") # Usar grid
            ctk.CTkLabel(sub_frame, text=titulo, font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
            label_valor = ctk.CTkLabel(sub_frame, text="---", font=ctk.CTkFont(size=16))
            label_valor.pack(pady=(0,5))
            component_dict[titulo] = label_valor

        # Guardar la referencia al diccionario de labels
        self.plot_components['estadisticas'] = component_dict
    
    def _crear_componente_principal(self, row, column, columnspan):
        container = ctk.CTkFrame(self.main_frame)
        container.grid(row=row, column=column, columnspan=columnspan, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(container, text="Análisis Espacial Interactivo", font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
        plot_frame = ctk.CTkFrame(container, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        controls_frame = ctk.CTkFrame(container)
        controls_frame.pack(fill="x", padx=5, pady=5)
        roi_button = ctk.CTkButton(controls_frame, text="Definir Región (ROI)", command=self.iniciar_definicion_roi)
        roi_button.pack(side="left", padx=10, pady=5)
        ctk.CTkLabel(controls_frame, text="Tiempo:", font=ctk.CTkFont(weight="bold")).pack(side="left")
        roi_tiempo_label = ctk.CTkLabel(controls_frame, text="--- s")
        roi_tiempo_label.pack(side="left", padx=5)
        ctk.CTkLabel(controls_frame, text="%:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(10,0))
        roi_porcentaje_label = ctk.CTkLabel(controls_frame, text="--- %")
        roi_porcentaje_label.pack(side="left", padx=5)
        dist_button = ctk.CTkButton(controls_frame, text="Calcular Distancia", command=self.actualizar_distancia_punto)
        dist_button.pack(side="right", padx=10, pady=5)
        punto_y_entry = ctk.CTkEntry(controls_frame, width=60)
        punto_y_entry.pack(side="right", padx=(2, 5))
        ctk.CTkLabel(controls_frame, text="Y:").pack(side="right")
        punto_x_entry = ctk.CTkEntry(controls_frame, width=60)
        punto_x_entry.pack(side="right", padx=(2, 5))
        ctk.CTkLabel(controls_frame, text="X:").pack(side="right")
        select_point_button = ctk.CTkButton(controls_frame, text="Seleccionar Punto...", command=self.abrir_ventana_seleccion)
        select_point_button.pack(side="right", padx=10, pady=5)
        self.plot_components['principal'] = {'plot_frame': plot_frame, 'roi_tiempo_label': roi_tiempo_label, 'roi_porcentaje_label': roi_porcentaje_label, 'punto_x_entry': punto_x_entry, 'punto_y_entry': punto_y_entry}

    def _crear_componente_mapa_calor(self, row, column):
        container = ctk.CTkFrame(self.main_frame)
        container.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
        controls_frame = ctk.CTkFrame(container, fg_color="transparent")
        controls_frame.pack(fill="x", padx=10)
        ctk.CTkLabel(controls_frame, text="Rejilla (X,Y):").pack(side="left", padx=(0, 5))
        bins_x_entry = ctk.CTkEntry(controls_frame, width=50)
        bins_x_entry.insert(0, "50")
        bins_x_entry.pack(side="left", padx=5)
        bins_y_entry = ctk.CTkEntry(controls_frame, width=50)
        bins_y_entry.insert(0, "30")
        bins_y_entry.pack(side="left", padx=5)
        heatmap_button = ctk.CTkButton(controls_frame, text="Actualizar Mapa", command=self.actualizar_mapa_calor, height=20)
        heatmap_button.pack(side="left", padx=10)
        plot_frame = ctk.CTkFrame(container, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.plot_components['mapa_calor'] = {'bins_x': bins_x_entry, 'bins_y': bins_y_entry, 'plot_frame': plot_frame}

    def _crear_componente_movilidad(self, row, column):
        container = ctk.CTkFrame(self.main_frame)
        container.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
        controls_frame = ctk.CTkFrame(container, fg_color="transparent")
        controls_frame.pack(fill="x", padx=10, pady=10)
        slider_label = ctk.CTkLabel(controls_frame, text="Umbral Inmóvil (mm/s): 2.0")
        slider_label.pack(side="left")
        slider = ctk.CTkSlider(controls_frame, from_=0, to=10, number_of_steps=100)
        slider.set(2.0)
        slider.pack(side="left", padx=10, expand=True, fill="x")
        slider.configure(command=self.actualizar_grafico_movilidad)
        plot_frame = ctk.CTkFrame(container, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.plot_components['movilidad'] = {'slider_label': slider_label, 'slider': slider, 'plot_frame': plot_frame}

    def actualizar_estadisticas(self):
        """Actualiza los labels de estadísticas en la vista de análisis único."""
        component = self.plot_components.get('estadisticas')
        if not component: return # Si el componente no existe, salir

        # Si no hay datos procesados, poner '---' en todos los labels y salir
        if self.df_procesado is None or self.df_procesado.empty:
            for label in component.values():
                label.configure(text="---")
            return

        # Obtener dimensiones de la caja (NECESARIO para Exploración)
        dimensiones_caja = None
        # Prioridad 1: Calibración visual guardada
        if self.single_analysis_calibration and "dimensiones_mm" in self.single_analysis_calibration:
             dimensiones_caja = self.single_analysis_calibration["dimensiones_mm"]
             print("DEBUG actualizar_estadisticas: Usando dimensiones de calibración visual.")
        # Prioridad 2: Intentar estimar si hay vídeo y escala manual válida
        elif self.ruta_video_cargado:
             try:
                 escala_manual = float(self.escala_entry.get())
                 if escala_manual > 0:
                     frame = self.obtener_primer_frame(self.ruta_video_cargado)
                     if frame is not None:
                         h, w, _ = frame.shape
                         dimensiones_caja = (w / escala_manual, h / escala_manual)
                         print(f"DEBUG actualizar_estadisticas: Estimando dimensiones {dimensiones_caja} con escala manual.")
                     else:
                          print("DEBUG actualizar_estadisticas: No se pudo obtener frame para estimar dimensiones.")
                 else:
                      print("DEBUG actualizar_estadisticas: Escala manual no válida para estimar dimensiones.")
             except (ValueError, TypeError):
                 print("DEBUG actualizar_estadisticas: Error al leer escala manual para estimar dimensiones.")
                 pass # No hacer nada si la escala no es válida o no hay vídeo
        else:
             print("DEBUG actualizar_estadisticas: No hay calibración visual ni vídeo/escala para obtener dimensiones.")


        # Llamar a calcular_estadisticas_completas pasando las dimensiones
        # La función en analisis.py ahora devuelve None para métricas si no se pueden calcular
        estadisticas = calcular_estadisticas_completas(
            self.df_procesado,
            dimensiones_caja_mm_para_exploracion=dimensiones_caja # Pasa las dimensiones (o None)
        )
        print(f"DEBUG actualizar_estadisticas: Estadísticas calculadas: {estadisticas}") # Debug

        #  Actualizar los labels 
        for titulo, label_widget in component.items():
            valor_numerico = estadisticas.get(titulo) # Obtiene el número (float) o None

            # Formatear aquí para mostrar en la GUI
            if valor_numerico is not None:
                # Formato específico para porcentajes
                if titulo in ["Área Explorada (%)", "Tiempo Activo (%)"]:
                    texto_valor = f"{valor_numerico:.1f}%"
                # Formato para metros o índice sinuosidad (más decimales)
                elif titulo in ["Distancia Total (m)", "Índice de Sinuosidad"]:
                     texto_valor = f"{valor_numerico:.2f}"
                # Formato general para mm/s, mm/s², mm (un decimal)
                else:
                    texto_valor = f"{valor_numerico:.1f}"
            else:
                # Si el valor es None (porque no se pudo calcular), mostrar N/A
                texto_valor = "N/A"

            # Actualizar el texto del label
            label_widget.configure(text=texto_valor)

    def actualizar_grafico_trayectoria(self):
        component = self.plot_components.get('principal')
        if not component or self.df_procesado is None: return

        background_image = None
        # Comprobamos si la lógica automática encontró una ruta de vídeo válida
        if self.ruta_video_cargado and os.path.exists(self.ruta_video_cargado):
            background_image = self.obtener_primer_frame(self.ruta_video_cargado)

        # Le pasamos la imagen (o None si no la encontró) a la función de análisis
        fig = generar_grafico_trayectoria(self.df_procesado, background_frame=background_image)

        self.embed_figure(fig, component['plot_frame'], name_for_canvas='trayectoria')

    def actualizar_mapa_calor(self):
        component = self.plot_components.get('mapa_calor')
        if not component: return
        try:
            bins_x = int(component['bins_x'].get() or 50)
            bins_y = int(component['bins_y'].get() or 30)
            fig = generar_mapa_calor(self.df_procesado, bins_x=bins_x, bins_y=bins_y)
            self.embed_figure(fig, component['plot_frame'])
        except (ValueError, KeyError): pass

    def actualizar_grafico_movilidad(self, nuevo_umbral=None):
        component = self.plot_components.get('movilidad')
        if not component or self.df_procesado is None: return
        if nuevo_umbral is None: nuevo_umbral = component['slider'].get()
        component['slider_label'].configure(text=f"Umbral Inmóvil (mm/s): {float(nuevo_umbral):.1f}")
        try:
            fps = int(self.fps_entry.get())
            fig = generar_grafico_movilidad(self.df_procesado.copy(), float(nuevo_umbral), fps) 
            self.embed_figure(fig, component['plot_frame'])
        except (ValueError, KeyError): pass
    
    def actualizar_graficos_simples(self):
        if self.df_procesado is None: return
        #histogramas y lineas de vel/acel
        self.embed_figure(generar_histograma(self.df_procesado, 'velocidad_mms', '', 'mm/s'), self.plot_components['velocidad_hist']['parent'])
        self.embed_figure(generar_histograma(self.df_procesado, 'aceleracion_mms2', '', 'mm/s²'), self.plot_components['aceleracion_hist']['parent'])
        self.embed_figure(generar_grafico_velocidad(self.df_procesado), self.plot_components['velocidad_line']['parent'])
        self.embed_figure(generar_grafico_aceleracion(self.df_procesado), self.plot_components['aceleracion_line']['parent'])
        if 'distancia_borde' in self.plot_components:
            self.embed_figure(generar_grafico_distancia_al_borde(self.df_procesado), self.plot_components['distancia_borde']['parent'])
    
    def actualizar_grafico_distancia_centro(self, dimensiones_caja):
        """Actualiza el gráfico de distancia al centro."""
        # Busca el contenedor usando la nueva clave
        component = self.plot_components.get('distancia_centro_container')
        # Salir si no existe el contenedor o no hay datos procesados
        if not component or self.df_procesado is None: return

        # Generar el gráfico llamando a la nueva función de analisis.py
        # La función interna ya maneja si df o dimensiones_caja son inválidos
        fig = generar_grafico_distancia_centro(self.df_procesado, dimensiones_caja)

        # Incrustar la figura en el frame padre del componente
        self.embed_figure(fig, component['parent'])

    def embed_figure(self, fig, parent_frame, name_for_canvas=None):
        for widget in parent_frame.winfo_children(): widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        if name_for_canvas:
            if 'canvas_store' not in self.plot_components: self.plot_components['canvas_store'] = {}
            self.plot_components['canvas_store'][name_for_canvas] = canvas
        plt.close(fig)

    def limpiar_vistas_principales(self):
        for widget in self.main_frame.winfo_children(): widget.destroy()
        self.plot_components.clear()
    
    def on_closing(self):
        print("Cerrando la aplicación de forma segura...")
        try:
            plt.close('all')
            self.quit()
        finally: self.destroy()

    def procesar_cola_mensajes(self):
        try:
            message = self.output_queue.get_nowait()
            
            if message.startswith("[DONE]"): 
                self.notificar_procesamiento_completado(message.split(" ", 1)[1])
                
            elif message.startswith("[DONE_PROYECTO]"):
                # Formato: "[DONE_PROYECTO] ruta_csv|video_name"
                partes = message.split(" ", 1)[1].split("|")
                if len(partes) == 2:
                    ruta_csv, video_name = partes
                    self.notificar_procesamiento_proyecto_completado(ruta_csv, video_name)
                    
            elif message.startswith("[ERROR_PROYECTO]"):
                # Formato: "[ERROR_PROYECTO] mensaje_error|video_name"
                partes = message.split(" ", 1)[1].split("|")
                if len(partes) == 2:
                    error_msg, video_name = partes
                    self.finalizar_procesamiento_proyecto_con_error(error_msg, video_name)
                    
            elif message.startswith("[ERROR]"): 
                self.finalizar_procesamiento_con_error(message.split(" ", 1)[1])
                
        except queue.Empty: 
            pass
        finally: 
            self.after(100, self.procesar_cola_mensajes)

    def _actualizar_estado_botones_procesar(self):
        """
        Actualiza el estado de TODOS los botones de "Procesar" de la aplicación.
        """
        # 1. Botón principal
        if hasattr(self, 'procesar_button'):
            nuevo_estado = "disabled" if self.is_processing else "normal"
            self.procesar_button.configure(state=nuevo_estado)

        # 2. Refresca toda la lista del proyecto para asegurar que todos los botones se actualizan
        if hasattr(self, 'project_video_list_frame'):
            self.refrescar_lista_videos_proyecto()

    def iniciar_procesamiento_video(self):
        if self.is_processing: return
        ruta_video = filedialog.askopenfilename(filetypes=(("Archivos MP4", "*.mp4"),))
        if not ruta_video: return

        directorio_trabajo = self._get_directorio_trabajo()
        nombre_video = os.path.basename(ruta_video)
        nombre_base = nombre_video.split('.')[0]
        ruta_csv_existente = os.path.join(directorio_trabajo, 'results', 'stats', f"{nombre_base}_interpolated_denormalized.csv")
        
        if os.path.exists(ruta_csv_existente):
            self.status_label.configure(text=f"Resultados encontrados para '{nombre_video}'.")
            self.notificar_procesamiento_completado(ruta_csv_existente)
            return
        
        self.is_processing = True
        self._actualizar_estado_botones_procesar()
        self.notification_frame.pack_forget()
        self.status_label.pack(fill="x")
        self.status_label.configure(text=f"Procesando: {os.path.basename(ruta_video)}...")
        self.progress_bar.pack(fill="x", padx=10, pady=2)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        threading.Thread(target=self._ejecutar_tracking_en_hilo, args=(ruta_video,), daemon=True).start()

    def _ejecutar_tracking_en_hilo(self, ruta_video):
        nombre_video_con_extension = os.path.basename(ruta_video)
        directorio_trabajo = os.path.dirname(os.path.abspath(__file__))
        
        # Definimos el nombre base UNA SOLA VEZ 
        nombre_base = nombre_video_con_extension.split('.')[0]
        
        try:
            ruta_conf = os.path.join(directorio_trabajo, 'config', 'run_conf.yaml')
            with open(ruta_conf, 'r') as f: config_data = yaml.safe_load(f)
            
            if self.render_video_checkbox.get():
                config_data['output']['render_videos'] = [nombre_base]
            else:
                config_data['output']['render_videos'] = []

            with open(ruta_conf, 'w') as f: yaml.dump(config_data, f)
        except Exception as e:
            self.output_queue.put(f"[ERROR] No se pudo modificar el run_conf.yaml: {e}")
            return

        ruta_carpeta_videos = os.path.join(directorio_trabajo, 'videos')
        try:
            if not os.path.exists(ruta_carpeta_videos): os.makedirs(ruta_carpeta_videos)
            for archivo in os.listdir(ruta_carpeta_videos): os.remove(os.path.join(ruta_carpeta_videos, archivo))
            
            # Copiamos el vídeo RENOMBRÁNDOLO con el nombre base
            nuevo_nombre_video = f"{nombre_base}.mp4"
            shutil.copy(ruta_video, os.path.join(ruta_carpeta_videos, nuevo_nombre_video))

        except Exception as e:
            self.output_queue.put(f"[ERROR] Error al preparar archivos: {e}")
            return

        ruta_docker_compose = os.path.join(directorio_trabajo, 'docker-hub-docker-compose.yaml')
        comando = ["docker", "compose", "-f", ruta_docker_compose, "up"]
        try:
            process = subprocess.Popen(comando, cwd=directorio_trabajo, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            _, stderr = process.communicate()
            if process.returncode == 0:
                ruta_csv = os.path.join(directorio_trabajo, 'results', 'stats', f"{nombre_base}_interpolated_denormalized.csv")
                self.output_queue.put(f"[DONE] {ruta_csv}")
            else:
                self.output_queue.put(f"[ERROR] {stderr}")
        except Exception as e:
            self.output_queue.put(f"[ERROR] {str(e)}")

    def notificar_procesamiento_completado(self, ruta_csv):
        self.is_processing = False
        self.ruta_ultimo_resultado = ruta_csv
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.notification_label.configure(text=f"¡'{os.path.basename(ruta_csv)}' listo!")
        self.notification_frame.pack(fill="x", pady=(5,0))
        self._actualizar_estado_botones_procesar() # Actualiza la UI
        
    def finalizar_procesamiento_con_error(self, error_msg):
        self.is_processing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.notification_frame.pack_forget() # Asegurar que está oculto
        self.status_label.pack(fill="x", pady=5)
        self.status_label.configure(text=f"Error en el procesamiento.", text_color="red")
        print(f"--- ERROR DE DOCKER O PROCESO ---\n{error_msg}")
        self._actualizar_estado_botones_procesar() # Actualiza la UI

    def cargar_resultados_notificados(self):
        if self.ruta_ultimo_resultado and os.path.exists(self.ruta_ultimo_resultado):
            self.cargar_nuevo_csv(self.ruta_ultimo_resultado)
        self.notification_frame.pack_forget()
        self.status_label.pack(fill="x", pady=5)

    def cargar_nuevo_csv(self, ruta_archivo=None):
        if not self.is_processing:
            self.notification_frame.pack_forget()
            self.status_label.pack(fill="x", pady=5)

        if ruta_archivo is None:
            ruta_archivo = filedialog.askopenfilename(
                title="Selecciona el archivo CSV con los resultados",
                filetypes=(("Archivos CSV", "*.csv"),)
            )
        if not ruta_archivo: return
        
        self.ruta_csv_cargado = ruta_archivo
        self.ruta_video_cargado = None
        self.single_analysis_calibration = None

        try:
            nombre_csv = os.path.basename(ruta_archivo)
            # Usamos la lógica robusta para nombres con puntos
            nombre_base = nombre_csv.replace("_interpolated_denormalized.csv", "").split('.')[0]
            nombre_video = f"{nombre_base}.mp4"
            
            directorio_trabajo = os.path.dirname(os.path.abspath(__file__))
            
            # Búsqueda automática
            ruta_video_posible = os.path.join(directorio_trabajo, 'videos', nombre_video)
            if os.path.exists(ruta_video_posible):
                self.ruta_video_cargado = ruta_video_posible
            else:
                ruta_video_renderizado = os.path.join(directorio_trabajo, 'results', 'videos', nombre_video)
                if os.path.exists(ruta_video_renderizado):
                    self.ruta_video_cargado = ruta_video_renderizado

            # Si la búsqueda automática falla, preguntamos al usuario
            if self.ruta_video_cargado is None:
                self.status_label.configure(text=f"No se encontró el vídeo '{nombre_video}' automáticamente. Por favor, localízalo.", text_color="orange")
                ruta_manual = filedialog.askopenfilename(
                    title=f"Localiza el vídeo original para '{nombre_csv}'",
                    filetypes=(("Archivos MP4", "*.mp4"),)
                )
                if ruta_manual:
                    self.ruta_video_cargado = ruta_manual

        except Exception as e:
            print(f"Error al intentar encontrar el vídeo asociado: {e}")

        # Ahora llamamos al diálogo de selección, que funcionará si tenemos una ruta de vídeo
        self.solicitar_opcion_calibracion()

    def abrir_ventana_seleccion(self):
        if self.df_procesado is None: return
        selection_window = ctk.CTkToplevel(self)
        selection_window.title("Selecciona un punto")
        selection_window.geometry("800x600")
        ctk.CTkLabel(selection_window, text="Haz clic en el mapa para seleccionar un punto de interés.").pack(pady=10)
        fig = generar_grafico_trayectoria(self.df_procesado)
        canvas = FigureCanvasTkAgg(fig, master=selection_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        def on_click(event):
            if event.xdata is not None and event.ydata is not None:
                component = self.plot_components.get('principal')
                if component:
                    component['punto_x_entry'].delete(0, 'end')
                    component['punto_x_entry'].insert(0, f"{event.xdata:.2f}")
                    component['punto_y_entry'].delete(0, 'end')
                    component['punto_y_entry'].insert(0, f"{event.ydata:.2f}")
                selection_window.destroy()
        fig.canvas.mpl_connect('button_press_event', on_click)
        selection_window.grab_set()

    def iniciar_definicion_roi(self):
        if self.df_procesado is None: return
        try:
            canvas = self.plot_components.get('canvas_store', {}).get('trayectoria')
            if not canvas: return
            ax = canvas.figure.axes[0]
        except (KeyError, IndexError): return
        
        self.roi_rect_patch = None
        self.roi_start_point = None
        self.status_label.configure(text="Modo 'Definir Región': Clic y arrastra sobre la trayectoria.", text_color="cyan")

        def on_press(event):
            if event.inaxes != ax: return
            self.roi_start_point = (event.xdata, event.ydata)
            if hasattr(self, 'roi_rect_patch') and self.roi_rect_patch is not None:
                self.roi_rect_patch.remove()
            self.roi_rect_patch = patches.Rectangle(self.roi_start_point, 0, 0, linewidth=1.5, edgecolor='lime', facecolor='lime', alpha=0.3)
            ax.add_patch(self.roi_rect_patch)
            canvas.draw()

        def on_motion(event):
            if event.inaxes != ax or self.roi_start_point is None: return
            x0, y0 = self.roi_start_point
            x1, y1 = event.xdata, event.ydata
            if x1 is None or y1 is None: return
            self.roi_rect_patch.set_width(x1 - x0)
            self.roi_rect_patch.set_height(y1 - y0)
            self.roi_rect_patch.set_xy((x0, y0))
            canvas.draw()

        def on_release(event):
            if self.roi_start_point is None: return
            x0, y0 = self.roi_start_point
            x1, y1 = event.xdata, event.ydata
            if x1 is None or y1 is None: return
            
            self.roi_start_point = None
            canvas.mpl_disconnect(cid_press)
            canvas.mpl_disconnect(cid_motion)
            canvas.mpl_disconnect(cid_release)
            
            roi_coords = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            
            try:
                fps = int(self.fps_entry.get())
                
                # Comprobamos si estamos en modo 'píxeles'
                # Si hay un vídeo cargado, significa que el fondo es una imagen y las coordenadas son píxeles.
                usar_pixeles = bool(self.ruta_video_cargado and os.path.exists(self.ruta_video_cargado))
                
                # Le pasamos la información a la función de análisis
                resultados_roi = analizar_region_de_interes(self.df_procesado, roi_coords, fps, in_pixels=usar_pixeles)
                
                component = self.plot_components.get('principal')
                if component:
                    component['roi_tiempo_label'].configure(text=f"{resultados_roi.get('tiempo_s', '---')} s")
                    component['roi_porcentaje_label'].configure(text=f"{resultados_roi.get('porcentaje', '---')}")
                
                if not self.is_processing: self.status_label.configure(text=f"Mostrando: {os.path.basename(self.ruta_csv_cargado)}")
            
            except (ValueError, KeyError):
                self.status_label.configure(text="Error: Parámetros de FPS no válidos.", text_color="orange")

        cid_press = canvas.mpl_connect('button_press_event', on_press)
        cid_motion = canvas.mpl_connect('motion_notify_event', on_motion)
        cid_release = canvas.mpl_connect('button_release_event', on_release)

    def mostrar_vista_analisis(self):
        """Oculta la vista de proyecto y muestra la de análisis de archivo único."""
        # 1. Ocultar el frame del proyecto
        self.project_frame.grid_forget()

        # 2. Mostrar los widgets de análisis único usando .pack()
        #    en el orden en que estaban en __init__
        self.params_title_label.pack(pady=(20, 5), padx=20, fill="x", anchor="w")
        self.params_frame.pack(pady=0, padx=15, fill="x")
        self.apply_button.pack(pady=15, padx=20, fill="x")
        self.render_video_checkbox.pack(pady=5, padx=20, anchor="w")

        # 3. Re-configurar el botón de vista
        self.view_button.configure(text="Gestión de Proyecto", command=self.mostrar_vista_proyecto)
        # Asegurarse de que el botón de vista esté en el lugar correcto (probablemente después de apply_button)
        # Re-empacarlo puede ser necesario si el orden se altera
        self.apply_button.pack_forget()
        self.view_button.pack_forget()
        self.render_video_checkbox.pack_forget()
        
        self.apply_button.pack(pady=15, padx=20, fill="x")
        self.view_button.pack(pady=5, padx=20, fill="x")
        self.render_video_checkbox.pack(pady=5, padx=20, anchor="w")


        # 4. Mostrar el frame principal de análisis
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    def actualizar_nombre_proyecto(self, event=None):
        """Actualiza el nombre del proyecto cuando se modifica el campo"""
        if self.project_data is not None:
            nuevo_nombre = self.nombre_proyecto_entry.get()
            if nuevo_nombre:
                self.project_data["nombre_proyecto"] = nuevo_nombre

    def mostrar_vista_proyecto(self):
        """Oculta la vista de análisis y muestra la de gestión de proyectos."""
        self.main_frame.grid_forget()
        self.params_title_label.pack_forget()
        self.params_frame.pack_forget()
        self.apply_button.pack_forget()
        self.render_video_checkbox.pack_forget()
        
        self.project_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.project_frame.grid_columnconfigure(0, weight=1)
        self.project_frame.grid_rowconfigure(3, weight=1)

        self.view_button.configure(text="<- Volver a Análisis", command=self.mostrar_vista_analisis)

        for widget in self.project_frame.winfo_children():
            widget.destroy()

        ctk.CTkLabel(self.project_frame, text="Gestor de Proyectos de Análisis", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        # Campo de nombre del proyecto
        nombre_frame = ctk.CTkFrame(self.project_frame)
        nombre_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        ctk.CTkLabel(nombre_frame, text="Nombre:", width=80).pack(side="left", padx=5)
        self.nombre_proyecto_entry = ctk.CTkEntry(nombre_frame)
        self.nombre_proyecto_entry.pack(side="left", padx=5, fill="x", expand=True)
        if self.project_data and "nombre_proyecto" in self.project_data:
            self.nombre_proyecto_entry.insert(0, self.project_data["nombre_proyecto"])
        self.nombre_proyecto_entry.bind("<FocusOut>", self.actualizar_nombre_proyecto)
        
        # Botones de acción
        project_actions_frame = ctk.CTkFrame(self.project_frame)
        project_actions_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkButton(project_actions_frame, text="Nuevo", command=self.nuevo_proyecto).pack(side="left", padx=3)
        ctk.CTkButton(project_actions_frame, text="Abrir", command=self.abrir_proyecto).pack(side="left", padx=3)
        ctk.CTkButton(project_actions_frame, text="Guardar", command=self.guardar_proyecto).pack(side="left", padx=3)
        ctk.CTkButton(project_actions_frame, text="Gestionar Grupos", command=self.abrir_ventana_gestion_grupos).pack(side="left", padx=3)
        ctk.CTkButton(project_actions_frame, text="Añadir Archivos", command=self.añadir_archivos_al_proyecto).pack(side="left", padx=3)
        ctk.CTkButton(project_actions_frame, text="Analizar Proyecto", command=self.solicitar_seleccion_analisis_y_mostrar).pack(side="left", padx=3)
        self.pdf_button = ctk.CTkButton(project_actions_frame, text="PDF", command=self.generar_reporte_pdf, state="disabled")
        self.pdf_button.pack(side="left", padx=3)

        list_header_frame = ctk.CTkFrame(self.project_frame)
        list_header_frame.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.render_all_var = ctk.BooleanVar(value=True) 
        render_all_checkbox = ctk.CTkCheckBox(list_header_frame, text="Renderizar Todos", variable=self.render_all_var,
                                              command=self._alternar_todos_renderizados)
        render_all_checkbox.pack(side="left", padx=10, pady=5)


        # Lista de videos
        self.project_video_list_frame = ctk.CTkScrollableFrame(self.project_frame, label_text="Archivos en el Proyecto")
        self.project_video_list_frame.grid(row=3, column=0, padx=20, pady=20, sticky="nsew")
        
        self.refrescar_lista_videos_proyecto()


    def nuevo_proyecto(self):
        """Crea un nuevo proyecto vacío"""
        self.project_data = {
            "nombre_proyecto": "Proyecto Sin Nombre",
            "grupos": {},
            "videos": {},
            "fecha_creacion": str(pd.Timestamp.now()),
            "version": "1.0"
        }
        self.refrescar_lista_videos_proyecto()
        self.status_label.configure(text="Nuevo proyecto creado. Añade archivos y gestiona grupos.")

    def guardar_proyecto(self):
        """Guarda el proyecto actual en un archivo JSON con rutas relativas"""
        if self.project_data is None:
            self.status_label.configure(text="No hay proyecto para guardar.", text_color="orange")
            return
            
        # Pedir al usuario donde guardar el proyecto
        ruta_archivo = filedialog.asksaveasfilename(
            title="Guardar proyecto",
            defaultextension=".json",
            filetypes=[("Archivos de proyecto", "*.json"), ("Todos los archivos", "*.*")]
        )
        
        if not ruta_archivo:
            return  # Usuario canceló
        
        try:
            # Crear una copia del proyecto para modificar las rutas
            proyecto_data = self.project_data.copy()
            proyecto_dir = os.path.dirname(ruta_archivo)
            
            # Convertir rutas absolutas a relativas
            for video_name, video_data in proyecto_data["videos"].items():
                for key in ["ruta_original", "ruta_csv"]:
                    if video_data.get(key) and os.path.isabs(video_data[key]):
                        try:
                            # Convertir a ruta relativa respecto al directorio del proyecto
                            video_data[key] = os.path.relpath(video_data[key], proyecto_dir)
                        except ValueError:
                            # Si están en diferentes unidades, mantener absoluta
                            # Esto pasa cuando los archivos están en otra unidad de disco (C: vs D:)
                            pass
            
            # Actualizar metadatos antes de guardar
            proyecto_data["fecha_modificacion"] = str(pd.Timestamp.now())
            proyecto_data["directorio_proyecto"] = proyecto_dir  # Guardar el directorio como referencia
            
            # Guardar en formato JSON
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                json.dump(proyecto_data, f, indent=4, ensure_ascii=False)
                
            self.status_label.configure(text=f"Proyecto guardado en: {os.path.basename(ruta_archivo)}")
            
        except Exception as e:
            self.status_label.configure(text=f"Error al guardar proyecto: {str(e)}", text_color="red")
            import traceback
            traceback.print_exc()

    def abrir_proyecto(self):
        """Abre un proyecto desde un archivo JSON y convierte rutas relativas a absolutas"""
        # Pedir al usuario que seleccione un archivo de proyecto
        ruta_archivo = filedialog.askopenfilename(
            title="Abrir proyecto",
            filetypes=[("Archivos de proyecto", "*.json"), ("Todos los archivos", "*.*")]
        )
        
        if not ruta_archivo:
            return  # Usuario canceló
        
        try:
            # Cargar el proyecto desde JSON
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                proyecto_data = json.load(f)

            # Si el proyecto es antiguo y no tiene la clave 'grupos', la añadimos.
            if "grupos" not in proyecto_data:
                proyecto_data["grupos"] = {}
            # Hacemos lo mismo para el campo 'grupo' en cada vídeo
            for video_info in proyecto_data.get("videos", {}).values():
                if "grupo" not in video_info:
                    video_info["grupo"] = None
                
            # Convertir rutas relativas a absolutas
            proyecto_dir = os.path.dirname(ruta_archivo)
            
            for video_name, video_data in proyecto_data["videos"].items():
                for key in ["ruta_original", "ruta_csv"]:
                    if video_data.get(key):
                        # Si es una ruta relativa, convertir a absoluta
                        if not os.path.isabs(video_data[key]):
                            video_data[key] = os.path.join(proyecto_dir, video_data[key])
                        # Verificar que el archivo existe
                        if not os.path.exists(video_data[key]):
                            self.status_label.configure(
                                text=f"Advertencia: Archivo no encontrado: {os.path.basename(video_data[key])}", 
                                text_color="orange"
                            )
                
            # Asignar los datos procesados al proyecto
            self.project_data = proyecto_data
            self.mostrar_vista_proyecto()
            self.status_label.configure(text=f"Proyecto cargado: {os.path.basename(ruta_archivo)}")
            
        except Exception as e:
            self.status_label.configure(text=f"Error al cargar proyecto: {str(e)}", text_color="red")
            import traceback
            traceback.print_exc()


    def refrescar_lista_videos_proyecto(self):
        if not hasattr(self, 'project_video_list_frame'):
            return

        for widget in self.project_video_list_frame.winfo_children():
            widget.destroy()

        if self.project_data is None or not self.project_data.get("videos"):
            ctk.CTkLabel(self.project_video_list_frame, text="No hay archivos en el proyecto.").pack(pady=20)
            return

        nombres_grupos = ["Sin Asignar"] + [g["nombre"] for g in self.project_data.get("grupos", {}).values()]
        mapa_nombre_a_id = {g["nombre"]: g_id for g_id, g in self.project_data.get("grupos", {}).items()}

        def _actualizar_grupo_video(video_name, nombre_grupo_seleccionado):
            if nombre_grupo_seleccionado == "Sin Asignar":
                self.project_data["videos"][video_name]["grupo"] = None
            else:
                self.project_data["videos"][video_name]["grupo"] = mapa_nombre_a_id.get(nombre_grupo_seleccionado)
        
        def _actualizar_opcion_renderizar(video_name, checkbox_var):
            self.project_data["videos"][video_name]["renderizar"] = bool(checkbox_var.get())

        for nombre_video, datos in self.project_data["videos"].items():
            row_frame = ctk.CTkFrame(self.project_video_list_frame)
            row_frame.pack(fill="x", padx=5, pady=5)
            
            render_var = ctk.BooleanVar(value=datos.get("renderizar", True))
            render_checkbox = ctk.CTkCheckBox(row_frame, text="", variable=render_var, width=24,
                                            command=lambda v_name=nombre_video, var=render_var: _actualizar_opcion_renderizar(v_name, var))
            render_checkbox.pack(side="left", padx=(10, 5))

            # Lógica para acortar nombres largos
            MAX_FILENAME_LENGTH = 40
            nombre_a_mostrar = nombre_video
            if len(nombre_a_mostrar) > MAX_FILENAME_LENGTH:
                # Acortamos el nombre y añadimos "..."
                nombre_a_mostrar = nombre_a_mostrar[:MAX_FILENAME_LENGTH - 3] + "..."
            
            # Usamos el nombre acortado en el Label
            ctk.CTkLabel(row_frame, text=nombre_a_mostrar, anchor="w").pack(side="left", padx=5, expand=True, fill="x")
            
            estado = datos.get("estado", "Desconocido")
            color = {"Pendiente": "orange", "Calibrado": "lightgreen", "Procesado": "green", "Procesando": "blue"}.get(estado, "white")
            estado_text = estado
            if datos.get("calibracion_px"): estado_text += " ✓"
            ctk.CTkLabel(row_frame, text=f"Estado: {estado_text}", text_color=color, width=120).pack(side="left", padx=10)
            
            grupo_actual_id = datos.get("grupo")
            nombre_grupo_actual = "Sin Asignar"
            for g_id, g_info in self.project_data.get("grupos", {}).items():
                if g_id == grupo_actual_id:
                    nombre_grupo_actual = g_info["nombre"]
                    break
            grupo_menu = ctk.CTkOptionMenu(row_frame, values=nombres_grupos, width=150,
                                        command=lambda nuevo_grupo, v_name=nombre_video: _actualizar_grupo_video(v_name, nuevo_grupo))
            grupo_menu.set(nombre_grupo_actual)
            grupo_menu.pack(side="left", padx=10)
            
            btn_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            btn_frame.pack(side="right", padx=5)

            #desactivamos el boton de calibrar en el caso de que se añada un csv y no se tenga la ruta del video para dicho csv
            estado_calibrar = "normal"
            if not datos.get("ruta_original") or not os.path.exists(datos["ruta_original"]):
                estado_calibrar = "disabled"
            
            calibrar_btn = ctk.CTkButton(btn_frame, text="Calibrar", width=80,
                                        command=lambda v=nombre_video: self.iniciar_calibracion(v), state=estado_calibrar)
            calibrar_btn.pack(side="left", padx=5)
            
            estado_individual = "normal" if estado in ["Pendiente", "Calibrado"] else "disabled"
            estado_final = "disabled" if self.is_processing else estado_individual
            
            procesar_btn = ctk.CTkButton(btn_frame, text="▶ Procesar", width=90, fg_color="#2E8B57", hover_color="#3CB371",
                                        state=estado_final,
                                        command=lambda v=nombre_video: self.procesar_video_proyecto(v))
            procesar_btn.pack(side="left", padx=5)

            eliminar_btn = ctk.CTkButton(btn_frame, text="🗑️", width=40, fg_color="firebrick",
                                        command=lambda v=nombre_video: self._eliminar_video_proyecto(v))
            eliminar_btn.pack(side="left", padx=5)

    def añadir_archivos_al_proyecto(self):
        if self.project_data is None:
            self.project_data = {"nombre_proyecto": "Proyecto Sin Nombre", "grupos": {}, "videos": {}, "fecha_creacion": str(pd.Timestamp.now()), "version": "1.1"}

        rutas_archivos = filedialog.askopenfilenames(title="Seleccionar ficheros", filetypes=(("Soportados", "*.mp4 *.csv"), ("Vídeos", "*.mp4"), ("Resultados", "*.csv")))
        if not rutas_archivos: return

        directorio_trabajo = self._get_directorio_trabajo()

        for ruta in rutas_archivos:
            nombre_archivo = os.path.basename(ruta)
            
            if nombre_archivo.endswith('.mp4'):
                video_key = nombre_archivo
                if video_key not in self.project_data["videos"]:
                    self.project_data["videos"][video_key] = {"grupo": None, "renderizar": True}

                nombre_base = video_key.split('.')[0]
                ruta_csv_posible = os.path.join(directorio_trabajo, 'results', 'stats', f"{nombre_base}_interpolated_denormalized.csv")
                
                self.project_data["videos"][video_key]["ruta_original"] = ruta
                self.project_data["videos"][video_key]["calibracion_px"] = None

                if os.path.exists(ruta_csv_posible):
                    self.project_data["videos"][video_key]["estado"] = "Procesado"
                    self.project_data["videos"][video_key]["ruta_csv"] = ruta_csv_posible
                else:
                    self.project_data["videos"][video_key]["estado"] = "Pendiente"
                    self.project_data["videos"][video_key]["ruta_csv"] = None

            elif nombre_archivo.endswith('.csv'):
                nombre_base = nombre_archivo.replace("_interpolated_denormalized.csv", "")
                video_key = nombre_base.split('.')[0] + ".mp4"
            
                ruta_video_encontrada = None
                ruta_video_renderizado = os.path.join(directorio_trabajo, 'results', 'videos', video_key)
                ruta_video_entrada = os.path.join(directorio_trabajo, 'videos', video_key)

                if os.path.exists(ruta_video_renderizado): ruta_video_encontrada = ruta_video_renderizado
                elif os.path.exists(ruta_video_entrada): ruta_video_encontrada = ruta_video_entrada
                
                #en el caso de que no se encuentre la ruta al video del csv que indica el usuario, entoonces se le pide que lo introduzca manualmente
                if ruta_video_encontrada is None:
                    self.status_label.configure(text=f"No se encontró el vídeo '{video_key}' automáticamente. Por favor, localízalo.", text_color="orange")
                    # Forzar que la ventana de diálogo aparezca por encima de la principal
                    self.attributes('-topmost', False) # Asegurarse que la app no está "siempre encima"
                    ruta_manual = filedialog.askopenfilename(
                        title=f"Localiza el vídeo original para '{nombre_archivo}'",
                        filetypes=(("Archivos MP4", "*.mp4"),),
                        parent=self # Asegurar que el diálogo es modal a la app
                        )
                    if ruta_manual: # Si el usuario selecciona un archivo
                        ruta_video_encontrada = ruta_manual
                        self.status_label.configure(text="Vídeo localizado manualmente.", text_color="green")
                    else: # Si el usuario pulsa "Cancelar"
                        self.status_label.configure(text=f"Advertencia: No se asoció vídeo para '{video_key}'. No se podrá calibrar.", text_color="orange")


                if video_key not in self.project_data["videos"]:
                    self.project_data["videos"][video_key] = {"grupo": None, "renderizar": True}
                
                self.project_data["videos"][video_key]["ruta_original"] = ruta_video_encontrada
                self.project_data["videos"][video_key]["estado"] = "Procesado"
                self.project_data["videos"][video_key]["ruta_csv"] = ruta
                
        self.project_data["fecha_modificacion"] = str(pd.Timestamp.now())
        self.refrescar_lista_videos_proyecto()
  
    def _eliminar_video_proyecto(self, video_name_a_eliminar):
        """
        Elimina una entrada de vídeo del diccionario del proyecto.
        """
        if self.project_data and "videos" in self.project_data:
            if video_name_a_eliminar in self.project_data["videos"]:
                # Elimina la clave del diccionario
                del self.project_data["videos"][video_name_a_eliminar]
                
                # Refresca la lista en la interfaz para mostrar el cambio
                self.refrescar_lista_videos_proyecto()
                self.status_label.configure(text=f"Se ha eliminado '{video_name_a_eliminar}' del proyecto.")

    def procesar_video_proyecto(self, video_name):
        if self.is_processing:
            self.status_label.configure(text="Ya hay un procesamiento en curso.", text_color="orange")
            return

        if self.project_data is None or video_name not in self.project_data["videos"]: return
        video_data = self.project_data["videos"][video_name]

        if video_data["estado"] not in ["Pendiente", "Calibrado"]:
            directorio_trabajo = self._get_directorio_trabajo()
            ruta_video_renderizado_posible = os.path.join(directorio_trabajo, 'results', 'videos', video_name)
            debe_renderizar = video_data.get("renderizar", True)
            if debe_renderizar and not os.path.exists(ruta_video_renderizado_posible):
                pass
            else:
                self.status_label.configure(text=f"El vídeo '{video_name}' ya está procesado.", text_color="orange")
                return
        
        if not video_data.get("ruta_original") or not os.path.exists(video_data["ruta_original"]):
            self.status_label.configure(text="Error: No se encuentra el archivo de vídeo original.", text_color="red")
            return
            
        self.is_processing = True
        video_data["estado"] = "Procesando"
        self.refrescar_lista_videos_proyecto()

        self.notification_frame.pack_forget()
        self.status_label.pack(fill="x")
        nombre_proyecto = self.project_data.get("nombre_proyecto", "")
        self.status_label.configure(text=f"Procesando '{video_name}' del proyecto '{nombre_proyecto}'...")
        self.progress_bar.pack(fill="x", padx=10, pady=2)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        
        threading.Thread(target=self._ejecutar_procesamiento_video_proyecto, args=(video_name, video_data["ruta_original"]), daemon=True).start()


    def _alternar_todos_renderizados(self):
        if self.project_data is None or not self.project_data.get("videos"):
            return
        nuevo_estado = self.render_all_var.get()
        for video_name in self.project_data["videos"]:
            self.project_data["videos"][video_name]["renderizar"] = nuevo_estado
        self.refrescar_lista_videos_proyecto()

    def _ejecutar_procesamiento_video_proyecto(self, video_name, ruta_video):
        directorio_trabajo = os.path.dirname(os.path.abspath(__file__))
        
        #  Definimos el nombre base UNA SOLA VEZ
        nombre_base = video_name.split('.')[0]

        try:
            ruta_conf = os.path.join(directorio_trabajo, 'config', 'run_conf.yaml')
            with open(ruta_conf, 'r') as f: config_data = yaml.safe_load(f)

            debe_renderizar = self.project_data["videos"][video_name].get("renderizar", True)
            
            if debe_renderizar:
                config_data['output']['render_videos'] = [nombre_base]
            else:
                config_data['output']['render_videos'] = []

            with open(ruta_conf, 'w') as f: yaml.dump(config_data, f)
        except Exception as e:
            self.output_queue.put(f"[ERROR_PROYECTO] No se pudo modificar el run_conf.yaml: {e}|{video_name}")
            return

        ruta_carpeta_videos = os.path.join(directorio_trabajo, 'videos')
        try:
            if not os.path.exists(ruta_carpeta_videos): os.makedirs(ruta_carpeta_videos)
            for archivo in os.listdir(ruta_carpeta_videos): os.remove(os.path.join(ruta_carpeta_videos, archivo))
            
            # Copiamos el vídeo RENOMBRÁNDOLO con el nombre base
            nuevo_nombre_video = f"{nombre_base}.mp4"
            shutil.copy(ruta_video, os.path.join(ruta_carpeta_videos, nuevo_nombre_video))

        except Exception as e:
            self.output_queue.put(f"[ERROR_PROYECTO] Error al preparar archivos: {e}|{video_name}")
            return
            
        ruta_docker_compose = os.path.join(directorio_trabajo, 'docker-hub-docker-compose.yaml')
        comando = ["docker", "compose", "-f", ruta_docker_compose, "up"]
        
        try:
            process = subprocess.Popen(comando, cwd=directorio_trabajo, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            _, stderr = process.communicate()
            
            if process.returncode == 0:
                ruta_csv = os.path.join(directorio_trabajo, 'results', 'stats', f"{nombre_base}_interpolated_denormalized.csv")
                
                if os.path.exists(ruta_csv):
                    self.output_queue.put(f"[DONE_PROYECTO] {ruta_csv}|{video_name}")
                else:
                    self.output_queue.put(f"[ERROR_PROYECTO] No se encontró el CSV (nombre esperado: {os.path.basename(ruta_csv)})|{video_name}")
            else:
                self.output_queue.put(f"[ERROR_PROYECTO] {stderr}|{video_name}")
        except Exception as e:
            self.output_queue.put(f"[ERROR_PROYECTO] {str(e)}|{video_name}")


    def notificar_procesamiento_proyecto_completado(self, ruta_csv, video_name):
        """Maneja la finalización exitosa del procesamiento de un video del proyecto"""
        self.is_processing = False

        if self.project_data and video_name in self.project_data["videos"]:
            video_data = self.project_data["videos"][video_name]
            video_data["estado"] = "Procesado"
            video_data["ruta_csv"] = ruta_csv
            
            if "dimensiones_caja_mm" not in video_data and "escala_px_mm" in video_data:
                # Si no tenemos dimensiones pero sí escala, estimar dimensiones basadas en el video original
                try:
                    if video_data.get("ruta_original") and os.path.exists(video_data["ruta_original"]):
                        cap = cv2.VideoCapture(video_data["ruta_original"])
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        # Calcular dimensiones aproximadas basadas en la escala
                        escala = video_data["escala_px_mm"]
                        ancho_real = width / escala
                        alto_real = height / escala
                        video_data["dimensiones_caja_mm"] = (ancho_real, alto_real)
                except:
                    pass
            
            # Actualizar fecha de modificación del proyecto
            self.project_data["fecha_modificacion"] = str(pd.Timestamp.now())
            
            self.refrescar_lista_videos_proyecto()
            self.status_label.configure(text=f"Procesamiento de {video_name} completado.")

        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self._actualizar_estado_botones_procesar() # Actualiza la UI
            
    def finalizar_procesamiento_proyecto_con_error(self, error_msg, video_name):
        """Maneja errores en el procesamiento de un video del proyecto"""
        self.is_processing = False

        if self.project_data and video_name in self.project_data["videos"]:
            video_data = self.project_data["videos"][video_name]
            video_data["estado"] = "Pendiente"
            
        self.status_label.configure(text=f"Error procesando {video_name}", text_color="red")
        print(f"--- ERROR DE PROCESAMIENTO DE PROYECTO ---\nVideo: {video_name}\nError: {error_msg}")
        
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self._actualizar_estado_botones_procesar() 


    def actualizar_distancia_punto(self):
        if self.df_procesado is None: return
        component_controls = self.plot_components.get('principal')
        if not component_controls: return
        try:
            punto_x = float(component_controls['punto_x_entry'].get())
            punto_y = float(component_controls['punto_y_entry'].get())
            plot_container = self.plot_components.get('distancia_punto_plot')
            if not plot_container: 
                plot_container = {'parent': self._crear_contenedor_grafico("Distancia a Punto vs. Tiempo", row=5, column=1)}
                self.plot_components['distancia_punto_plot'] = plot_container
            fig = generar_grafico_distancia_punto(self.df_procesado, punto_x, punto_y)
            self.embed_figure(fig, plot_container['parent'])
        except (ValueError, KeyError):
            self.status_label.configure(text="Error: Coordenadas del punto no válidas.", text_color="orange")



    def _abrir_ventana_analisis_conjunto(self, grupo_id_seleccionado=None, **kwargs_postproceso):
        """
        Abre la ventana de análisis conjunto.
        Si se proporciona un 'grupo_id_seleccionado', filtra los resultados para ese grupo.
        Si es None, analiza todos los vídeos.
        """
        self.last_project_analysis_kwargs = kwargs_postproceso
        if not self.project_data:
            self.status_label.configure(text="No hay proyecto cargado.", text_color="orange")
            return

        videos_a_procesar = {}
        for video_name, video_data in self.project_data["videos"].items():
            if grupo_id_seleccionado is None or video_data.get("grupo") == grupo_id_seleccionado:
                videos_a_procesar[video_name] = video_data
        
        videos_validos = [
            v_name for v_name, v_data in videos_a_procesar.items()
            if (v_data.get("estado") == "Procesado" and v_data.get("ruta_csv") and 
                os.path.exists(v_data["ruta_csv"]) and v_data.get("calibracion_px"))
        ]

        if not videos_validos:
            msg = "No hay vídeos procesados/calibrados en el proyecto."
            if grupo_id_seleccionado:
                nombre_grupo = self.project_data["grupos"][grupo_id_seleccionado]["nombre"]
                msg = f"No hay vídeos procesados/calibrados en el grupo '{nombre_grupo}'."
            self.status_label.configure(text=msg, text_color="orange")
            return

        self.analysis_window = ctk.CTkToplevel(self)
        titulo_ventana = f"Análisis Conjunto - {self.project_data['nombre_proyecto']}"
        if grupo_id_seleccionado:
            titulo_ventana += f" (Grupo: {self.project_data['grupos'][grupo_id_seleccionado]['nombre']})"
        self.analysis_window.title(titulo_ventana)
        self.analysis_window.geometry("1400x900")
        self.analysis_window.grab_set()

        tabview = ctk.CTkTabview(self.analysis_window)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        hay_videos_en_grupos = False
        for video_info in self.project_data.get("videos", {}).values():
            if video_info.get("grupo") is not None:
                hay_videos_en_grupos = True
                break # Encontramos uno, no hace falta seguir buscando

        #comprueba si hay vídeos asignados a grupos
        if grupo_id_seleccionado is None and hay_videos_en_grupos:
            # Solo si se analiza el proyecto completo Y hay vídeos en grupos, se muestran estas pestañas
            tabview.add("Comparativa por Grupos")
            tabview.add("Heatmaps por Grupo")
            self.mostrar_comparativa_grupos(tabview.tab("Comparativa por Grupos"), **kwargs_postproceso)
            self.mostrar_heatmaps_por_grupo(tabview.tab("Heatmaps por Grupo"), **kwargs_postproceso)

        #estas se muestrtan siempre:
        tabview.add("Resumen General")
        tabview.add("Mapa de Calor Conjunto")
        tabview.add("Análisis Individuales")
        
        self.mostrar_resumen_general(tabview.tab("Resumen General"), videos_a_procesar, **kwargs_postproceso)
        self.mostrar_mapa_calor_conjunto(tabview.tab("Mapa de Calor Conjunto"), videos_a_procesar, **kwargs_postproceso)
        self.mostrar_analisis_individuales(tabview.tab("Análisis Individuales"), videos_a_procesar, **kwargs_postproceso)

        if hasattr(self, 'pdf_button'):
             self.pdf_button.configure(state="normal") # Activa el botón PDF


    def mostrar_mapa_calor_conjunto(self, parent_frame, videos_a_procesar, **kwargs_postproceso):
        """
        Carga los datos aplicando post-procesado y muestra un mapa de calor conjunto.
        """
        lista_dfs_procesados = []
        max_ancho_local = 0
        max_alto_local = 0
        default_fps = 20 # O leerlo desde self.project_data

        for video_name, video_data in videos_a_procesar.items():
            if (video_data.get("estado") == "Procesado" and
                video_data.get("ruta_csv") and os.path.exists(video_data["ruta_csv"]) and
                video_data.get("calibracion_px") and video_data.get("dimensiones_caja_mm")):

                try:
                    df = cargar_y_preparar_datos(
                        ruta_archivo_csv=video_data["ruta_csv"],
                        fps=default_fps,
                        puntos_calibracion_px=video_data["calibracion_px"],
                        dimensiones_reales_mm=video_data["dimensiones_caja_mm"],
                        **kwargs_postproceso # Pasar parámetros
                    )
                    if df is not None and not df.empty:
                        lista_dfs_procesados.append(df)
                        # Actualizar dimensiones máximas locales para este conjunto
                        ancho_actual, alto_actual = video_data["dimensiones_caja_mm"]
                        if ancho_actual > max_ancho_local: max_ancho_local = ancho_actual
                        if alto_actual > max_alto_local: max_alto_local = alto_actual
                except Exception as e:
                    print(f"Error al procesar {video_name} para mapa de calor conjunto: {e}")
                    continue

        if not lista_dfs_procesados or max_ancho_local == 0 or max_alto_local == 0:
            ctk.CTkLabel(parent_frame, text="No hay vídeos con datos válidos para generar el mapa de calor.").pack(pady=20)
            return

        # Generar el mapa con las dimensiones máximas del conjunto actual
        figura_heatmap = generar_mapa_calor_conjunto(lista_dfs_procesados, (max_ancho_local, max_alto_local))

        # Embed en la interfaz (como antes)
        canvas = FigureCanvasTkAgg(figura_heatmap, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        plt.close(figura_heatmap)



    def mostrar_resumen_general(self, parent_frame, videos_a_procesar, **kwargs_postproceso):
        """
        Muestra un resumen general, aplicando los parámetros de post-procesado.
        """
        datos_videos_validos = [] # Lista para almacenar diccionarios con resultados por video
        default_fps = int(self.project_data.get("default_fps", 20)) # O leerlo desde self.project_data

        for video_name, video_data in videos_a_procesar.items():
            # Comprobar si el video tiene todo lo necesario
            if (video_data.get("estado") == "Procesado" and
                video_data.get("ruta_csv") and os.path.exists(video_data.get("ruta_csv","")) and
                video_data.get("calibracion_px") and video_data.get("dimensiones_caja_mm")):

                dims_mm = video_data["dimensiones_caja_mm"] # Guardar dimensiones
                try:
                    df = cargar_y_preparar_datos(
                        ruta_archivo_csv=video_data["ruta_csv"],
                        fps=default_fps,
                        puntos_calibracion_px=video_data["calibracion_px"],
                        dimensiones_reales_mm=video_data["dimensiones_caja_mm"],
                        **kwargs_postproceso # Pasar parámetros
                    )

                    if df is not None and not df.empty:
                        # Calcular métricas necesarias para el resumen
                        df = calcular_distancia_al_borde(df, dims_mm)
                        stats = calcular_estadisticas_completas(df, dimensiones_caja_mm_para_exploracion=dims_mm) # Devuelve floats/None
                        zonas = analizar_zonas_caja(df, dims_mm, default_fps)

                        # Guardar solo las métricas necesarias para los gráficos de resumen
                        datos_videos_validos.append({
                            'nombre': video_name,
                            'zonas': zonas, # Necesario para gráfico de preferencia espacial
                            'distancia_total': stats.get("Distancia Total (m)"),
                            'velocidad_media': stats.get("Velocidad Media (mm/s)"),
                            'sinuosidad': stats.get("Índice de Sinuosidad")
                        })
                except Exception as e:
                     print(f"Error procesando {video_name} para resumen general: {e}")
                     pass # Continuar

        if not datos_videos_validos:
            ctk.CTkLabel(parent_frame, text="No hay vídeos con datos válidos para mostrar el resumen.").pack(pady=20)
            return

        # Generación de Gráficos 2x2 (como antes, pero usando los datos recolectados)
        nombres_zonas = list(next(iter(datos_videos_validos))['zonas'].keys()) if datos_videos_validos else []
        # Calcular promedio por zona SOLO con vídeos válidos
        promedio_por_zona = {
            zona: sum(v['zonas'][zona]['porcentaje_total'] for v in datos_videos_validos if zona in v.get('zonas', {})) / len(datos_videos_validos)
            for zona in nombres_zonas
        }

        nombres_cortos_zonas = ['ESI', 'CS', 'ESD', 'CI', 'C', 'CD', 'EII', 'CIN', 'EID']
        porcentajes_promedio = [promedio_por_zona.get(zona, 0) for zona in nombres_zonas] # Usar get con default 0


        nombres_videos = [d['nombre'] for d in datos_videos_validos]
        # Extraer métricas, manejando posibles Nones si alguna falló
        distancias_totales = [d.get('distancia_total') for d in datos_videos_validos]
        velocidades_medias = [d.get('velocidad_media') for d in datos_videos_validos]
        sinuosidad = [d.get('sinuosidad') for d in datos_videos_validos]

        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3) # Aumentar hspace
        fig.suptitle('Resumen General del Comportamiento', fontsize=16, weight='bold')

        # Gráfico 1: Preferencia Espacial Promedio
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(nombres_cortos_zonas, porcentajes_promedio, color='teal')
        ax1.set_title('Preferencia Espacial Promedio (9 Zonas)')
        ax1.set_ylabel('Tiempo Promedio en Zona (%)')
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.set_ylim(0, max(porcentajes_promedio) * 1.1 if porcentajes_promedio else 10) # Ajustar límite y

        # Gráficos Dinámicos (Barra vs Histograma)
        UMBRAL_VIDEOS = 15 # Límite para cambiar de barra a histograma

        def limpiar_datos_para_plot(datos):
             return [d for d in datos if d is not None and not np.isnan(d)]

        # Gráfico 2: Distancia Total
        ax2 = fig.add_subplot(gs[0, 1])
        distancias_limpias = limpiar_datos_para_plot(distancias_totales)
        if distancias_limpias:
            if len(datos_videos_validos) < UMBRAL_VIDEOS:
                ax2.bar(nombres_videos, [d if d is not None else 0 for d in distancias_totales], color='purple') # Poner 0 si es None para bar
                ax2.set_title('Distancia Total Recorrida por Vídeo')
                ax2.set_ylabel('Distancia (m)')
                ax2.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                ax2.hist(distancias_limpias, bins=10, color='purple', edgecolor='black')
                ax2.set_title('Distribución de Distancias Totales')
                ax2.set_xlabel('Distancia (m)')
                ax2.set_ylabel('Frecuencia (Nº de Vídeos)')
        else: ax2.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax2.transAxes)


        # Gráfico 3: Velocidad Media
        ax3 = fig.add_subplot(gs[1, 0])
        velocidades_limpias = limpiar_datos_para_plot(velocidades_medias)
        if velocidades_limpias:
            if len(datos_videos_validos) < UMBRAL_VIDEOS:
                ax3.bar(nombres_videos, [d if d is not None else 0 for d in velocidades_medias], color='skyblue')
                ax3.set_title('Velocidad Media por Vídeo')
                ax3.set_ylabel('Velocidad (mm/s)')
                ax3.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                ax3.hist(velocidades_limpias, bins=10, color='skyblue', edgecolor='black')
                ax3.set_title('Distribución de Velocidades Medias')
                ax3.set_xlabel('Velocidad (mm/s)')
                ax3.set_ylabel('Frecuencia (Nº de Vídeos)')
        else: ax3.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax3.transAxes)

        # Gráfico 4: Índice de Sinuosidad
        ax4 = fig.add_subplot(gs[1, 1])
        sinuosidad_limpia = limpiar_datos_para_plot(sinuosidad)
        if sinuosidad_limpia:
            if len(datos_videos_validos) < UMBRAL_VIDEOS:
                ax4.bar(nombres_videos, [d if d is not None else 0 for d in sinuosidad], color='salmon')
                ax4.set_title('Índice de Sinuosidad por Vídeo')
                ax4.set_ylabel('Índice')
                ax4.axhline(y=1, color='gray', linestyle='--', linewidth=1)
                ax4.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                ax4.hist(sinuosidad_limpia, bins=10, color='salmon', edgecolor='black')
                ax4.set_title('Distribución del Índice de Sinuosidad')
                ax4.set_xlabel('Índice de Sinuosidad')
                ax4.set_ylabel('Frecuencia (Nº de Vídeos)')
        else: ax4.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax4.transAxes)

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para título

        # Embed en la interfaz (como antes)
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def mostrar_analisis_individuales(self, parent_frame, videos_a_procesar, **kwargs_postproceso):
        """
        Muestra el análisis individual de cada video, aplicando post-procesado.
        """
        scrollable = ctk.CTkScrollableFrame(parent_frame)
        scrollable.pack(fill="both", expand=True)
        default_fps = int(self.project_data.get("default_fps", 20)) # O leerlo
        videos_mostrados = 0

        for video_name, video_data in videos_a_procesar.items():
            if (video_data.get("estado") == "Procesado" and
                video_data.get("ruta_csv") and os.path.exists(video_data["ruta_csv"]) and
                video_data.get("calibracion_px") and video_data.get("dimensiones_caja_mm")):

                dimensiones = video_data["dimensiones_caja_mm"] # Guardar dimensiones
                try:
                    df = cargar_y_preparar_datos(
                        ruta_archivo_csv=video_data["ruta_csv"],
                        fps=default_fps,
                        puntos_calibracion_px=video_data["calibracion_px"],
                        dimensiones_reales_mm=dimensiones,
                        **kwargs_postproceso #Pasar parámetros
                    )
                    if df is None or df.empty:
                        print(f"Info: El procesado de {video_name} resultó vacío para análisis individuales.")
                        continue # Saltar a siguiente video si el df está vacío

                    # Calcular distancia al borde para su pestaña
                    df = calcular_distancia_al_borde(df, dimensiones)

                    # Crear el contenedor y pestañas para este vídeo
                    videos_mostrados += 1
                    video_frame = ctk.CTkFrame(scrollable)
                    video_frame.pack(fill="x", padx=10, pady=10, expand=True) # expand=True
                    ctk.CTkLabel(video_frame, text=video_name, font=ctk.CTkFont(weight="bold")).pack(pady=5, anchor="w", padx=10)

                    video_tabview = ctk.CTkTabview(video_frame, height=350) # Darle una altura mínima
                    video_tabview.pack(fill="x", padx=10, pady=5, expand=True)

                    # Añadir pestañas
                    tab_pref = video_tabview.add("Preferencia")
                    tab_zonas = video_tabview.add("9 Zonas")
                    tab_bordes = video_tabview.add("Análisis Bordes")
                    tab_stats = video_tabview.add("Estadísticas")
                    tab_dist_centro = video_tabview.add("Distancia Centro")

                    dimensiones = video_data["dimensiones_caja_mm"]

                    # Rellenar Pestaña de Preferencia (Centro vs Bordes)
                    resultados_pref = analizar_preferencia_espacial(df, dimensiones, default_fps)
                    info_text_pref = (
                        f"Tiempo en Centro: {resultados_pref['centro']['porcentaje']:.1f}% ({resultados_pref['centro']['tiempo_segundos']:.1f}s)\n"
                        f"Tiempo en Bordes: {resultados_pref['bordes']['porcentaje']:.1f}% ({resultados_pref['bordes']['tiempo_segundos']:.1f}s)"
                    )
                    ctk.CTkLabel(tab_pref, text=info_text_pref, justify="left").pack(pady=10, padx=10, anchor="w")

                    # Rellenar Pestaña de 9 Zonas
                    resultados_zonas = analizar_zonas_caja(df, dimensiones, default_fps)
                    zonas_frame_inner = ctk.CTkScrollableFrame(tab_zonas, label_text="Tiempo por Zona") # Frame con scroll por si no caben
                    zonas_frame_inner.pack(fill="both", expand=True, padx=5, pady=5)
                    for zona, datos in resultados_zonas.items():
                        zone_text = f"{zona.replace('_', ' ').title()}: {datos['porcentaje_total']:.1f}% ({datos['tiempo_segundos']:.1f}s)"
                        ctk.CTkLabel(zonas_frame_inner, text=zone_text).pack(anchor="w", padx=10, pady=2)

                    # Rellenar Pestaña de Análisis de Bordes
                    fig_bordes = generar_grafico_distancia_al_borde(df) # Usa la columna ya calculada
                    canvas_bordes = FigureCanvasTkAgg(fig_bordes, master=tab_bordes)
                    canvas_bordes.draw()
                    canvas_bordes.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                    plt.close(fig_bordes)

                    # Rellenar Pestaña Distancia Centro
                    fig_dist_centro = generar_grafico_distancia_centro(df, dimensiones) # Llamar a la nueva función
                    canvas_dist_centro = FigureCanvasTkAgg(fig_dist_centro, master=tab_dist_centro)
                    canvas_dist_centro.draw()
                    canvas_dist_centro.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                    plt.close(fig_dist_centro) # Cerrar figura para liberar memoria

                    # Rellenar Pestaña de Estadísticas
                    stats = calcular_estadisticas_completas(df,dimensiones_caja_mm_para_exploracion=dimensiones) # Devuelve floats/None
                    stats_frame_inner = ctk.CTkScrollableFrame(tab_stats, label_text="Métricas Calculadas") # Frame con scroll
                    stats_frame_inner.pack(fill="both", expand=True, padx=5, pady=5)
                    
                    #iteramos sobre las métricas
                    for key, value in stats.items():
                        # Formateo mejorado para distinguir N/A de 0.0
                        if value is not None:
                             if key in ["Área Explorada (%)", "Tiempo Activo (%)"]:
                                 valor_formateado = f"{value:.1f}%"
                             elif key in ["Distancia Total (m)", "Índice de Sinuosidad"]:
                                  valor_formateado = f"{value:.2f}"
                             else: # mm/s, mm/s², mm
                                  valor_formateado = f"{value:.1f}"
                        else:
                             valor_formateado = "N/A" # Mostrar N/A si es None

                        ctk.CTkLabel(stats_frame_inner, text=f"{key}: {valor_formateado}").pack(anchor="w", padx=10, pady=2)

                except Exception as e:
                    print(f"Error procesando {video_name} para análisis individual: {e}")
                    error_frame = ctk.CTkFrame(scrollable)
                    error_frame.pack(fill="x", padx=10, pady=10)
                    ctk.CTkLabel(error_frame, text=f"Error al analizar {video_name}: {e}", text_color="orange", wraplength=400).pack()

        if videos_mostrados == 0:
             ctk.CTkLabel(scrollable, text="No se encontraron vídeos válidos con datos completos en esta selección.").pack(pady=20)

    #no se usa:
    def mostrar_comparativa(self, parent_frame):
        """Muestra análisis comparativo entre videos"""
        datos_comparativa = []
        for video_name, video_data in self.project_data["videos"].items():
            if (video_data.get("estado") == "Procesado" and
                video_data.get("ruta_csv") and os.path.exists(video_data["ruta_csv"]) and
                video_data.get("calibracion_px") and video_data.get("dimensiones_caja_mm")):
                
                try:
                    df = cargar_y_preparar_datos(
                        ruta_archivo_csv=video_data["ruta_csv"],
                        fps=20,
                        puntos_calibracion_px=video_data["calibracion_px"],
                        dimensiones_reales_mm=video_data["dimensiones_caja_mm"]
                    )
                    
                    if df is not None:
                        dimensiones = video_data["dimensiones_caja_mm"]
                        preferencia = analizar_preferencia_espacial(df, dimensiones, 25)
                        
                        datos_comparativa.append({
                            'nombre': video_name,
                            'centro': preferencia['centro']['porcentaje'],
                            'bordes': preferencia['bordes']['porcentaje']
                        })
                except Exception as e:
                    print(f"Error procesando {video_name} para comparativa: {e}")
                    continue

        if not datos_comparativa:
            ctk.CTkLabel(parent_frame, text="No hay datos para comparar").pack(pady=20)
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Gráfico de barras comparativo
        nombres = [d['nombre'] for d in datos_comparativa]
        centro = [d['centro'] for d in datos_comparativa]
        bordes = [d['bordes'] for d in datos_comparativa]

        x = range(len(nombres))
        ax1.bar(x, centro, width=0.4, label='Centro', alpha=0.7)
        ax1.bar(x, bordes, width=0.4, label='Bordes', alpha=0.7, bottom=centro)
        ax1.set_xticks(x)
        ax1.set_xticklabels(nombres, rotation=45, ha='right')
        ax1.set_ylabel('Porcentaje (%)')
        ax1.legend()
        ax1.set_title('Comparativa Centro vs Bordes')

        # Gráfico de dispersión
        ax2.scatter(centro, bordes)
        for i, nombre in enumerate(nombres):
            ax2.annotate(nombre, (centro[i], bordes[i]), fontsize=8, alpha=0.7)
        ax2.set_xlabel('Tiempo en Centro (%)')
        ax2.set_ylabel('Tiempo en Bordes (%)')
        ax2.set_title('Relación Centro vs Bordes')
        ax2.plot([0, 100], [0, 100], 'r--', alpha=0.3)  # Línea de referencia

        plt.tight_layout()

        # Embed en la interfaz
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def abrir_ventana_gestion_grupos(self):
        if self.project_data is None:
            self.status_label.configure(text="Crea o abre un proyecto primero.", text_color="orange")
            return

        # Crear la ventana emergente (Toplevel)
        self.grupos_window = ctk.CTkToplevel(self)
        self.grupos_window.title("Gestor de Grupos Experimentales")
        self.grupos_window.geometry("500x400")
        self.grupos_window.grab_set()  # Hace que esta ventana sea modal

        # Frame para añadir nuevos grupos
        add_frame = ctk.CTkFrame(self.grupos_window)
        add_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(add_frame, text="Nombre del nuevo grupo:").pack(side="left", padx=5)
        nombre_grupo_entry = ctk.CTkEntry(add_frame)
        nombre_grupo_entry.pack(side="left", padx=5, expand=True, fill="x")
        
        # Frame para la lista de grupos existentes
        self.lista_grupos_frame = ctk.CTkScrollableFrame(self.grupos_window, label_text="Grupos existentes")
        self.lista_grupos_frame.pack(fill="both", expand=True, padx=10, pady=10)

        def refrescar_lista_grupos_ui():
            # Limpiar la lista actual
            for widget in self.lista_grupos_frame.winfo_children():
                widget.destroy()
            
            # Volver a dibujar la lista desde self.project_data
            for grupo_id, grupo_info in self.project_data.get("grupos", {}).items():
                row_frame = ctk.CTkFrame(self.lista_grupos_frame)
                row_frame.pack(fill="x", pady=2)
                ctk.CTkLabel(row_frame, text=grupo_info["nombre"], anchor="w").pack(side="left", padx=10, expand=True, fill="x")
                
                # El comando lambda necesita capturar el valor actual de grupo_id
                eliminar_btn = ctk.CTkButton(row_frame, text="Eliminar", width=80, fg_color="firebrick",
                                            command=lambda g_id=grupo_id: eliminar_grupo(g_id))
                eliminar_btn.pack(side="right", padx=10)
        
        def anadir_grupo():
            nombre_nuevo = nombre_grupo_entry.get()
            if nombre_nuevo and nombre_nuevo not in [g["nombre"] for g in self.project_data["grupos"].values()]:
                # Crear un ID único para el grupo (ej: grupo_1725791163)
                nuevo_id = f"grupo_{int(pd.Timestamp.now().timestamp())}"
                self.project_data["grupos"][nuevo_id] = {"nombre": nombre_nuevo}
                nombre_grupo_entry.delete(0, "end")
                refrescar_lista_grupos_ui()
            else:
                # Mostrar un pequeño error si el nombre está vacío o ya existe
                nombre_grupo_entry.configure(border_color="red")
        
        def eliminar_grupo(grupo_id_a_eliminar):
            # Eliminar el grupo del diccionario de grupos
            if grupo_id_a_eliminar in self.project_data["grupos"]:
                del self.project_data["grupos"][grupo_id_a_eliminar]
            
            # Desasignar este grupo de cualquier vídeo que lo tuviera
            for video_info in self.project_data["videos"].values():
                if video_info.get("grupo") == grupo_id_a_eliminar:
                    video_info["grupo"] = None
            
            refrescar_lista_grupos_ui()

        # Conectar el botón de añadir con su función
        add_button = ctk.CTkButton(add_frame, text="Añadir", width=80, command=anadir_grupo)
        add_button.pack(side="left", padx=5)

        # Botón para cerrar la ventana
        close_button = ctk.CTkButton(self.grupos_window, text="Cerrar", command=self.cerrar_ventana_grupos)
        close_button.pack(pady=10)

        # Cargar la lista inicial
        refrescar_lista_grupos_ui()

    # Añade también esta función para cerrar la ventana y refrescar la lista de vídeos
    def cerrar_ventana_grupos(self):
        if hasattr(self, 'grupos_window'):
            self.grupos_window.destroy()
            # Refrescamos la lista de vídeos por si los grupos han cambiado
            self.refrescar_lista_videos_proyecto()


    def solicitar_seleccion_analisis_y_mostrar(self):
        """
        Muestra una ventana para que el usuario elija qué analizar
        Y TAMBIÉN para configurar los parámetros de post-procesado para este análisis.
        """
        if not self.project_data:
             self.status_label.configure(text="No hay proyecto cargado.", text_color="orange")
             return

        # Crear la ventana de selección
        seleccion_window = ctk.CTkToplevel(self)
        seleccion_window.title("Configurar Análisis de Proyecto")
        seleccion_window.geometry("450x550") # Hacerla más alta
        seleccion_window.grab_set()

        # Sección 1: Qué Analizar
        ctk.CTkLabel(seleccion_window, text="¿Qué conjunto de vídeos analizar?", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))

        # Frame para los botones de selección de grupo/todos
        grupo_frame = ctk.CTkFrame(seleccion_window, fg_color="transparent")
        grupo_frame.pack(fill="x", padx=20, pady=5)

        # Sección 2: Parámetros de Post-Procesado
        ctk.CTkLabel(seleccion_window, text="Parámetros de Post-Procesado", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))

        # Frame para los controles de post-procesado
        postpro_frame = ctk.CTkFrame(seleccion_window)
        postpro_frame.pack(fill="x", padx=20, pady=5)
        postpro_frame.grid_columnconfigure(1, weight=1)

        # Añadir controles (copiar estructura del sidebar, pero usando variables locales)
        ctk.CTkLabel(postpro_frame, text="Interpolación:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        interp_menu_var = ctk.StringVar(value='linear')
        interp_menu_popup = ctk.CTkOptionMenu(postpro_frame, variable=interp_menu_var, values=['linear', 'polynomial', 'none'], width=120)
        interp_menu_popup.grid(row=0, column=1, sticky="e", padx=10, pady=5)

        ctk.CTkLabel(postpro_frame, text="Max Gap (frames):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        interp_gap_entry_popup = ctk.CTkEntry(postpro_frame, width=80); interp_gap_entry_popup.insert(0, "5")
        interp_gap_entry_popup.grid(row=1, column=1, sticky="e", padx=10, pady=5)

        ctk.CTkLabel(postpro_frame, text="Eliminar Atípicos (%):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        outlier_entry_popup = ctk.CTkEntry(postpro_frame, width=80); outlier_entry_popup.insert(0, "100")
        outlier_entry_popup.grid(row=2, column=1, sticky="e", padx=10, pady=5)

        smoothing_label_popup = ctk.CTkLabel(postpro_frame, text="Suavizado (ventana): 1")
        smoothing_label_popup.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(10,0))
        smoothing_slider_popup = ctk.CTkSlider(postpro_frame, from_=1, to=25, number_of_steps=24, command=lambda v: smoothing_label_popup.configure(text=f"Suavizado (ventana): {int(v)}"))
        smoothing_slider_popup.set(1)
        smoothing_slider_popup.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        # Función que se llamará al pulsar un botón:
        def _on_select(grupo_id_seleccionado):
            try:
                # Recoger parámetros de post-procesado del pop-up
                kwargs_postproceso = {
                    "interpolation_method": interp_menu_var.get(),
                    "max_gap_frames": int(interp_gap_entry_popup.get()),
                    "outlier_percentile": float(outlier_entry_popup.get() or 100.0),
                    "smoothing_window_size": int(smoothing_slider_popup.get()),
                    "interpolation_order": 2 # Añadir por si se usa polynomial
                }
                # Validaciones básicas 
                if kwargs_postproceso["max_gap_frames"] < 0: raise ValueError("Max Gap >= 0")
                if not (0 <= kwargs_postproceso["outlier_percentile"] <= 100): raise ValueError("Atípicos 0-100")
                if kwargs_postproceso["smoothing_window_size"] < 1: raise ValueError("Suavizado >= 1")

            except ValueError as e:
                 # Mostrar error dentro del pop-up o en status_label principal
                 print(f"Error en parámetros del pop-up: {e}") # Mejorar esto con un label en el pop-up
                 return

            seleccion_window.destroy()
            # Pasar los parámetros recogidos a la función principal
            self._abrir_ventana_analisis_conjunto(
                grupo_id_seleccionado=grupo_id_seleccionado,
                **kwargs_postproceso # Pasar los parámetros
            )

        # Crear los botones de selección (dentro de grupo_frame)
        # Botón para analizar todos los vídeos
        ctk.CTkButton(grupo_frame, text="Todos los Vídeos (Visión General)", command=lambda: _on_select(None)).pack(pady=5, padx=10, fill="x")

        # Botones para cada grupo (si existen)
        if self.project_data.get("grupos"):
             ctk.CTkLabel(grupo_frame, text="O analizar un grupo específico:").pack(pady=(10, 5))
             for grupo_id, grupo_info in self.project_data["grupos"].items():
                 nombre_grupo = grupo_info["nombre"]
                 # Usar lambda para capturar el grupo_id correcto
                 ctk.CTkButton(grupo_frame, text=f"Solo Grupo: '{nombre_grupo}'",
                              command=lambda g_id=grupo_id: _on_select(g_id)).pack(pady=3, padx=10, fill="x")
            

    def mostrar_comparativa_grupos(self, parent_frame, **kwargs_postproceso):
        if not self.project_data or not self.project_data.get("grupos"):
            ctk.CTkLabel(parent_frame, text="No hay grupos definidos...").pack(pady=20)
            return

        datos_por_grupo = {}
        metricas_calculadas = [
            "Distancia Total (m)", "Velocidad Media (mm/s)",
            "% Tiempo en Centro", "Distancia Media al Borde (mm)",
            "Área Explorada (%)" 
        ]
        metricas_internas = {
            "Distancia Total (m)": "distancias",
            "Velocidad Media (mm/s)": "velocidades_medias",
            "% Tiempo en Centro": "porcentajes_centro",
            "Distancia Media al Borde (mm)": "distancias_al_borde",
            "Área Explorada (%)": "areas_exploradas" 
        }
        # Asegurarse que datos_por_grupo se inicializa con la nueva clave
        for grupo_id, grupo_info in self.project_data.get("grupos", {}).items():
             datos_por_grupo[grupo_info["nombre"]] = {v: [] for v in metricas_internas.values()}

        default_fps = int(self.project_data.get("default_fps", 20)) # Usar un FPS guardado o default

        # Bucle para cargar y procesar datos de cada video
        for video_name, video_data in self.project_data["videos"].items():
            grupo_id = video_data.get("grupo")
            nombre_grupo = self.project_data.get("grupos", {}).get(grupo_id, {}).get("nombre")

            if nombre_grupo and video_data.get("estado") == "Procesado":
                ruta_csv = video_data.get("ruta_csv")
                calib_px = video_data.get("calibracion_px")
                dims_mm = video_data.get("dimensiones_caja_mm") # NECESARIO para exploración

                # Comprobar que tenemos todo lo necesario
                if not ruta_csv or not os.path.exists(ruta_csv) or not calib_px or not dims_mm:
                    #print(f"Advertencia: Faltan datos para {video_name}...") # Opcional
                    continue

                try:
                    df = cargar_y_preparar_datos(
                        ruta_archivo_csv=ruta_csv,
                        fps=default_fps,
                        puntos_calibracion_px=calib_px,
                        dimensiones_reales_mm=dims_mm,
                        **kwargs_postproceso
                    )

                    if df is not None and not df.empty:
                        df = calcular_distancia_al_borde(df, dims_mm)

                        # LLAMAR PASANDO LAS DIMENSIONES
                        stats = calcular_estadisticas_completas(
                            df,
                            dimensiones_caja_mm_para_exploracion=dims_mm # Pasar dimensiones
                        )
                        # Calcular preferencia (si se usa)
                        preferencia = analizar_preferencia_espacial(df, dims_mm, default_fps)

                        # Añadir datos (% Tiempo Centro)
                        if "centro" in preferencia:
                             valor_centro = preferencia["centro"].get("porcentaje")
                             if valor_centro is not None:
                                  datos_por_grupo[nombre_grupo][metricas_internas["% Tiempo en Centro"]].append(valor_centro)

                        # Añadir resto de métricas (incluye Exploración)
                        for metrica_larga, metrica_corta in metricas_internas.items():
                             if metrica_larga != "% Tiempo en Centro":
                                 valor = stats.get(metrica_larga) # Obtiene float o None
                                 if valor is not None:
                                     datos_por_grupo[nombre_grupo][metrica_corta].append(valor)

                except Exception as e:
                    print(f"Error procesando {video_name} para comparativa: {e}")
                    pass # Continuar

        # Filtrar grupos que no tengan datos válidos para ninguna métrica
        datos_graficos = {
            nombre: datos for nombre, datos in datos_por_grupo.items()
            if any(any(v is not None for v in lista) for lista in datos.values()) # Comprueba si hay al menos un valor no nulo en alguna métrica
        }

        if not datos_graficos or len(datos_graficos) < 1:
            ctk.CTkLabel(parent_frame, text="No hay suficientes datos...").pack(pady=20)
            return

        # Generación de Gráficos (Box Plots)
        scrollable_frame = ctk.CTkScrollableFrame(parent_frame, fg_color="transparent")
        scrollable_frame.pack(fill="both", expand=True)
        nombres_grupos = list(datos_graficos.keys())
        num_metricas = len(metricas_internas) # Ahora 5
        cols = 3 # Cambiar a 3 columnas para que quepa mejor
        rows = (num_metricas + cols - 1) // cols # Ahora serán 2 filas

        fig = plt.figure(figsize=(5 * cols, 5 * rows)) # Ajustar tamaño
        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.5, wspace=0.35) # Ajustar espacio
        fig.suptitle('Análisis Comparativo por Grupos', fontsize=16, weight='bold')

        # Función auxiliar para filtrar Nones antes de boxplot
        def filtrar_nones_para_boxplot(lista_de_listas):
             return [[valor for valor in sublista if valor is not None and not np.isnan(valor)] for sublista in lista_de_listas]

        axes = {}
        metricas_info = { # Para títulos y etiquetas
             "distancias": ("Distribución de Distancia Recorrida", "Distancia Total (m)", 'skyblue'),
             "velocidades_medias": ("Distribución de Velocidad Media", "Velocidad Media (mm/s)", 'salmon'),
             "porcentajes_centro": ("Distribución de % Tiempo en Centro", "Porcentaje (%)", 'lightgreen'),
             "distancias_al_borde": ("Distribución de Distancia Media al Borde", "Distancia (mm)", 'gold'),
             "areas_exploradas": ("Área Explorada", "Área Explorada (%)", 'lightcoral')
        }

        idx = 0
        # Iterar sobre las claves definidas en metricas_info
        for metrica_corta, (titulo, ylabel, color) in metricas_info.items():
            ax_row, ax_col = divmod(idx, cols)
            ax = fig.add_subplot(gs[ax_row, ax_col])
            axes[metrica_corta] = ax

            # Recolectar datos y filtrar Nones
            datos_metrica = [datos_graficos[nombre].get(metrica_corta, []) for nombre in nombres_grupos]
            datos_filtrados = filtrar_nones_para_boxplot(datos_metrica)

            # Dibujar boxplot si hay datos
            if any(len(d) > 0 for d in datos_filtrados): # Comprobar si alguna lista tiene datos
                bp = ax.boxplot(datos_filtrados, tick_labels=nombres_grupos, patch_artist=True, boxprops=dict(facecolor=color), showfliers=True)
                ax.set_title(titulo)
                ax.set_ylabel(ylabel)
                ax.grid(True, linestyle='--', alpha=0.6)
                # Limitar eje Y para porcentajes
                if metrica_corta in ["porcentajes_centro", "areas_exploradas"]:
                    ax.set_ylim(0, 100)
                ax.tick_params(axis='x', rotation=30, labelsize=9) # Rotar etiquetas
            else:
                 ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=ax.transAxes)
                 ax.set_title(titulo + " (Sin datos)")
            idx += 1

        # Ocultar ejes extras si num_metricas no es múltiplo de cols
        while idx < rows * cols:
            ax_row, ax_col = divmod(idx, cols)
            try:
                # Intentar obtener el eje y ocultarlo
                extra_ax = fig.add_subplot(gs[ax_row, ax_col])
                extra_ax.set_visible(False)
            except Exception: # Puede fallar si ya se añadió antes, ignorar
                pass
            idx += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar márgenes

        # Embed y cerrar figura (como antes)
        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas.draw(); canvas.get_tk_widget().pack(fill="x", padx=10, pady=10)
        plt.close(fig)

        # Análisis Estadístico (incluir nueva métrica)
        texto_estadistico = ""
        resultados_test = []
        nombres_grupos_con_datos = nombres_grupos

        if len(nombres_grupos_con_datos) >= 2: # Solo si hay al menos dos grupos para comparar
            for metrica_corta, (titulo, _, _) in metricas_info.items():
                # La función comparar_grupos_estadisticamente debe manejar los Nones internamente
                nombre_test, p_valor = comparar_grupos_estadisticamente(datos_graficos, metrica_corta)

                if p_valor is not None:
                    conclusion = "la diferencia es estadísticamente significativa" if p_valor < 0.05 else "la diferencia observada podría deberse al azar"
                    if len(nombres_grupos_con_datos) > 2:
                        resultados_test.append(f" • {titulo}: p={p_valor:.3f} ({conclusion} entre los grupos)")
                    else:
                        resultados_test.append(f" • {titulo}: p={p_valor:.3f} ({conclusion})")
                else:
                     resultados_test.append(f" • {titulo}: No se pudo realizar el test (datos insuficientes).")


            if resultados_test:
                 # Intentar obtener el nombre del test usado (asumiendo que es el mismo para todas las métricas si hay >2 grupos)
                 nombre_test_usado = "N/A"
                 if len(nombres_grupos_con_datos) >= 2:
                     test_info = comparar_grupos_estadisticamente(datos_graficos, next(iter(metricas_internas.values())))
                     nombre_test_usado = test_info[0]

                 texto_estadistico = f"Resultados del Test de {nombre_test_usado} (p < 0.05 es significativo):\n" + "\n".join(resultados_test)

        if texto_estadistico:
             stats_label = ctk.CTkLabel(scrollable_frame, text=texto_estadistico, justify="left", font=ctk.CTkFont(family="Consolas", size=11))
             stats_label.pack(pady=10, padx=20, fill="x", anchor="w")
        elif len(nombres_grupos_con_datos) < 2:
              ctk.CTkLabel(scrollable_frame, text="Se necesita al menos dos grupos con datos para realizar el análisis estadístico.", justify="left").pack(pady=10, padx=20, fill="x", anchor="w")



    def mostrar_heatmaps_por_grupo(self, parent_frame, **kwargs_postproceso):
        """
        Genera y muestra un mapa de calor conjunto para cada grupo experimental.
        """
        if not self.project_data or not self.project_data.get("grupos"):
            ctk.CTkLabel(parent_frame, text="No hay grupos definidos para generar heatmaps.").pack(pady=20)
            return

        scrollable_frame = ctk.CTkScrollableFrame(parent_frame, fg_color="transparent")
        scrollable_frame.pack(fill="both", expand=True)

        default_fps = 20 # O leerlo desde self.project_data
        grupos_con_datos = False
        max_ancho_global = 0 # Para asegurar escala comparable si es posible
        max_alto_global = 0

        # Primero, encontrar las dimensiones máximas globales
        for video_data in self.project_data["videos"].values():
             dims_mm = video_data.get("dimensiones_caja_mm")
             if dims_mm:
                 if dims_mm[0] > max_ancho_global: max_ancho_global = dims_mm[0]
                 if dims_mm[1] > max_alto_global: max_alto_global = dims_mm[1]

        if max_ancho_global == 0 or max_alto_global == 0:
             ctk.CTkLabel(scrollable_frame, text="No se encontraron dimensiones válidas en ningún vídeo calibrado.").pack(pady=20)
             return


        for grupo_id, grupo_info in self.project_data.get("grupos", {}).items():
            lista_dfs_grupo = []
            videos_en_grupo = [] # Para saber qué videos contribuyen

            for video_name, video_data in self.project_data["videos"].items():
                if video_data.get("grupo") == grupo_id and video_data.get("estado") == "Procesado":
                    ruta_csv = video_data.get("ruta_csv")
                    calib_px = video_data.get("calibracion_px")
                    dims_mm = video_data.get("dimensiones_caja_mm")

                    if not ruta_csv or not os.path.exists(ruta_csv) or not calib_px or not dims_mm:
                        continue # Omitir si faltan datos esenciales

                    try:
                        df = cargar_y_preparar_datos(
                            ruta_archivo_csv=ruta_csv,
                            fps=default_fps,
                            puntos_calibracion_px=calib_px,
                            dimensiones_reales_mm=dims_mm,
                            **kwargs_postproceso # <-- Pasar parámetros
                        )
                        if df is not None and not df.empty:
                            lista_dfs_grupo.append(df)
                            videos_en_grupo.append(video_name)
                    except Exception as e:
                        print(f"Omitiendo video {video_name} para heatmap de grupo '{grupo_info['nombre']}': {e}")

            if lista_dfs_grupo:
                grupos_con_datos = True
                container = ctk.CTkFrame(scrollable_frame)
                # Usar pack(side="left") para que queden uno al lado del otro si caben
                container.pack(side="left", fill="both", expand=True, padx=10, pady=10)

                ctk.CTkLabel(container, text=f"Grupo: {grupo_info['nombre']} (N={len(videos_en_grupo)})", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 10))

                # Usar dimensiones globales para escala comparable
                figura_heatmap = generar_mapa_calor_conjunto(lista_dfs_grupo, (max_ancho_global, max_alto_global))

                canvas = FigureCanvasTkAgg(figura_heatmap, master=container)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                plt.close(figura_heatmap)
            #else:
            #    print(f"No hay datos válidos para el heatmap del grupo: {grupo_info['nombre']}")


        if not grupos_con_datos:
            ctk.CTkLabel(scrollable_frame, text="No hay grupos con datos procesados válidos para generar mapas de calor.").pack(pady=20)

    def generar_reporte_pdf(self):
        """Genera un reporte PDF profesional y completo con todos los resultados del análisis."""
        if not self.last_project_analysis_kwargs:
            self.status_label.configure(text="Error: Debes 'Analizar Proyecto' al menos una vez antes de generar el PDF.", text_color="orange")
            return
            
        # Rescatamos los parámetros del último análisis
        kwargs_pdf = self.last_project_analysis_kwargs
        if not self.project_data:
            self.status_label.configure(text="No hay proyecto cargado para generar reporte.", text_color="orange")
            return

        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")],
            title="Guardar reporte PDF como"
        )
        if not ruta_archivo:
            return

        archivos_temporales = []
        try:
            # 1. Configuración del Documento PDF
            doc = SimpleDocTemplate(ruta_archivo, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 2. Título e Información del Proyecto 
            title_style = ParagraphStyle('CustomTitle', parent=styles['h1'], fontSize=18, spaceAfter=20, alignment=1)
            story.append(Paragraph("Reporte de Análisis de Comportamiento", title_style))
            story.append(Paragraph(f"<b>Proyecto:</b> {self.project_data['nombre_proyecto']}", styles['h2']))
            
            info_text = f"<b>Fecha de generación:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            info_text += f"<b>Número total de vídeos:</b> {len(self.project_data['videos'])}<br/>"
            nombres_grupos = [g['nombre'] for g in self.project_data.get('grupos', {}).values()]
            info_text += f"<b>Grupos experimentales:</b> {', '.join(nombres_grupos) or 'N/A'}"
            story.append(Paragraph(info_text, styles['BodyText']))
            story.append(Spacer(1, 0.3 * inch))

            # 3.Análisis Comparativo (Box Plots)
            story.append(Paragraph("Análisis Comparativo por Grupos", styles['h2']))
            fig_comparativa = self.generar_figura_comparativa_grupos(**kwargs_pdf)
            if fig_comparativa:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig_comparativa.savefig(tmpfile.name, format='png', dpi=300, bbox_inches='tight')
                    ruta_temp_img = tmpfile.name
                    archivos_temporales.append(ruta_temp_img)
                img = Image(ruta_temp_img, width=7.5*inch, height=3.75*inch)
                story.append(img)
                plt.close(fig_comparativa)
            else:
                story.append(Paragraph("No hay suficientes datos para generar gráficos comparativos.", styles['BodyText']))
            story.append(Spacer(1, 0.3 * inch))

            # 4. Mapas de Calor por Grupo
            story.append(Paragraph("Mapas de Calor por Grupo", styles['h2']))
            grupos_con_datos = False
            for grupo_id, grupo_info in self.project_data.get("grupos", {}).items():
                lista_dfs_grupo = []
                max_ancho, max_alto = 0, 0
                for video_data in self.project_data["videos"].values():
                    if video_data.get("grupo") == grupo_id and video_data.get("estado") == "Procesado":
                        # Usamos un try-except dentro del bucle para que un vídeo no pare todo el reporte
                        try:
                            df = cargar_y_preparar_datos(
                                ruta_archivo_csv=video_data["ruta_csv"], fps=20,
                                puntos_calibracion_px=video_data["calibracion_px"],
                                dimensiones_reales_mm=video_data["dimensiones_caja_mm"],
                                **kwargs_pdf
                            )
                            if df is not None:
                                lista_dfs_grupo.append(df)
                                ancho, alto = video_data["dimensiones_caja_mm"]
                                if ancho > max_ancho: max_ancho = ancho
                                if alto > max_alto: max_alto = alto
                        except Exception as e:
                            print(f"Error procesando vídeo {video_data.get('nombre', '')} para PDF: {e}")
                
                if lista_dfs_grupo:
                    grupos_con_datos = True
                    story.append(Paragraph(f"<b>Grupo: {grupo_info['nombre']}</b>", styles['h3']))
                    fig_heatmap = generar_mapa_calor_conjunto(lista_dfs_grupo, (max_ancho, max_alto))
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        fig_heatmap.savefig(tmpfile.name, format='png', dpi=300, bbox_inches='tight')
                        ruta_temp_img = tmpfile.name
                        archivos_temporales.append(ruta_temp_img)
                    
                    img = Image(ruta_temp_img, width=5*inch, height=4*inch)
                    story.append(img)
                    plt.close(fig_heatmap)
                    story.append(Spacer(1, 0.2 * inch))

            if not grupos_con_datos:
                story.append(Paragraph("No hay grupos con datos procesados para generar mapas de calor.", styles['BodyText']))
            
            story.append(Spacer(1, 0.3 * inch))

            # 5. Tabla Detallada de Métricas por Vídeo
            story.append(Paragraph("Métricas Detalladas por Vídeo", styles['h2']))
            
            datos_tabla = [['Vídeo', 'Grupo', 'Dist. (m)', 'Vel. Med (mm/s)', 'Acel. Med (mm/s²)', '% Activo', 'Sinuosidad']]
            for video_name, video_data in self.project_data["videos"].items():
                if video_data.get("estado") == "Procesado":
                    try:
                        df = cargar_y_preparar_datos(
                            ruta_archivo_csv=video_data["ruta_csv"], fps=20,
                            puntos_calibracion_px=video_data["calibracion_px"],
                            dimensiones_reales_mm=video_data["dimensiones_caja_mm"],
                            **kwargs_pdf
                        )
                        if df is not None:
                            grupo_id = video_data.get("grupo")
                            nombre_grupo = self.project_data.get("grupos", {}).get(grupo_id, {}).get("nombre", "N/A")
                            stats = calcular_estadisticas_completas(df)
                            
                            datos_tabla.append([
                                Paragraph(video_name, styles['Normal']), # Para que el texto se ajuste si es largo
                                nombre_grupo,
                                f"{stats.get('Distancia Total (m)', 0):.2f}",
                                f"{stats.get('Velocidad Media (mm/s)', 0):.2f}",
                                f"{stats.get('Aceleración Media (mm/s²)', 0):.2f}",
                                f"{stats.get('Tiempo Activo (%)', 0):.1f}%",
                                f"{stats.get('Índice de Sinuosidad', 0):.2f}"
                            ])
                    except Exception:
                        continue
            
            col_widths = [2*inch, 1*inch, 0.8*inch, 1*inch, 1*inch, 0.8*inch, 0.9*inch]
            tabla = Table(datos_tabla, colWidths=col_widths)
            tabla.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D9E1F2')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(tabla)

            # 6. Construir y Guardar el PDF
            doc.build(story)
            self.status_label.configure(text=f"Reporte PDF generado: {os.path.basename(ruta_archivo)}")

        except Exception as e:
            self.status_label.configure(text=f"Error al generar PDF: {str(e)}", text_color="red")
            print(f"Error generando PDF: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 7. Limpieza de Archivos Temporales
            for path in archivos_temporales:
                try:
                    os.remove(path)
                except OSError:
                    print(f"No se pudo eliminar el archivo temporal: {path}")


    def generar_figura_comparativa_grupos(self, **kwargs_postproceso):
        """
        Función auxiliar que genera la figura de Matplotlib para la comparativa de grupos.
        Devuelve el objeto 'figure' o None si no hay datos.
        """
        # misma lógica de recopilación de datos que en 'mostrar_comparativa_grupos'
        datos_por_grupo = {}
        for grupo_id, grupo_info in self.project_data.get("grupos", {}).items():
            datos_por_grupo[grupo_info["nombre"]] = {"distancias": [], "velocidades_medias": [], "porcentajes_centro": []}

        for video_name, video_data in self.project_data["videos"].items():
            grupo_id = video_data.get("grupo")
            nombre_grupo = self.project_data.get("grupos", {}).get(grupo_id, {}).get("nombre")
            if nombre_grupo and video_data.get("estado") == "Procesado":
                try:
                    df = cargar_y_preparar_datos(
                        ruta_archivo_csv=video_data["ruta_csv"], fps=20,
                        puntos_calibracion_px=video_data["calibracion_px"],
                        dimensiones_reales_mm=video_data["dimensiones_caja_mm"],
                        **kwargs_postproceso
                    )
                    if df is not None:
                        datos_por_grupo[nombre_grupo]["distancias"].append(df['distancia_recorrida_mm'].sum() / 1000)
                        datos_por_grupo[nombre_grupo]["velocidades_medias"].append(df['velocidad_mms'].mean())
                        preferencia = analizar_preferencia_espacial(df, video_data["dimensiones_caja_mm"], 25)
                        datos_por_grupo[nombre_grupo]["porcentajes_centro"].append(preferencia["centro"]["porcentaje"])
                except Exception:
                    continue
        
        datos_graficos = {nombre: datos for nombre, datos in datos_por_grupo.items() if datos["distancias"]}
        if not datos_graficos:
            return None

        nombres_grupos = list(datos_graficos.keys())
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig)
        fig.suptitle('Análisis Comparativo por Grupos Experimentales', fontsize=16, weight='bold')

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.boxplot([datos["distancias"] for datos in datos_graficos.values()], tick_labels=nombres_grupos, patch_artist=True, boxprops=dict(facecolor='skyblue'))
        ax1.set_title('Distribución de Distancia Recorrida')
        ax1.set_ylabel('Distancia Total (m)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.boxplot([datos["velocidades_medias"] for datos in datos_graficos.values()], tick_labels=nombres_grupos, patch_artist=True, boxprops=dict(facecolor='salmon'))
        ax2.set_title('Distribución de Velocidad Media')
        ax2.set_ylabel('Velocidad Media (mm/s)')
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.boxplot([datos["porcentajes_centro"] for datos in datos_graficos.values()], tick_labels=nombres_grupos, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax3.set_title('Distribución de % Tiempo en Centro')
        ax3.set_ylabel('Porcentaje (%)')
        ax3.set_ylim(0, 100)
        ax3.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    #no se usa:
    def mostrar_estadisticas_conjuntas(self, parent_frame):
        """Muestra estadísticas conjuntas de todos los videos"""
        todos_datos = []
        for video_name, video_data in self.project_data["videos"].items():
            if (video_data.get("estado") == "Procesado" and
                video_data.get("ruta_csv") and os.path.exists(video_data["ruta_csv"]) and
                video_data.get("calibracion_px") and video_data.get("dimensiones_caja_mm")):
                
                try:
                    df = cargar_y_preparar_datos(
                        ruta_archivo_csv=video_data["ruta_csv"],
                        fps=20,
                        puntos_calibracion_px=video_data["calibracion_px"],
                        dimensiones_reales_mm=video_data["dimensiones_caja_mm"]
                    )
                    
                    if df is not None:
                        dimensiones = video_data["dimensiones_caja_mm"]
                        preferencia = analizar_preferencia_espacial(df, dimensiones, 25)
                        stats = calcular_estadisticas_generales(df)
                        
                        try:
                            distancia = float(stats.get("Distancia Total (m)", "0"))
                            velocidad_media = float(stats.get("Velocidad Media (mm/s)", "0"))
                            velocidad_max = float(stats.get("Velocidad Máxima (mm/s)", "0"))
                        except (ValueError, AttributeError):
                            distancia = velocidad_media = velocidad_max = 0
                        
                        todos_datos.append({
                            'nombre': video_name,
                            'centro': preferencia['centro']['porcentaje'],
                            'bordes': preferencia['bordes']['porcentaje'],
                            'distancia': distancia,
                            'velocidad_media': velocidad_media,
                            'velocidad_max': velocidad_max
                        })
                except Exception as e:
                    print(f"Error procesando {video_name} para estadísticas conjuntas: {e}")
                    continue

        if not todos_datos:
            ctk.CTkLabel(parent_frame, text="No hay datos para mostrar").pack(pady=20)
            return

        stats_frame = ctk.CTkScrollableFrame(parent_frame)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Encabezados de la tabla
        headers = ["Video", "Centro (%)", "Bordes (%)", "Distancia (m)", "Velocidad Media (mm/s)", "Velocidad Máx (mm/s)"]
        for i, header in enumerate(headers):
            ctk.CTkLabel(stats_frame, text=header, font=ctk.CTkFont(weight="bold")).grid(row=0, column=i, padx=5, pady=5)

        # Datos de la tabla
        for row, datos in enumerate(todos_datos, 1):
            ctk.CTkLabel(stats_frame, text=datos['nombre']).grid(row=row, column=0, padx=5, pady=2)
            ctk.CTkLabel(stats_frame, text=f"{datos['centro']:.2f}").grid(row=row, column=1, padx=5, pady=2)
            ctk.CTkLabel(stats_frame, text=f"{datos['bordes']:.2f}").grid(row=row, column=2, padx=5, pady=2)
            ctk.CTkLabel(stats_frame, text=f"{datos['distancia']:.2f}").grid(row=row, column=3, padx=5, pady=2)
            ctk.CTkLabel(stats_frame, text=f"{datos['velocidad_media']:.2f}").grid(row=row, column=4, padx=5, pady=2)
            ctk.CTkLabel(stats_frame, text=f"{datos['velocidad_max']:.2f}").grid(row=row, column=5, padx=5, pady=2)

        # Estadísticas resumen
        resumen_frame = ctk.CTkFrame(parent_frame)
        resumen_frame.pack(fill="x", padx=10, pady=10)
        
        # Calcular promedios
        promedios = {
            'centro': sum(d['centro'] for d in todos_datos) / len(todos_datos),
            'bordes': sum(d['bordes'] for d in todos_datos) / len(todos_datos),
            'distancia': sum(d['distancia'] for d in todos_datos) / len(todos_datos),
            'velocidad_media': sum(d['velocidad_media'] for d in todos_datos) / len(todos_datos),
            'velocidad_max': sum(d['velocidad_max'] for d in todos_datos) / len(todos_datos)
        }
        
        ctk.CTkLabel(resumen_frame, text="Promedios:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(resumen_frame, text=f"Centro: {promedios['centro']:.2f}%").pack(side="left", padx=10)
        ctk.CTkLabel(resumen_frame, text=f"Bordes: {promedios['bordes']:.2f}%").pack(side="left", padx=10)
        ctk.CTkLabel(resumen_frame, text=f"Distancia: {promedios['distancia']:.2f}m").pack(side="left", padx=10)
if __name__ == "__main__":
    app = App()
    app.mainloop()



#mirar lo de la trayectoria,que no haga lineas rectas raras(o preguntar pq las hace mas bien)

# en el caso de que haya mas de dos grupos que me aparezcan bien los graficos en heatmap por grupo(es lo de menos pq en la memoria no se va a ver)



