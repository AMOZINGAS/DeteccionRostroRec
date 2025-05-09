import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Configuración
ARCHIVO_BAG = "video_prueba.bag"  # Cambia esto a tu archivo real
OUTPUT_DIR = "clips_con_rostros"
FPS = 30
UMBRAL_CARA = 0.5  # Umbral de confianza para detectar cara

# Crear carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar modelo de detección de rostro
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Configurar pipeline para reproducir .bag
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, ARCHIVO_BAG, repeat_playback=True)  # Asegúrate de que repita el archivo
pipeline.start(config)

# Alineador de profundidad al color
align = rs.align(rs.stream.color)

# Inicialización de variables
clip_num = 0
clip_activo = False
frames_buffer = []
writer = None
cara_presente = False

try:
    while True:
        try:
            frames = pipeline.wait_for_frames()  # Esto se queda esperando los frames
        except RuntimeError:
            print("No se recibieron frames en el tiempo esperado. Reintentando...")
            continue  # Intenta continuar el bucle si no llega ningún frame
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            break

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        (h, w) = color_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        cara_detectada = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > UMBRAL_CARA:
                cara_detectada = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Recorte seguro de ROI
                cx = int((startX + endX) / 2)
                cy = int((startY + endY) / 2)
                distance = depth_frame.get_distance(cx, cy)
                print(f"👤 Cara detectada a {distance:.2f} metros")

                break  # Solo necesitas detectar una cara

        # Guardar frame en buffer
        frames_buffer.append(color_image)

        if cara_detectada:
            if not clip_activo:
                clip_num += 1
                nombre_clip = f"{OUTPUT_DIR}/clip_{clip_num:03d}.mp4"
                height, width = color_image.shape[:2]
                writer = cv2.VideoWriter(nombre_clip, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
                print(f"🎬 Iniciando clip: {nombre_clip}")
                # Agrega frames previos por contexto
                for f in frames_buffer[-10:]:
                    writer.write(f)
            clip_activo = True
            writer.write(color_image)
        else:
            if clip_activo:
                print(f"⏹️ Finalizando clip {clip_num:03d}")
                writer.release()
                writer = None
                clip_activo = False
            frames_buffer = frames_buffer[-10:]  # Limita el buffer para evitar overflow

finally:
    if writer:
        writer.release()
    pipeline.stop()
    print("✅ Proceso completado.")
