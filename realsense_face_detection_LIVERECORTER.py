import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# Cargar modelo de detección de caras
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Crear carpeta de grabaciones
os.makedirs("grabaciones", exist_ok=True)

# Inicializar pipeline para color
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

try:
    pipeline.start(config)
except Exception as e:
    print(f"❌ Error al iniciar la cámara: {e}")
    exit(1)

# Control de grabación
grabando = False
video_writer = None
nombre_archivo = None
tiempo_inicio = None

# Archivo de log
log = open("deteccion_caras_tiempos.txt", "w", encoding="utf-8")
tiempo_base = time.time()

print("🟢 Sistema activo. Presiona 'q' para salir.")

try:
    while True:
        # Esperar frame del pipeline
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            print("⚠️ Frame no recibido. Reintentando...")
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convertir a imagen de OpenCV
        color_image = np.asanyarray(color_frame.get_data())
        (h, w) = color_image.shape[:2]

        # Detectar rostros
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        current_time = time.time() - tiempo_base
        cara_detectada = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                cara_detectada = True
                break

        if cara_detectada:
            if not grabando:
                tiempo_inicio = current_time
                timestamp = int(time.time())
                nombre_archivo = f"grabaciones/cara_{timestamp}.avi"
                print(f"🎥 Iniciando grabación: {nombre_archivo}")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(nombre_archivo, fourcc, 30.0, (1920, 1080))
                grabando = True

            if video_writer:
                video_writer.write(color_image)

        else:
            if grabando:
                tiempo_fin = current_time
                print(f"🛑 Deteniendo grabación: {nombre_archivo}")
                video_writer.release()
                log.write(f"Cara detectada desde {tiempo_inicio:.2f}s hasta {tiempo_fin:.2f}s -> {nombre_archivo}\n")
                log.flush()
                grabando = False
                video_writer = None

        # (Opcional) Mostrar ventana
        cv2.imshow("Video", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🚪 Saliendo...")
            break

finally:
    if grabando and video_writer:
        video_writer.release()
    pipeline.stop()
    log.close()
    cv2.destroyAllWindows()
