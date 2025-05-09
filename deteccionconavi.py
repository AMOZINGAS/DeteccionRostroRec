import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# Cargar modelo de detección de caras
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Crear directorio para grabaciones
os.makedirs("grabaciones", exist_ok=True)

# Configurar pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Control de grabación manual
grabando = False
writer = None
nombre_archivo = None
tiempo_inicio = None

# Archivo de registro
log = open("deteccion_caras_tiempos.txt", "w")
tiempo_base = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        (h, w) = color_image.shape[:2]

        # Detección de rostro
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob, "data")
        detections = net.forward("detection_out")

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
                print(f"➡️ Iniciando grabación: {nombre_archivo}")
                # Iniciar grabador de video
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(nombre_archivo, fourcc, 30.0, (w, h))
                grabando = True
            if writer:
                writer.write(color_image)
        else:
            if grabando:
                tiempo_fin = current_time
                print(f"⏹️ Deteniendo grabación: {nombre_archivo}")
                writer.release()
                log.write(f"Cara detectada desde {tiempo_inicio:.2f}s hasta {tiempo_fin:.2f}s -> {nombre_archivo}\n")
                log.flush()
                grabando = False
                writer = None

        # Omitimos cv2.imshow ya que no quieres mostrar video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if grabando and writer:
        writer.release()
    pipeline.stop()
    log.close()
    cv2.destroyAllWindows()