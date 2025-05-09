Deteccion de rostro basado en el proyecto (**https://github.com/AMOZINGAS/DeteccionRostroRec.git**)

Se modifico dicho poryecto para usar la camara y grabar los frames donde aparece un rostro

Este programa guarda archivos .vin de los frames donde se encontraron rostros, ademas agrega a un txt los segundos donde aparecieron rostros

El programa usa una calidad de  1920x1080 por lo que es necesario que cuentes con un puerto USB 3

El programa utiliza python 3.11 por lo que es necesario usar esta version ya que tambien el pyrealsense2 solo es soportado hasta python3.11

Reomenadiones
  
  Para instalar o correr el porgrama (si cuentas con las dos versiones de python, la mas actualizada y la 3.11) para 
  
  iniciar el poryecto es necesario escribir el siguiente comando
    
    py -3.11 -m pip install pyrealsense2  
   
    py -3.11 -m realsense_face_detection_LIVERECORTER.py 
