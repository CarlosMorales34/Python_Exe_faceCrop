import cv2
import os
import urllib.request
import easygui

# Solicitar al usuario la ruta para guardar las fotos
ruta_guardado = easygui.diropenbox("Selecciona el directorio para guardar las fotos")

# Validar y crear la carpeta de fotos en la ruta proporcionada
if not ruta_guardado:
    print("No se seleccionó un directorio. Saliendo...")
    exit()

carpeta_fotos = os.path.join(ruta_guardado, 'Fotos_recorteAU')
if not os.path.exists(carpeta_fotos):
    os.makedirs(carpeta_fotos)

# Rutas de los clasificadores haarcascade (puedes cambiar estas URL según sea necesario)
face_cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'

# Descargar los clasificadores desde las URL
face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye.xml'
urllib.request.urlretrieve(face_cascade_url, face_cascade_path)
urllib.request.urlretrieve(eye_cascade_url, eye_cascade_path)

# Cargar los clasificadores haarcascade
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Índice de la cámara USB
usb_camera_index = 1

# Intentar abrir la cámara
cap = cv2.VideoCapture(usb_camera_index)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print(f"No se pudo abrir la cámara USB en el índice {usb_camera_index}. Verifica la conexión.")
    exit()

print(f"Se conectó a la cámara {usb_camera_index}")

# Función para realizar el recorte del rostro
def recortar_rostro(frame, x, y, w, h):
    # Parámetros para modular el recorte del rostro
    porcentaje_arriba = 0.5
    porcentaje_abajo = 0.9
    porcentaje_izquierda = 0.4
    porcentaje_derecha = 0.4

    # Calcular las nuevas coordenadas y dimensiones para el recorte
    y_recorte = int(y - porcentaje_arriba * h)
    h_recorte = int(h + (porcentaje_arriba + porcentaje_abajo) * h)
    x_recorte = int(x - porcentaje_izquierda * w)
    w_recorte = int(w + (porcentaje_izquierda + porcentaje_derecha) * w)

    # Asegurarse de que las coordenadas de recorte no sean negativas
    y_recorte = max(y_recorte, 0)
    h_recorte = max(h_recorte, 0)
    x_recorte = max(x_recorte, 0)
    w_recorte = max(w_recorte, 0)

    # Realizar el recorte del rostro
    rostro_recortado = frame[y_recorte:y_recorte + h_recorte, x_recorte:x_recorte + w_recorte]

    return rostro_recortado

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo capturar el frame. Saliendo...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        rostro_recortado = recortar_rostro(frame, x, y, w, h)

        if rostro_recortado.shape[0] > 0 and rostro_recortado.shape[1] > 0:
            cv2.imshow('Rostro Recortado', rostro_recortado)

    cv2.imshow('Reconocimiento Facial y de Ojos', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Utilizar easygui para obtener el nombre del archivo
        nombre_archivo = easygui.enterbox("Ingresa el nombre para la foto y presiona OK para guardar (o Cancel para salir): ")

        if nombre_archivo is None:
            break

        rostro_recortado = recortar_rostro(frame, x, y, w, h)

        if rostro_recortado.shape[0] > 0 and rostro_recortado.shape[1] > 0:
            ruta_guardado = os.path.join(carpeta_fotos, f'{nombre_archivo}.jpg')
            cv2.imwrite(ruta_guardado, rostro_recortado)

            print(f"Foto tomada y guardada como {ruta_guardado}. Presiona 'q' para salir.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
