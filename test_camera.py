import cv2

# Prueba diferentes índices
for i in range(3):
    print(f"Probando cámara con índice {i}")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"¡Éxito! La cámara con índice {i} funciona")
            cv2.imshow(f'Cámara {i}', frame)
            cv2.waitKey(0)
        else:
            print(f"La cámara {i} está abierta pero no puede leer fotogramas")
    else:
        print(f"No se pudo abrir la cámara {i}")
    cap.release()

cv2.destroyAllWindows()
print("Prueba finalizada")