# Autor: Bernardo Gularte Kirsch

import cv2
import numpy as np
import face_recognition

vid = cv2.VideoCapture(0)
classifFaces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

nomes_pessoas = ["Bernardo"]

imagens_referencia = []

for nome in nomes_pessoas:
    imagem = face_recognition.load_image_file(f"{nome}.png")
    face_encoding = face_recognition.face_encodings(imagem)[0]
    imagens_referencia.append(face_encoding)

limiar_reconhecimento = 0.8

while True:
    ok, frame = vid.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetected = classifFaces.detectMultiScale(frameGray, minSize=(50, 50))

    for (x, y, w, h) in facesDetected:
        face_gray = frameGray[y:y + h, x:x + w]

        face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]

        resultados = face_recognition.compare_faces(imagens_referencia, face_encoding)

        if True in resultados:
            indice_max_correspondencia = np.argmax(resultados)
            nome_pessoa = nomes_pessoas[indice_max_correspondencia]
            cv2.putText(frame, nome_pessoa, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()