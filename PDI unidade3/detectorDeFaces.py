
import cv2
import dlib
 
 
img = cv2.imread('C:/Users/Alisson Moreira/Desktop/alisson002.github.io/PDI unidade3/img3.jpg')
 
detector = dlib.get_frontal_face_detector()
 
 
while True:
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
 
    i = 0
    for face in faces:
 
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
 
        i = i+1
        
        cv2.putText(img, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(face, i)
 
    cv2.imshow('imagem', img)
 
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
 
 
img.release()
cv2.destroyAllWindows()