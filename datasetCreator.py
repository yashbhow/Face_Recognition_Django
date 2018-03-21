import cv2
cam = cv2.VideoCapture(0);
faceDetector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id = raw_input('enter user id')
sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1;
        cv2.imwrite("imagesdb/image." + id + "." + str(sampleNum) + ".jpg",gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('faces',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif(sampleNum>30):
        break
cam.release()
cv2.destroyAllWindows()
