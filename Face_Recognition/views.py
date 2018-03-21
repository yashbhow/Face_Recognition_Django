from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse
from camera_web import Camera
import cv2,subprocess
import csv
# Create your views here.
cam_open = False
def index(request):
	return render(request,'Face_Recognition/index.html')

def start(request):
        a = Camera().Start_Recognizer()        
	return render(request,'Face_Recognition/start.html')

def live(request):
	return render(request,'Face_Recognition/video_feed_show.html')

def name(request):
        names = request.POST['name']
        Camera().storedataset(names)
        return HttpResponse('you submitted this:' + names)

def stop(request):
        Camera.stop_cam()
	return render(request,'Face_Recognition/stop.html')
'''
def storedataset(names):
        ids = []
        with open('names.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                 ids.append(int(row['id']))
        last_id = ids[-1]
        last_id += 1
        myData = [[last_id,names]]
        myFile = open('names.csv', 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(myData)
        cam = cv2.VideoCapture(0);
        faceDetector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        sampleNum = 0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetector.detectMultiScale(gray, 1.3, 5);
            for (x,y,w,h) in faces:
                sampleNum = sampleNum + 1;
                cv2.imwrite("imagesdb/image." + str(last_id) + "." + str(sampleNum) + ".jpg",gray[y:y+h, x:x+w])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('faces',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif(sampleNum>30):
                break
        cam.release()
        cv2.destroyAllWindows()
        p = subprocess.Popen('b.bat',creationflags=subprocess.CREATE_NEW_CONSOLE)
''''''
def gen(camera): 
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(self):
    return StreamingHttpResponse(gen(Camera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')


'''
