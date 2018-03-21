import time
import io
import threading
import cv2
import subprocess
import csv,os
import numpy as np
from PIL import Image
import requests,json
from pprint import pprint

#from pyautogui import press

class Camera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  #time of last client access to the camera
    p = None
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    cam = None
    fontface=cv2.FONT_HERSHEY_SIMPLEX
    Stop_Thread = False
    
    def Start_Recognizer(self):
        Camera.thread = threading.Thread(target= self._thread)
        Camera.Stop_Thread = False
        Camera.thread.start()
       
    @classmethod
    def stop_cam(cls):
        cls.Stop_Thread = True
        
        #Camera.cam.release()
        #cv2.destroyAllWindows()
        

    def storedataset(self,names):
        ids = []
        with open('names.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                 ids.append(int(row['id']))
        if not ids:
            ids = [0]
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
        file_base_path = 'imagesdb/s' + str(last_id)
        if not os.path.exists(file_base_path):
            os.makedirs(file_base_path)
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetector.detectMultiScale(gray, 1.3, 5);
            for (x,y,w,h) in faces:
                sampleNum = sampleNum + 1;
                cv2.imwrite(file_base_path+"/image." + str(last_id) + "." + str(sampleNum) + ".jpg",gray[y:y+h, x:x+w])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('faces',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif(sampleNum>30):
                break
        cam.release()
        cv2.destroyAllWindows()
        Camera.TrainData()
        return last_id
    

    @classmethod
    def _thread(self):
        Camera.cam = cv2.VideoCapture(0)
        rec = cv2.createLBPHFaceRecognizer()
        rec.load("trainingData.yml")
        name = "un"
        id = 0
        namedict = {}
        with open('names.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                namedict[int(row['id'])] = row['name']
        while(True):
            ret,img= Camera.cam.read();
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = Camera.faceDetect.detectMultiScale(gray,1.3,5);
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                id,conf =rec.predict(gray[y:y+h,x:x+w])
                #print (type(id))
                print conf
                if(conf<90):
                    if(id>0):
                        name = namedict[id]
                    else:
                        name = "unknown"
                        id = 0
                else:
                    name = "unknown"
                    id = 0
            #print id
                cv2.putText(img,name,(x,y+h),Camera.fontface,1,(0,0,255),2);
                cv2.imshow("rec",img);
                '''if( id == 0):
                    cv2.imwrite("Face_Recognition/static/img/yash.jpg",img)
                    fbid = 1011083955671891	
                    PAGE_ACCESS_TOKEN = "EAACyZBrP01zgBAD1lMPOJc5m5uAZBgCwkctdHMV5WYiafYpx2cwYonEZA3Ep8zdtkwAZCHHNDPKRfWBY6gQWzKpthWCDI4zZCXEI57cWFwApOi4tCdgAz3mXbPQKvGwvUF40ceO5AxA5xMoF4WZAd3ZBEYpTBjNYADAAywWnKnXIAZDZD"
                    post_message_url = 'https://graph.facebook.com/v2.6/me/messages?access_token=%s'%PAGE_ACCESS_TOKEN
                    response_msg = json.dumps({"recipient":{"id":fbid}, "message":{"attachment":{"type":"image", "payload":{"url":'https://66bf3dd2.ngrok.io/static/img/yash.jpg'}}}})
                    status = requests.post(post_message_url, headers={"Content-Type": "application/json"},data=response_msg)
                    pprint(status.json())
                    time.sleep(10)'''
            
            if(Camera.Stop_Thread == True):
                break;
        Camera.cam.release()
        cv2.destroyAllWindows()
		
    @classmethod				
    def TrainData(cls):
	recognizer = cv2.createLBPHFaceRecognizer()
	def getImagesWithID(path):
            #imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            dirs = os.listdir(path)
            faces=[]
            IDs=[]
            for dir_name in dirs:
                label = int(dir_name.replace("s", ""))
                subject_dir_path = path + "/" + dir_name
                #imagePaths = os.listdir(subject_dir_path)
                imagePaths = [os.path.join(subject_dir_path,f) for f in os.listdir(subject_dir_path)]
                #print(imagePathes)
                for imagePath in imagePaths:
                    faceImg = Image.open(imagePath).convert('L');
                    faceNp = np.array(faceImg,'uint8')
                    #ID = int(os.path.split(imagePath) [-1].split('.')[1])
                    faces.append(faceNp)
                    IDs.append(label)
            return faces,IDs

        faces,ids  = getImagesWithID('imagesdb')
        recognizer.train(faces,np.array(ids))
	recognizer.save('trainingData.yml')
        
'''def initialize(self):
        if Camera.thread is None:
            # start background frame thread
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()
            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def get_frame(self):
        #Camera.last_access = time.time()
        ret,img= Camera.cam.read();
        ret, jpeg = cv2.imencode('.jpg', img)
        self.frame = jpeg.tobytes()
        #self.initialize()
        return self.frame'''
