from sklearn.neighbors import KNeighborsClassifier
import cv2 #OpenCV can be used for real-time image processing, object detection, face recognition, and many other applications.
import pickle # save your model on disc with dump() function and de-pickle it into your python code with load() function
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
def speak(str1):
      speak=Dispatch(("SAPI.SpVoice"))
      speak.Speak(str1)

video = cv2.VideoCapture(0) #This function is used to capture video from a camera or file and return it as a matrix. It can also be used to read video frames from a file.
facedetect = cv2.CascadeClassifier('C:/Users/KIIT/Desktop/Face Detection System/data/haarcascade_frontalface_default.xml') #Cascade classifiers are trained using several positive (with faces or objects) images and arbitrary negative (without faces or objects) images. OpenCV contains several pretrained cascading classifiers used in image processing to detect frontal views of faces and the upper body.
with open('data/names.pkl', 'rb') as f:  #If the file exists, it loads the existing list of names using pickle.load.
        LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f) 


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)

COL_NAMES=['NAME','TIME']
while True:
    ref, frame = video.read() #Then, use the VideoCapture.read() method in a loop to read each frame from the camera as a NumPy array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #cvtColor converts frame to grey colour ,grey because many machine learning models find it easy to deal with grey colours
    faces = facedetect.detectMultiScale(gray, 1.3, 5)#This is a method provided by the cv2.CascadeClassifier class. It's used to detect objects (in this case, faces) in the given image based on the trained classifier.
                                                     #grey variable holds the grayscale image you obtained by converting the original frame    
                                                      #1.3: This is the scale factor. It indicates how much the image size will be reduced at each image scale. A higher value implies a coarser search and faster detection, but might miss smaller faces. Lower values lead to a finer search and potentially finding smaller faces, but it might be slower.
                                                      #5: This is the minimum number of neighbors required to be considered a face detection. It helps reduce false positives (detecting non-face objects as faces). A higher value increases the detection accuracy but might miss some valid faces.
    for (x, y, w, h) in faces:  #x: X-coordinate of the top-left corner of the face rectangle.
                                #y: Y-coordinate of the top-left corner of the face rectangle.
                                #w: Width of the face rectangle.
                                #h: Height of the face rectangle.
        crop_img = frame[y:y+h, x:x+w, :] #y:y+h, x:x+w: This part specifies the rectangular region to be cropped within the frame image. It's a slicing operation using NumPy indexing

        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1,-1) #cv2.resize is a function provided by the OpenCV library (cv2). It's used to resize an image to a new desired size.
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist=os.path.isfile("Attendance/Attendance_"+date+".csv")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)

        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1) #draws a rectangle around the face
        attendance=[str(output[0]),str(timestamp)]

    cv2.imshow("frame", frame) #cv2.imshow function from OpenCV to display the current video frame (frame) in a window titled "frame". This allows you to see the real-time output of your face detection application.
    k = cv2.waitKey(1) #cv2.waitKey(1) pauses the program execution for 1 millisecond and checks for any pressed keys. It returns an integer value representing the key code if a key is pressed.
    if k==ord('o'):
          speak("Attendance Taken..")
          time.sleep(3)
          if exist:
                  with open("Attendance/Attendance_"+date+".csv", "a") as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(attendance)
                  csvfile.close()
          else:
                with open("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                      writer=csv.writer(csvfile)
                      writer.writerow(COL_NAMES)
                      writer.writerow(attendance)
                csvfile.close()
           
    if k == ord('q'):
        break

video.release() # releases the resources associated with the video capture object (video). This is essential to free up system memory and avoid potential issues when dealing with video capture.
cv2.destroyAllWindows() # closes all OpenCV windows that were created using cv2.imshow



