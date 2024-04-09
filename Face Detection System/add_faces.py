#COLLECT DATA 
import cv2 #OpenCV can be used for real-time image processing, object detection, face recognition, and many other applications.
import pickle # save your model on disc with dump() function and de-pickle it into your python code with load() function
import numpy as np
import os

video = cv2.VideoCapture(0) #This function is used to capture video from a camera or file and return it as a matrix. It can also be used to read video frames from a file.
facedetect = cv2.CascadeClassifier('C:/Users/KIIT/Desktop/Face Detection System/data/haarcascade_frontalface_default.xml') #Cascade classifiers are trained using several positive (with faces or objects) images and arbitrary negative (without faces or objects) images. OpenCV contains several pretrained cascading classifiers used in image processing to detect frontal views of faces and the upper body.

faces_data = []
i = 0
name = input("Enter your name: ")

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

        resized_img = cv2.resize(crop_img, (50, 50)) #cv2.resize is a function provided by the OpenCV library (cv2). It's used to resize an image to a new desired size.
        if len(faces_data) <= 100 and i % 10 == 0: #Takes 100 images of you
            faces_data.append(resized_img) #resized_img is stored in the array faces_data
        i = i + 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1) #cv2.puttext(): This is a function within OpenCV specifically designed for drawing text onto an image.
                                                                                                          # the text will be positioned at (50, 50) pixels from the bottom-left corner of the image.
                                                                                                          #cv2.FONT_HERSHEY_COMPLEX-This constant specifies the font style to be used for the text. OpenCV offers various font styles
                                                                                                          #This value represents the font scale factor. It controls the size of the text. A value of 1 indicates the default font size.
                                                                                                          #(50,50,255):This tuple defines the color of the text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1) #draws a rectangle around the face

    cv2.imshow("frame", frame) #cv2.imshow function from OpenCV to display the current video frame (frame) in a window titled "frame". This allows you to see the real-time output of your face detection application.
    k = cv2.waitKey(1) #cv2.waitKey(1) pauses the program execution for 1 millisecond and checks for any pressed keys. It returns an integer value representing the key code if a key is pressed.
    if k == ord('q') or len(faces_data) == 100: #if the key q is pressed it exits
        break

video.release() # releases the resources associated with the video capture object (video). This is essential to free up system memory and avoid potential issues when dealing with video capture.
cv2.destroyAllWindows() # closes all OpenCV windows that were created using cv2.imshow

faces_data = np.asarray(faces_data) # Converts the faces_data list, which holds lists or NumPy arrays representing individual face images, into a NumPy array. 
faces_data = faces_data.reshape(100, -1)  # Reshape to 2D array with 100 rows 
#SAVING NAMES
# Data persistence (assuming data/ directory exists)
if 'names.pkl' not in os.listdir('data/'): #It checks if a file named names.pkl exists in a directory called data/.
    names = [name] * 100 #If the file doesn't exist, it creates a list containing the entered name repeated 100 times (assuming you want to store a name for each potential face) and saves it using pickle.dump.
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:  #If the file exists, it loads the existing list of names using pickle.load.
        names = pickle.load(f)
    names = names + [name] * 100 #It then appends the entered name 100 times to the existing list, effectively duplicating the name for each potential face.
    with open('data/names.pkl', 'wb') as f: #Finally, it saves the updated list of names back to the names.pkl file using pickle.dump
        pickle.dump(names, f)
#SAVING FACES
if 'faces_data.pkl' not in os.listdir('data/'): #It checks if a file named faces_data.pkl exists in a directory called data/.
    with open('data/faces_data.pkl', 'wb') as f: #If the file doesn't exist, it saves the reshaped faces_data NumPy array directly using pickle.dump.
        pickle.dump(faces_data, f)  # Dump faces_data
else: #else if the file exists, it first loads the existing data using pickle.load. This could be previously collected faces from past runs.
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)   #loads the existing data using pickle.load.
    faces = np.append(faces, faces_data, axis=0)  # Append new data with the existing collected data
    with open('data/faces_data.pkl', 'wb') as f: #Finally, it saves the updated combined dataset back to the faces_data.pkl file using pickle.dump.
        pickle.dump(faces, f)  

