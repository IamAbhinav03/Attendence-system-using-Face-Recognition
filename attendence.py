#                                   CONTACTLESS ATTENDENCE SYSTEM

#PLEASE NOTE THAT THE LIBRARY face-recognition SHOULD BE INSTALLED USING THE COMMAND
#pip install face-recognition 
#THE LIBRARY REQUIRES ADDITIONAL INSTALLATIONS IF YOU ARE INSTALLING IT ON WINDOWS. PLEASE REFER THE DOCUMENTATION GIVEN BELOW FOR
#THE PROPER INSTALLATION OF face-recognition LIBRARY IN YOUR OS.
#https://pypi.org/project/face-recognition/

#Execute the script on terminal using the command
#python attendence.py
#To end the script press Ctrl+c (Cmd+c for Mac users)
#=============================================================================================================================

#importing the necessary libraries
import cv2
import face_recognition
import os
import time
from datetime import datetime

#All the images of the employees for training should be stored in the dataset folder with their name
path = 'dataset'
images = []                     #A list that holds the images
classNames = []                 #A list that holds the name of the imgage(employees's name)
myList = os.listdir(path)
myList.remove('.DS_Store')      

#Iterating through the images and storing the necessary detatils to the lists
for image in myList:
    current_img = cv2.imread(f'{path}/{image}')
    images.append(current_img)
    classNames.append(os.path.splitext(image)[0])

#The model extracts information of the face. This process is called encoding. These encodings are compared to find
#matching faces
#Learn more about encodings from the documentation given above.
#Function to find encodings of the images.
def find_encodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

#Function to mark the attendence of the recognized faces
def markAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


KnownEncode = find_encodings(images)

print("[INFO] Encoding Complete")
print("[INFO] starting webcam....")

#Starting the webcamera
cap = cv2.VideoCapture(0)

#Giving inputs to the trained model
while True:
    sucess, frame = cap.read()
    frameSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)         #Resizing the caputred frame for fast processing
    frameSmall = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)              #face-recognition uses RGB images for recognition

    face = face_recognition.face_locations(frameSmall)              #Detection the location of faces in the frame
    encodeFace = face_recognition.face_encodings(frameSmall, face)  #Creating encoding of the located frame. 

    #Comparing the detected face with the encodings of trained images
    for ef,fl in zip(encodeFace, face):
        matches = face_recognition.compare_faces(KnownEncode, ef)
        faceDis = face_recognition.face_distance(KnownEncode, ef)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name) 
            #Drawing bounding boxes around the deteted faces
            y1, x2, y2, x1 = fl
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendence(name)

    #Showing live feed of the webcam
    cv2.imshow("Webcame", frame) 
    cv2.waitKey(1)