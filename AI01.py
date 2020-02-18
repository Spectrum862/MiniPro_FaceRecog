import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os, cv2, sys, numpy
import requests
from io import BytesIO
import tkinter.font as font
import numpy as np
from PIL import Image
import pandas as pd
#----------------------------------------------------------
def createdata():
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'  #All the faces data will be present this folder

    path = os.path.join(datasets)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (150, 150)    # defining the size of images 


    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this

    #Input name and save to csv
    name  = entrybox_1.get()
    if os.path.isfile('name.csv'):
        namef = pd.read_csv('name.csv')
        (nrow,ncolumn) = namef.shape
        idn = nrow+1
        new_row = {'Name' : name}
        namef = namef.append(new_row, ignore_index = True)
        namef.to_csv('name.csv', index = False)
    else:
        df = pd.DataFrame({'name' : [name]})
        df.to_csv('name.csv',index = True)


    # The program loops until it has 100 images of the face.
    count = 1
    while count < 101: 
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))       
            cv2.imwrite("datasets/Person." + str(idn) + '.' + str(count) + ".jpg", face_resize)
        count += 1
        
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(150)
        if key == 13:
            break

        label_5.config(text="ระบบจดจำใบหน้าเสร็จสิ้น โปรดเลือกการเทรน")
#-------------------------------------------------------------
def trainingface():
        
    #Method for checking existence of path i.e the directory
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    # We will be using Local Binary Patterns Histograms for face recognization since it's quite accurate than the rest
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    #method getting the images and label data

    def getImagesAndLabels(path):

        # Getting all file paths
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
        
        #empty face sample initialised
        faceSamples=[]
        
        # IDS for each individual
        ids = []

        # Looping through all the file path
        for imagePath in imagePaths:

            # converting image to grayscale
            PIL_img = Image.open(imagePath).convert('L')

            # converting PIL image to numpy array using array() method of numpy
            img_numpy = np.array(PIL_img,'uint8')

            # Getting the image id
            id = int(os.path.split(imagePath)[-1].split(".")[1])

            # Getting the face from the training images
            faces = detector.detectMultiScale(img_numpy)

            # Looping for each face and appending it to their respective IDs
            for (x,y,w,h) in faces:

                # Add the image to face samples
                faceSamples.append(img_numpy[y:y+h,x:x+w])

                # Add the ID to IDs
                ids.append(id)

        # Passing the face array and IDs array
        return faceSamples,ids

    # Getting the faces and IDs
    faces,ids = getImagesAndLabels('datasets')

    # Training the model using the faces and IDs
    print('Training....')
    recognizer.train(faces, np.array(ids))

    # Saving the model into s_model.yml
    assure_path_exists('saved_model/')
    recognizer.write('saved_model/s_model.yml')



    label_5.config(text="การเทรนเสร็จสิ้น สามารถเริ่มการสแกนได้")
#-------------------------------------------------------------

def face_recognition():
    
    #Method for checking existence of path i.e the directory
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Create Local Binary Patterns Histograms for face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    assure_path_exists("saved_model/")

    # Load the  saved pre trained mode
    recognizer.read('saved_model/s_model.yml')

    # Load prebuilt classifier for Frontal Face detection
    cascadePath = "haarcascade_frontalface_default.xml"

    # Create classifier from prebuilt model
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # font style
    font = cv2.FONT_HERSHEY_SIMPLEX

    # readname
    name = pd.read_csv('name.csv')
    (nrow,ncolumn) = name.shape
    namelist = name.Name.tolist()

    # Initialize and start the video frame capture from webcam
    webcam = cv2.VideoCapture(0)
    webcam.set(3,1920)
    webcam.set(4,1080)
    webcam.set(5,30)

    # Looping starts here
    while True:
        # Read the video frame
        ret, im =webcam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # Getting all faces from the video frame
        faces = faceCascade.detectMultiScale(gray, 1.2,5) #default

        # For each face in faces, we will start predicting using pre trained model
        for(x,y,w,h) in faces:
            # Create rectangle around the face
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

            # Recognize the face belongs to which ID
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])  #Our trained model is working here
            # Set the name according to id
            
            if confidence > 90 :
                Id = "Unknown" 
            else : 
                for i in range(nrow) :
                    if Id == i+1 :
                        nametext = namelist[i]
                        Id = nametext + " {0:.2f}%".format(round(100-confidence, 2))
                    # Put text describe who is in the picture  
                    else: 
                        pass

            # Set rectangle around face and name of the person
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

        # Display the video frame with the bounded rectangle
        cv2.imshow('im',im) 

        # press q to close the program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Terminate video
    webcam.release()

    # Close all windows
    cv2.destroyAllWindows()


#-------------------------------------------------------------

def inputname():
    nameuser = entrybox_1.get()
    print(nameuser)
    label_5.config(text="คุณ"+nameuser+"เราได้ชื่อของคุณแล้ว\nโปรดกด จดจำใบหน้า")  

#สร้างหน้าต่าง
root = Tk()
#หัวชื่อโปรแกรม
root.title('Face Recognition')
#ปรับขนาดพร้อมตำแหน่งที่เกิดของหน้าต่าง
root.geometry('500x650+500+80')


myFont = font.Font(size=20)

#ทำให้ไม่สามารถปรับขนาดหน้าต่างได้
root.resizable(width =FALSE, height =FALSE)
#ปรับสี
#root.configure(background="#94d42b")

#root.option_add("*Font","consolas 25")
#Label(root,text="\nตรวจจับภาพบุคคล และจดจำใบหน้า").pack()

label_1 =Label(root, text = "ตรวจจับภาพบุคคล และจดจำใบหน้า",font="time 20",height=2)

label_1.pack()


label_2 =Label(root, text = "โปรดระบุชื่อของคุณ",font="time 15",height=2)
label_2.pack()


entrybox_1 = Entry(root,font="time 20",bg ="black",fg="white")
entrybox_1.pack(fill=X, padx=100, pady=20)

#สร้างปุ่ม

button_0 = Button(root,text="ใส่ชื่อ",command=inputname, height = 1, width = 20)
button_0['font'] = myFont
button_0.pack()

button_1 = Button(root,text="เริ่มการจดจำใบหน้า",command=createdata, height = 1, width = 20)
button_1['font'] = myFont
button_1.pack()

button_2 = Button(root,text="เทรน",command=trainingface, height = 1, width =20 )
button_2['font'] = myFont
button_2.pack()

label_3 =Label(root, text = "\nระบบสแกนใบหน้า\n*กดQเพื่อออก\n",font="time 15",height=2)
label_3.pack(fill=X, padx=10)

button_3 = Button(root,text="สแกนใบหน้า",command=face_recognition, height = 1, width =20 )
button_3['font'] = myFont
button_3.pack()

label_4 =Label(root, text = "\nสถานะ\n",font="time 15",height=2)
label_4.pack(fill=X, padx=10)

label_5 =Label(root, text = "\nนิ่ง\n",font="time 15",height=2)
label_5.pack(fill=X, padx=10)

root.mainloop()
