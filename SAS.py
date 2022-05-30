import cv2
import os
from datetime import datetime
from csv import writer
import face_recognition

vid = cv2.VideoCapture(0)

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
size=len(myList)
print(myList)

encodeList = []
def findEncodings():
    for i in range(size):
        img = face_recognition.load_image_file('ImagesAttendance/'+myList[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(myList[i])
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

def markAttendance(name):
    with open('Attendance.csv', mode ='r+')as f: 
    #with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            #dtString = now.strftime('%H:%M:%S')
            #f.writelines(f'{name},present,{now}')
            with open('Attendance.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow([name,'present',now])
                print(name,'present',now)

print('Encoding start')
findEncodings()
print('Encoding Complete')

while(True):
    
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    
    cv2.imwrite("img.jpg", frame)
    
    imgElon = face_recognition.load_image_file('img.jpg')
    imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

    if(len(face_recognition.face_locations(imgElon))==0):
        continue

    faceLoc = face_recognition.face_locations(imgElon)[0]
    encodeElon = face_recognition.face_encodings(imgElon)[0]

    for i in range(size):                  
        results = face_recognition.compare_faces([encodeElon],encodeList[i])
        faceDis = face_recognition.face_distance([encodeElon],encodeList[i])
        print(i,results[0],faceDis)

        if results[0] == True:
            markAttendance(os.path.splitext(myList[i])[0])
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()



