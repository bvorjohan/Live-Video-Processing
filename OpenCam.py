import cv2
import numpy as np
import math

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

brow_top=2
brow_bottom=1.5
brow_out=5
brow_in=1

font = cv2.FONT_HERSHEY_SIMPLEX

e0=(0,0,0,0)
e1=(0,0,0,0)
th=0
dist=1
midx=0
midy=0
sth=0
cth=1

b00=(0,0)
b01=(0,0)
b10=(0,0)
b11=(0,0)
mag=.18

while True:
    _, frame = cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray,1.3,5)
    #for (x,y,w,h) in eyes:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    if(len(eyes)==2):
        e0=eyes[0]
        e1=eyes[1]
        th=math.atan((e0[1]-e1[1])/(e0[0]-e1[0]))

        dist=math.sqrt(((e0[0]-e1[0])**2)+((e0[1]-e1[1])**2))
        midx = ( e0[0] + e0[2]/2 + (e1[0] + e1[2]/2) ) / 2
        midy = ( e0[1] + e0[3]/2 + (e1[1] + e1[3]/2) ) / 2
        sth=math.sin(th)
        cth=math.cos(th)

        b00=((midx+mag*dist*(-brow_in*cth+brow_bottom*sth)),(midy+mag*dist*(-brow_in*sth-brow_bottom*cth)))
        b01=((midx+mag*dist*(-brow_out*cth+brow_top*sth)),(midy+mag*dist*(-brow_out*sth-brow_top*cth)))
        b10=((midx-mag*dist*(-brow_in*cth-brow_bottom*sth)),(midy+mag*dist*(brow_in*sth-brow_bottom*cth)))
        b11=((midx-mag*dist*(-brow_out*cth-brow_top*sth)),(midy+mag*dist*(brow_out*sth-brow_top*cth)))

    cv2.line(frame,(int(b01[0]),int(b01[1])),(int(b00[0]),int(b00[1])),(19,69,139),int(.1*dist))
    cv2.line(frame,(int(b11[0]),int(b11[1])),(int(b10[0]),int(b10[1])),(19,69,139),int(.1*dist))
    #cv2.putText(frame,str(sth),(100,50),font,1,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("Face",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
