import cv2
import numpy as np
import math

# Haar Cascade loaders
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade_mouth.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_face.xml")

# Image loaders
glasses01=cv2.imread("glasses01.png",0)
translatedAviator=glasses01
aviatorFactor=3



# Load our overlay image: mustache.png
imgAviator = cv2.imread("glasses01.png",-1)

# Create the mask for the mustache
aviator_mask = imgAviator[:,:,3]

# Create the inverted mask for the mustache
aviator_mask_inv = cv2.bitwise_not(aviator_mask)

# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgAviator = imgAviator[:,:,0:3]
origAviatorHeight, origAviatorWidth = imgAviator.shape[:2]



# Initializing eyebrow constants
brow_top=2
brow_bottom=1.5
brow_out=5
brow_in=1
mag=.18

# Choosing font for display
font = cv2.FONT_HERSHEY_SIMPLEX

# Initializing eye parameters (dummy values)
e0=(0,0,0,0)
e1=(0,0,0,0)
th=0
dist=1
midx=300
midy=300
sth=0
cth=1
b00=(0,0)
b01=(0,0)
b10=(0,0)
b11=(0,0)

# Variables used in image processing (Abstract away?)
rows=1
cols=1

# Booleans indicating what is to be operated
eyebrows=False
aviators=False
lips=True

# Opens webcam object (specific to machine)
cap = cv2.VideoCapture(0)

# Opens window
cv2.namedWindow("Main Display",1)


'''
Trackeyes takes a haarcascade eye output with 2 eye objects and outputs useful
position data about these eyes. Assumes e is an array with 2 eye objects.
'''
def trackEyes(eyes):
    e0=eyes[0]
    e1=eyes[1]

    # Angle and distance between eyes
    th=math.atan((e0[1]-e1[1])/(e0[0]-e1[0]))
    dist=math.sqrt(((e0[0]-e1[0])**2)+((e0[1]-e1[1])**2))

    # Midpoints of eyes
    midx = ( e0[0] + e0[2]/2 + (e1[0] + e1[2]/2) ) / 2
    midy = ( e0[1] + e0[3]/2 + (e1[1] + e1[3]/2) ) / 2

    return (e0,e1,th,dist,midx,midy)


'''
Main program structure loop
'''
while True:
    _, frame = cap.read()
    frame=cv2.flip(frame,1)
    roi_color=frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if(eyebrows):

        eyes=eye_cascade.detectMultiScale(gray,1.3,5)


        if(len(eyes)==2):
            e0,e1,th,dist,midx,midy = trackEyes(eyes)
            sth=math.sin(th)
            cth=math.cos(th)

            b00=((midx+mag*dist*(-brow_in*cth+brow_bottom*sth)),(midy+mag*dist*(-brow_in*sth-brow_bottom*cth)))
            b01=((midx+mag*dist*(-brow_out*cth+brow_top*sth)),(midy+mag*dist*(-brow_out*sth-brow_top*cth)))
            b10=((midx-mag*dist*(-brow_in*cth-brow_bottom*sth)),(midy+mag*dist*(brow_in*sth-brow_bottom*cth)))
            b11=((midx-mag*dist*(-brow_out*cth-brow_top*sth)),(midy+mag*dist*(brow_out*sth-brow_top*cth)))

        cv2.line(frame,(int(b01[0]),int(b01[1])),(int(b00[0]),int(b00[1])),(19,69,139),int(.1*dist))
        cv2.line(frame,(int(b11[0]),int(b11[1])),(int(b10[0]),int(b10[1])),(19,69,139),int(.1*dist))


    if(aviators):
        eyes=eye_cascade.detectMultiScale(gray,1.3,5)


        if(len(eyes)==2):
            e0,e1,th,dist,midx,midy = trackEyes(eyes)

        # The mustache should be three times the width of the nose
        aviatorWidth =  aviatorFactor * dist
        aviatorHeight = aviatorWidth * origAviatorHeight / origAviatorWidth

        # Center the mustache on the bottom of the nose
        x1 = int(midx - (aviatorWidth/aviatorFactor))
        x2 = int(midx + (aviatorWidth/aviatorFactor))
        y1 = int(midy - (aviatorHeight/aviatorFactor))
        y2 = int(midy + (aviatorHeight/aviatorFactor))


        # Check for clipping
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > 640:
            x2 = 640
        if y2 > 480:
            y2 = 480

        # Re-calculate the width and height of the mustache image
        aviatorWidth = x2 - x1
        aviatorHeight = y2 - y1

        # Re-size the original image and the masks to the mustache sizes
        # calcualted above
        aviator = cv2.resize(imgAviator, (aviatorWidth,aviatorHeight), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(aviator_mask, (aviatorWidth,aviatorHeight), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(aviator_mask_inv, (aviatorWidth,aviatorHeight), interpolation = cv2.INTER_AREA)



        # take ROI for mustache from background equal to size of mustache image
        roi = roi_color[y1:y2, x1:x2]

        # roi_bg contains the original image only where the mustache is not
        # in the region that is the size of the mustache.
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # roi_fg contains the image of the mustache only where the mustache is
        roi_fg = cv2.bitwise_and(aviator,aviator,mask = mask)

        # join t he roi_bg and roi_fg
        dst = cv2.add(roi_bg,roi_fg)

        # place the joined image, saved to dst back over the original image
        roi_color[y1:y2, x1:x2] = dst


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame,(x,y+int(h*.6)),(x+w,y+h),(255,0,0),2)
        roi_gray_lips = gray[y+int(h*.6):y+h, x:x+w]
        roi_color_lips = frame[y+int(h*.6):y+h, x:x+w]

        if(lips):
            mouths=mouth_cascade.detectMultiScale(roi_gray_lips,1.3,5)


            for (lx, ly, lw, lh) in mouths:
                rx0 = x+lx-20
                rx1 = x+lx+lw+20
                ry0 = y+int(h*.6)+ly
                ry1 = y+int(h*.6)+ly+lh
                # cv2.rectangle(frame,(rx0,ry0),(rx1,ry1),(0,255,0),2)
                lip_area = frame[ry0:ry1, rx0:rx1]
                lip_area = cv2.flip(lip_area,0)
                lip_area = cv2.flip(lip_area,1)
                frame[ry0:ry1, rx0:rx1] = lip_area






    cv2.imshow("Main Display",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
