import numpy as np
import cv2

def detect(image):
    img=image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,th=cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    contours,hierachy=cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    if len(contours)==1:
        mi,ma,_,_=cv2.minMaxLoc(gray)
        str=cv2.convertScaleAbs(gray,alpha=255/(ma-mi),beta=-255*mi/(ma-mi))
        ret,th=cv2.threshold(str,28,255,cv2.THRESH_BINARY)
        contours,hierachy=cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i])>2000:
            break
    contour=contours[i]
    rect=cv2.minAreaRect(contour)
    w,h=int(rect[1][0]),int(rect[1][1])
    box=cv2.boxPoints(rect)
    src=np.array(box,dtype='float32')
    dst=np.array([[0,h-1],[0,0],[w-1,0],[w-1,h-1]],dtype="float32")
    m=cv2.getPerspectiveTransform(src,dst)
    res=cv2.warpPerspective(img,m,(w,h))
    hsv=cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
    R=cv2.inRange(hsv,(0, 51, 51),(10, 255, 255))
    G=cv2.inRange(hsv,(51, 51, 51),(70, 255, 255))
    B=cv2.inRange(hsv,(111, 51, 51),(130, 255, 255))
    Rc=np.count_nonzero(R)
    Gc=np.count_nonzero(G)
    Bc=np.count_nonzero(B)
    M=max(Rc,Gc,Bc)
    if Rc==M:
        return 'R'
    elif Gc==M:
        return 'G'
    else:
        return 'B'
    
filename='test1.PNG'
image=cv2.imread(filename)
cv2.imshow('2',image)
cv2.waitKey(0)
print(detect(image))