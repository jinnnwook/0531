import cv2
import numpy as np
img=cv2.imread('public_imgs_public_20.PNG')
gray=cv2.imread('public_imgs_public_20.PNG',cv2.IMREAD_GRAYSCALE)
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
    print("R")
elif Gc==M:
    print("G")
else:
    print("B")