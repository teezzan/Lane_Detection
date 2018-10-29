# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:20:34 2018

@author: TEEE
"""
#importing modules
import numpy as np
import cv2

def drw_poly(lft,rgt,frame):
    for x1,y1,x2,y2 in lft:
        for xa1,ya1,xa2,ya2 in rgt:
            pts = np.array([[x1,y1],[x2,y2],[xa1,ya1],[xa2,ya2]], np.int32)
    pts = pts.reshape((-1,1,2))
    frame = cv2.polylines(frame,[pts],True,(0,255,255),3)

def extnd_lane(lft,lane='left',new_y=520):
    for x1,y1,x2,y2 in lft:
        slope=(y2-y1)/(x2-x1)
        inter=y1-(slope*x1)
    if lane=='left':
        
        y1=new_y
        x1=int((y1-inter)/slope)
        return [[x1,y1,x2,y2]]
    elif lane=='right':
        y2=new_y
        x2=int((y2-inter)/(slope))
        return [[x1,y1,x2,y2]]

def simply_dir(left,right):
    lft=[]
    rgt=[]
    mx2=0,
    mx1=10000,
    my2=10000,
    my1=0
    for line in left:
        for x1, y1, x2, y2 in line:
            mx1=min(mx1,x1)
            mx2=max(mx2,x2)
            my1=max(my1,y1)
            my2=min(my2,y2)
    lft.append([mx1,my1,mx2,my2])
    mx1=10000,
    mx2=0,
    my1=10000,
    my2=0
    for line in right:
        for x1, y1, x2, y2 in line:
            mx1=min(mx1,x1)
            mx2=max(mx2,x2)
            my1=min(my1,y1)
            my2=max(my2,y2)
    rgt.append([mx1,my1,mx2,my2])
    return lft,rgt
    
    
def split_dir(lines):
    left=[]
    right=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope=(y2-y1)/(x2-x1)
            if slope>0:
                right.append(line)
            elif slope==0:
                pass
            else:
                left.append(line)
    return left,right

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon). Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols*0.1, rows*0.95]
    top_left = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right = [cols*0.6, rows*0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

#importing the pictures

cap=cv2.VideoCapture('solidWhiteRight.mp4')
while True:
    ret,frame=cap.read()
    img1=frame
    #convert to grayscale
    #imgr1=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    #imgr2=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    
    #use color masking
    hls1=cv2.cvtColor(img1,cv2.COLOR_RGB2HLS)
    hsv1=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    lowerW = np.uint8([ 0, 200, 0])
    upperW = np.uint8([255, 255, 255])
    mask=cv2.inRange(hls1,lowerW,upperW)
    lower = np.uint8([ 20, 100, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(hsv1, lower, upper)
    # combine the mask
    mask1 = cv2.bitwise_or(mask, yellow_mask)
    img1=cv2.bitwise_and(img1,img1,mask=mask1)
    #convert to grayscale
    img1=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    
    #applying blur
    img1=cv2.GaussianBlur(img1, (9, 9), 0)
    
    
    
    #ROI
    img1=select_region(img1)
    
    
    #applying canny algorithm
    img1=cv2.Canny(img1, 30, 70)
    
    
    #hough line transform

    lines=cv2.HoughLinesP(img1, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
    left,right=split_dir(lines)
    lft,rgt=simply_dir(left,right)
#    rgt=extnd_lane(rgt,lane='right')
#    lft=extnd_lane(lft)
    for [x1,y1,x2,y2] in lft:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
    for [x1,y1,x2,y2] in rgt:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
    drw_poly(lft,rgt,frame)
    
    
    #displaying images
    cv2.imshow('img1',frame)
    k=cv2.waitKey(27)
    if k==27 & 0xff:
        break
cap.release()    
cv2.destroyAllWindows()
        