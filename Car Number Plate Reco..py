#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import imutils
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd="C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    #read the image file
image=cv2.imread("C:\\Users\\HP\\Desktop\\Car-Number1-Plate.jpg")

# resize the image change width to 500
image=imutils.resize(image,width=500)

#Display the Original image
cv2.imshow("Original image",image)
cv2.waitKey(0)

#RGB to grayscale conversion
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("1.- Grayscale conversion",gray)
cv2.waitKey(0)

#noise removal with iterative bilateral filter(removes moise while preserving edges)
gray=cv2.bilateralFilter(gray,11,17,17)
cv2.imshow("2. - Bilateral Filter",gray)
cv2.waitKey(0)

#Find edges of the grayscale images
edged=cv2.Canny(gray,170,200)
cv2.imshow("3. - Canny Edges",edged)
cv2.waitKey(0)

#Find Contours based on edges
cnts,new=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


#Create copy of original image to draw all contours
img1=image.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)
cv2.imshow("4. - All Contours",img1)
cv2.waitKey(0)

#Sort contours based on their area keeping minimum required area as '30' (anything smallesty than this will not be considered)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:30]
NumberPlateCnt=None  # We currently had no number plate contour


#top 30 contours
img2=image.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
cv2.imshow("5. - Top 30 Contours",img2)
cv2.waitKey(0)

#loop over our contours to find the best possible approximate contour of number plate
count=0
idx=7
for c in cnts:
    peri=cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    #print ("approx=",apprpox)
    if len(approx)==4:    #select the contours with 4 corners
        NumberPlateCnt=approx      #this is our approx number plate contour
        #crop those contours and store it in cropped image folder
        x,y,w,h=cv2.boundingRect(c) #this will findout co-ord for plate
        new_img=image[y: y +h ,x:x +w]#create new image
        
        cv2.imwrite("C:\\Users\\HP\Desktop\\images\\ci" +str(idx) +'.jpg',new_img) #store new image
        
        idx+=1
        break
        cv2.waitKey(0)

        
#drawing the selected contour on the original image
#prinr(NumberPlateCnt)
cv2.drawContours(image,[NumberPlateCnt],-1,(0,255,0),3)
cv2.imshow("Final image with number plate detected",image)
cv2.imshow("Cropped image",new_img)


cv2.waitKey(0)



# In[ ]:




