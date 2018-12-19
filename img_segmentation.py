import numpy as np
import cv2
from matplotlib import pyplot as plt


def convolute(img):
    
    height=img.shape[0]
    width=img.shape[1]
    kernel=[[8 for i in range(3)] for j in range(3)]
    kernel[0][0]=-1
    kernel[1][0]=-1
    kernel[2][0]=-1
    kernel[0][2]=-1
    kernel[1][2]=-1
    kernel[2][2]=-1
    kernel[0][1]=-1
    kernel[2][1]=-1
    
    
    print(kernel)
    img_x=[[0 for j in range(width+2)] for i in range(height+2)]
    
    padded_img=[[0 for j in range(width+2)] for i in range(height+2)]
    for i in range(1,height+1):
        for j in range(1,width+1):
            padded_img[i][j]=img[i-1][j-1]
            
    for x in range(1,height+1):
        for y in range(1,width+1):
            topLeft=padded_img[x-1][y-1]*kernel[0][0]
            topRight=padded_img[x-1][y+1]*kernel[0][2]
            bottomLeft=padded_img[x+1][y-1]*kernel[2][0]
            bottomRight=padded_img[x+1][y+1]*kernel[2][2]
            middleLeft=padded_img[x][y-1]*kernel[1][0]
            middleRight=padded_img[x][y+1]*kernel[1][2]
            sameTop=padded_img[x-1][y]*kernel[0][1]
            sameBottom=padded_img[x+1][y]*kernel[2][1]
            same=padded_img[x][y]*kernel[1][1]
            val=topLeft+topRight+bottomLeft+bottomRight+middleLeft+middleRight+sameTop+sameBottom+same
            img_x[x][y]=val
    
    maxElement=abs(img_x[0][0])
    for i in range(1,height+1):
        for j in range(1,width+1):
            if abs(img_x[i][j])>maxElement:
                maxElement=abs(img_x[i][j])
                
    for i in range(1,height+1):
        for j in range(1,width+1):
            img_x[i][j]=abs(img_x[i][j])/maxElement
    return img_x


def detect_porosity(img,threshold):
    result_image = np.copy(img)
    porosity_pts=[]
    for i in range(2,len(img)-2):
       for j in range (2,len(img[0])-2):
           if img[i][j] > threshold:
               result_image[i][j] = 255
               porosity_pts.append((i-1,j-1))
           else :
               result_image[i][j] = 0
    result_image=result_image*255
    font = cv2.FONT_HERSHEY_SIMPLEX
    for pt in porosity_pts:
        cv2.putText(result_image,"("+str(pt[1])+","+str(pt[0])+")",(pt[1]+10,pt[0]+10),font,0.4,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite('task2.1.jpg' , np.asarray(result_image))


def segmentation(img):
    intensities = np.zeros(256)
    for i in range(len(img)):
       for j in range(len(img[0])):
           if img[i][j]>150:
               intensities[(img[i][j])]+=1
    
    plt.plot(intensities)
    
    for i in range(len(img)):
       for j in range(len(img[0])):
           if img[i][j] in (range(203,235)):
               img[i][j]=255
           else:
               img[i][j]=0
    cv2.imwrite('task2.2.jpg' , img)


threshold=0.4
img_matrix=cv2.imread('original_imgs/point.jpg',0)
img=convolute(img_matrix)
detect_porosity(img,threshold)

img_matrix=cv2.imread('original_imgs/segment.jpg',0)
segmentation(img_matrix)


topLeftCorner=[(180,126),(162,146),(187,150),(249,74),(335,20),(384,40)]
bottomRightCorner=[(201,139),(180,161),(201,163),(309,204),(370,289),(420,254)]


color_img=cv2.imread('task2.2.jpg')

for x,y in zip(topLeftCorner,bottomRightCorner):
    cv2.rectangle(color_img,x,y,(0,0,255),1)

cv2.imwrite("Bounding Box.jpg",color_img)