#TASK 2
import cv2
import numpy as np
#initial noise reduction of iron man image
img_1=cv2.imread("iron_man_noisy.jpg")
img_1=cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)
img=cv2.medianBlur(img_1,3)

kernel = np.ones((2,2),np.uint8)
img= cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

img=cv2.resize(img,(500,600))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("Iron Man",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#final noise reduction of iron man image
noisy_image=cv2.imread("iron_man_noisy.jpg",0)
_,thresh=cv2.threshold(noisy_image,127,255,cv2.THRESH_BINARY)
num_labels,labels,stats,_=cv2.connectedComponentsWithStats(thresh,connectivity=8)
clean=np.zeros_like(thresh)
min_area=7
for i in range(1,num_labels):
    area=stats[i,cv2.CC_STAT_AREA]
    if area>min_area:
        clean[labels==i]=255
clean=cv2.resize(clean,(500,600))
cv2.imshow("Clean_Img",clean)
cv2.waitKey(0)
cv2.destroyAllWindows()
#initial method
image=cv2.imread("noisy.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
new_image=cv2.bilateralFilter(image, d=21, sigmaColor=200, sigmaSpace=150)
new_image=cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)
cv2.imshow("CleanImage",new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#final method
image=cv2.imread("noisy.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
new_img=cv2.GaussianBlur(image,(3,3),0)
new_img=cv2.bilateralFilter(new_img, d=21, sigmaColor=200, sigmaSpace=150)
#new_img=cv2.addWeighted(image, 1.5, new_img, -0.5, 0)
new_img=cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)
cv2.imshow("Clean_Image",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()