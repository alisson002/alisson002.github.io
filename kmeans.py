import random 
import numpy as np
import cv2


nclusters = 8
attempts = 1

img = cv2.imread("C:/Users/Alisson Moreira/Desktop/PDI---Unidade1/kmeans.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
samples = img.reshape((-1,3))

lista = [img, img, img, img, img, img, img, img, img, img, img]

# convert to np.float32
samples = np.float32(samples)

# define criteria, number of clusters and apply kmeans()
# out -> labels: the classification of each pixel
# out -> centers: RGB value class of each cluster

for i in range(1, 11):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.1*random.randint(1,100))
    _, labels, centers = cv2.kmeans(samples, nclusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make the original image
    centers = np.uint8(centers)
    #centers[1:]= [255, 255, 255]
    vals = centers[labels.flatten()]
    newimg = vals.reshape((img.shape))

    cv2.imshow('New Image', newimg)
    cv2.imwrite("C:/Users/Alisson Moreira/Desktop/PDI---Unidade1/kmeansGif/kmeans.png", newimg)
    cv2.waitKey(0) 
     



#cv2.imshow('New Image', lista)

cv2.waitKey(0)
cv2.destroyAllWindows()