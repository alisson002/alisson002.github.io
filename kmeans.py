import random 
import numpy as np
import cv2

nclusters = 8
attempts = 1

img = cv2.imread("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/kmeans.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
samples = img.reshape((-1,3))

samples = np.float32(samples)

for i in range(1, 11):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.1*random.randint(1,100))
    x, labels, centers = cv2.kmeans(samples, nclusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    vals = centers[labels.flatten()]
    newimg = vals.reshape((img.shape))

    cv2.imshow('New Image', newimg)
    cv2.imwrite("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/kmeansGif/kmeans.png", newimg)
    cv2.waitKey(0) 

cv2.waitKey(0)
cv2.destroyAllWindows()