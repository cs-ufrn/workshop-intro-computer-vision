import cv2
import numpy as np

np.set_printoptions(threshold=100)

imgcars = cv2.imread("../img/carpeople.jpg", 0)
imgwrit = cv2.imread("../img/handwriting.jpg",0)
imgcat = cv2.imread("../img/cat.jpg",0)
imgiris = cv2.imread("../img/iris.jpg",0)

imgcars = imgcars[20:400,100:800]

imgcars = cv2.resize(imgcars, (400,200), interpolation=cv2.INTER_CUBIC)
imgwrit = cv2.resize(imgwrit, (400,200), interpolation=cv2.INTER_CUBIC)
imgcat = cv2.resize(imgcat, (400,200), interpolation=cv2.INTER_CUBIC)
imgiris = cv2.resize(imgiris, (400,200), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("../img/gcarpeople.jpg", imgcars)
cv2.imwrite("../img/ghandwriting.jpg", imgwrit)
cv2.imwrite("../img/gcat.jpg", imgcat)
cv2.imwrite("../img/giris.jpg", imgiris)

f = open("../img/gcarpeople.txt", "w")
f.write(np.array2string(imgcars))
f.close()

f = open("../img/ghandwriting.txt", "w")
f.write(np.array2string(imgwrit))
f.close()

f = open("../img/gcat.txt", "w")
f.write(np.array2string(imgcat))
f.close()

f = open("../img/giris.txt", "w")
f.write(np.array2string(imgiris))
f.close()

