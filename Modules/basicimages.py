import cv2 as cv
import numpy as np

image = cv.imread(r"King Domino dataset\Cropped and perspective corrected boards\1.jpg")
image_gray = cv.imread(r"King Domino dataset\Cropped and perspective corrected boards\1.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("Raw", image)

image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow("HSV", image_hsv)

blur_kernel = np.ones((5,5), np.float32)/30
image_mean = cv.filter2D(image, ddepth=-1, kernel=blur_kernel)
cv.imshow("Blurred Raw", image_mean)

sobel_h_kernel = np.array([[2,2,4,2,2],
                         [1,1,2,1,1],
                         [0,0,0,0,0],
                         [-1,-1,-2,-1,-1],
                         [-2,-2,-4,-2,-2]])
image_h_sobel = cv.filter2D(image_gray, ddepth=-1, kernel=sobel_h_kernel)

sobel_v_kernel = np.array([[2,1,0,-1,-2],
                         [2,1,0,-1,-2],
                         [4,2,0,-2,-4],
                         [2,1,0,-1,-2],
                         [2,1,0,-1,-2]])
image_v_sobel = cv.filter2D(image_gray, ddepth=-1, kernel=sobel_v_kernel)
image_sobel = cv.vconcat([image_h_sobel, image_v_sobel])
cv.imshow("Sobel", image_sobel)

image_sobel_combined = cv.add(image_h_sobel, image_v_sobel)
cv.imshow("Sobel Combined", image_sobel_combined)

im = image_gray.copy()
ret,image_binary = cv.threshold(im,127,255,cv.THRESH_BINARY)
cv.imshow("Binary", image_binary)

#im = image_gray.copy()
#contours, hierachy = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#cv.imshow("Contours", im)
#image_drawn_contours = cv.drawContours(image_gray, contours, -1, (0, 255, 0), 3)
#cv.imshow("Contours On Image", image_drawn_contours)


cv.waitKey()
cv.destroyAllWindows()