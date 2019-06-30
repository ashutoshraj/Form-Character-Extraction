# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:41:35 2018

@author: Ashutosh Raj
"""

import cv2
import numpy as np
import sys

def boxDetection(image):

	_,thresh = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

	dilated = cv2.dilate(thresh,kernel,iterations = 2)

	_, contours, hir = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	result = []

	for contour in contours:

		[x,y,w,h] = cv2.boundingRect(contour)

		if h>80 and w>80:

			continue

		if h<63 or w<63:

			continue
		# draw rectangle around contour on original image

		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)

		result.append([x,y,w,h])

	np.save("result.npy", result)

	resultant_image = sort_contours(thresh)

	return resultant_image

def sort_contours(image):

	cnt = np.load(r"result.npy")

	cnts = list(cnt)

	max_width = np.sum(cnt[::, (0, 2)], axis=1).max()

	max_height = np.max(cnt[::, 3])

	nearest = max_height * 1.4

	cnts.sort(key=lambda r: (int(nearest * round(float(r[1])/nearest)) * max_width + r[0]))

	arr = []

	for x, y, w, h in cnts:

        	img_n = cv2.resize((image[y:y+h,x:x+w]),(140,140))

        	arr.append(img_n)

	narr = arr[0]

	for i in range(1, len(arr)):

		narr = np.concatenate([narr, arr[i]], axis=1)			

	return narr

if __name__ == '__main__':

	#Reading the image and converting it into Grayscale for proper thresholding 
	img = cv2.imread(sys.argv[1], 0)

	image = boxDetection(img)

	cv2.imshow('sorted_and_extracted', image)

	cv2.imwrite('result.jpg', image)

	cv2.waitKey(0)
