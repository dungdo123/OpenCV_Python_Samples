import cv2
import numpy as np
from scipy import ndimage
import Convert_RGB_to_HSI as converter
import RegionGrowing

# Read Image
meat_test = cv2.imread('meat_1.jpg')
height, width = meat_test.shape[:2]
center =[int(height/2), int(width/2)]
print(center[1])
# Swap red and blue channels and convert to HSI
# This helps to remove blue backgrounds using image thresholding in Hue channel of HSI
# meat_test = cv2.cvtColor(meat_test, cv2.COLOR_BGR2RGB)
meat_hsi = converter.RGB_TO_HSI(meat_test)
meat_gray = cv2.cvtColor(meat_test, cv2.COLOR_BGR2GRAY)
# Otsu's thresholing - one of the most accurate and widely use methods for image segmentation
ret, meat_otsu = cv2.threshold(meat_gray, 0, 255, cv2.THRESH_OTSU)

# fill holes in the binary image
meat_fill = ndimage.morphology.binary_fill_holes(meat_otsu).astype(float)

# apply region growing algorithms
meat_region = RegionGrowing.region_growing(meat_fill, center)
cv2.imshow('meat image', meat_region)
cv2.waitKey(0)