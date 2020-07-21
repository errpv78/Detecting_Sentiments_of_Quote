from PIL import Image
import pytesseract
import cv2
import os
import imutils
import numpy as np

# Reading and grayscale image
img_path = '../Data Files/Sample Data Files/Sample_Negative.jpg'
preprocess_type = 'thresh' # Other option: blur
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshing:
"""The function cv.threshold is used to apply thresholding.
 The first argument is the source image, which should be a 
 grayscale image. The second argument is threshold value
 which is used to classify the pixel values. The third 
 argument is the maximum value which is assigned to pixel
 values exceeding the threshold. OpenCV provides different
 types of thresholding which is given by fourth parameter
 of the function.
 The method returns two outputs. The first is threshold 
 that was used and the second output is the thresholded 
 image.
"""
# Applying different preprocessing on images
gray_t = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Threshing
# cv2.THRESH_BINARY:
"""THRESH_BINARY :   dst(x,y)=  { maxval   if src(x,y)>thresh
                                  0         otherwise }"""
# cv2.THESH_OTSU
"""In global thresholding, we used an arbitrary chosen value
 as a threshold. In contrast, Otsu's method avoids having to 
 choose a value and determines it automatically.
Consider an image with only two distinct image values 
(bimodal image), where the histogram would only consist of 
two peaks. A good threshold would be in the middle of those 
two values. Similarly, Otsu's method determines an optimal 
global threshold value from the image histogram.
In order to do so, the cv.threshold() function is used, where
 cv.THRESH_OTSU is passed as an extra flag. The threshold 
 value can be chosen arbitrary. The algorithm then finds 
 optimal threshold value which is returned as first output."""

gray_b = cv2.medianBlur(gray, 3) # Blurring
gray_t_b = cv2.medianBlur(gray_t, 3) # Threshing then Blurring
gray_b_t = cv2.threshold(gray_b, 0, 255,  # Blurring then Threshing
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# Filter output of pytesseract
def filter_text(s):
    return ' '.join(s.split())

# Applying ocr on images preprocessed by threshing
filename = "Gray_img_dir/{}.png".format(os.getpid())
cv2.imwrite(filename, gray_t)
text_t = pytesseract.image_to_string(Image.open(filename))
text_t = filter_text(text_t)
os.remove(filename)

# Applying ocr on images preprocessed by blurring
filename = "Gray_img_dir/{}.png".format(os.getpid())
cv2.imwrite(filename, gray_b)
text_b = pytesseract.image_to_string(Image.open(filename))
text_b = filter_text(text_b)
os.remove(filename)

# Applying ocr on images preprocessed by threshing and then blurring
filename = "Gray_img_dir/{}.png".format(os.getpid())
cv2.imwrite(filename, gray_t_b)
text_t_b = pytesseract.image_to_string(Image.open(filename))
text_t_b = filter_text(text_t_b)
os.remove(filename)

# Applying ocr on images preprocessed by blurring and then threshing
filename = "Gray_img_dir/{}.png".format(os.getpid())
cv2.imwrite(filename, gray_b_t)
text_b_t = pytesseract.image_to_string(Image.open(filename))
text_b_t = filter_text(text_b_t)
os.remove(filename)

# Adding text of blurring and threshing
text_combined = text_t + ' ' + text_b

# Actual Image
image = imutils.resize(image, width=800)
cv2.imshow("Original", image)
cv2.waitKey(0)

# Performance Evaluation
print('Result of Blurring: ', text_b)
print('Result of Threshing: ', text_t)
print('Result of Threshing and then Blurring: ', text_t_b)
print('Result of Blurring and then Threshing: ', text_b_t)
print('Combined result of Blurring and Threshing: ', text_combined)

# Result on some images blurring better, on rest threshing, both together not so effective, combined provided good results.