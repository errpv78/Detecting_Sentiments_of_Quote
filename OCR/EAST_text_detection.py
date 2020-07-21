from imutils.object_detection import non_max_suppression
import numpy as np # Numpy is a dependency for OpenCV
import time
import cv2
import pytesseract
import imutils

# EAST: An Efficient and Accurate Scene Text Detector
"""OpenCV’s EAST text detector is a deep learning model,
 based on a novel architecture and training pattern. It 
 is capable of (1) running at near real-time at 13 FPS on
 720p images and (2) obtains state-of-the-art text 
 detection accuracy."""


def decode_predictions(scores, geometry):

	# Getting num of rows and columns from scores matrix
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# Looping over number of rows
	for y in range(0, numRows):

		# Extracting scores (probabilities) and geometrical data
		# to derive potential bounding box coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# Loop over the number of columns
		for x in range(0, numCols):

			# Ignoring less probable scores
			if scoresData[x] < min_conf:
				continue

			# Computing offset factor as resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# Extracting the rotation angle for the prediction and
			# computing the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# Using geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# Compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# Adding the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# Returning tuple of bounding boxes and associated confidences
	return (rects, confidences)

# Loading Image and initializing variables
img_path = '../Data Files/Sample Data Files/Sample_Negative.jpg'
east = 'frozen_east_text_detection.pb' # The EAST text detector.
min_conf = 0.5
res_width = 320
res_height = 320
padding = 0 # Amount of padding to add to each border of ROI


# The EAST text requires input image dimensions to be
# multiples of 32, therefore resizing
image = cv2.imread(img_path)
orig = image.copy()
(origH, origW) = image.shape[:2]
(newW, newH) = (res_width, res_height)
rW = origW / float(newW)
rH = origH / float(newH)

# Resizing the image, ignoring the aspect ratio
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]


# Output layers the EAST detector model to use for OCR
# "feature_fusion/Conv_7/Sigmoid":
"Output sigmoid activation that gives probability of \
region containing text or not."
# "feature_fusion/concat_3":
"Output feature map that represents 'geometry' of image \
to derive bounding box coordinates of text in input."
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]


# Loading the pre-trained EAST text detector model
print("Loading EAST text detector model...")
east_net = cv2.dnn.readNet(east)


# Constructing blob from the input image
# Blob:
"""Binary Large OBject (BLOB) is a collection of binary
 data stored as a single entity in a database management
 system. Blobs are typically images, audio or other 
 multimedia objects, though sometimes binary executable 
 code is stored as a blob."""
# cv2.dnn.blobFromImage:
"""This openCV function helps to facilitate image 
preprocessing for deep learning classification:
This function performs:
1. Mean Subtraction
2. Scaling
3. And optionally channel swapping
It creates 4-dimensional blob from image. Optionally 
resizes and crops image from center, subtract mean 
values, scales values by scalefactor, swap Blue and Red 
channels."""
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)


start = time.time()
east_net.setInput(blob) # Setting image blob as eas_net input
# Calling east_net.forward and passing layer names as input to return
# 2 feature maps
(scores, geometry) = east_net.forward(layerNames)
# scores map, contain probability of given region containing text
# geometry map to derive bounding box coordinates of text in image
end = time.time()
# Text detection time
print("Text detection took {:.6f} seconds".format(end - start))


# Decode the predictions
(rects, confidences) = decode_predictions(scores, geometry)
# rects is predicted bounding box coordinates, confidences is
# probability scores


# Applying non-maxima suppression to suppress weak, overlapping
# bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)


# Initializing the list of results
results = []

# Looping over the bounding boxes
for (startX, startY, endX, endY) in boxes:

	# Scaling the bounding box coordinates based on the respective ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# Applying padding to bounding boxes
	dX = int((endX - startX) * padding)
	dY = int((endY - startY) * padding)
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# Extracting actual padded ROI
	roi = orig[startY:endY, startX:endX]

	# In order to apply Tesseract v4 to OCR text we must supply
	# (1) a language,
	# (2) an OEM flag of 4, indicating that the we
	# wish to use the LSTM neural net model for OCR, and finally
	# (3) an OEM value, in this case, 7 which implies that we are
	# treating the ROI as a single line of text
	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)

	# Adding bounding box coordinates and OCR'd text to list of results
	results.append(((startX, startY, endX, endY), text))

# Sorting resulting bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

res = orig.copy()

# Looping over the results
for ((startX, startY, endX, endY), text) in results:
	# Displaying the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))

	# Stripping out non-ASCII text so we can draw text on image
	# using OpenCV, then drawing text and a bounding box surrounding
	# text region of input image
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = orig.copy()

	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
	cv2.rectangle(res, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(res, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)

res = imutils.resize(res, width=1500)

# Final result display:
cv2.imshow("Text Detection", res)
cv2.waitKey(0)

# Evaluation:
"""Good performance but difficulty in aligning text as bounding boxes sorted top to bottom
and left-right words are merged. So better performance by pytesseract ocr."""