# Detecting_Sentiments_of_Quote
Project for Hackerearth Machine Learning Challenge: Love is Love :https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-pride-month-edition/problems/<br>

### Problem Statement:
Task<br>
Your task is to build an engine that combines the concepts of OCR and NLP that accepts a .jpg file as input, extracts the text, if any, and classifies sentiment as positive or negative. If the text sentiment is neutral or an image file does not have any text, then it is classified as random.

### Project Submission: Submission1.zip
(Contains files used for submission)<br>
Contains: Approach.txt (approach used for problem), Bert-model (for sentimental analysis), Predict.ipynb(for using model and ocr for making predictions),
Test.csv(final predictions file)<br>


### Hackathon result
Registered late in the competition(just 4 days before the end)<br>
Secured 84th position out of 5000+ participants


# OCR:
pytesseract-ocr.py: ocr using pytesseract. Change the img_path variable for performing ocr on other images with suitable img_path.<br>
EAST_text_detection.py: using pretrained east model for text detection. Model not included due to large size, can download the model
from: https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV <br>



# Refrences:
A big thanks to Andrian from https://www.pyimagesearch.com/start-here/ for OCR basics and comprehensive implementation of EAST detector.<br>

