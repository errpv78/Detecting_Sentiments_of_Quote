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
Secured 84th position out of 5000+ participants<br>
Test Performance: 44.47199


# OCR:
pytesseract-ocr.py: ocr using pytesseract. Change the img_path variable for performing ocr on other images with suitable img_path.<br>
EAST_text_detection.py: using pretrained east model for text detection. Model not included due to large size, can download the model
from: https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV <br>


# Sentimental Analysis
Simple_nn: Sentimental analysis using simple neural network, dataset used was imdb movie reviews dataset. Trained model also available in that directory. Performance - 47.86517<br><br>

LSTM: Lstm model for sentimental analysis, dataset used was:  https://www.kaggle.com/kazanova/sentiment140 Performance - 46.33494<br><br>

Bert: Using pretrained bert model with previous dataset. Performance - 48.10594<br><br>

# Refrences:
A big thanks to Andrian from https://www.pyimagesearch.com/start-here/ for OCR basics and comprehensive implementation of EAST detector.<br>
For Lstm model for sentimental analysis: https://blog.usejournal.com/sentiment-classification-with-natural-language-processing-on-lstm-4dc0497c1f19  ; https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras<br>
For bert model: https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/ <br>


