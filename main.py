import cv2
import os

# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('data/sivakumar.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Save the output image
cv2.imwrite('output/detected_faces.jpg', img)

print("Face detection completed. Check the 'output' folder.")
