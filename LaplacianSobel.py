import cv2
import numpy as np
import os

# List files in the directory and prompt user for input
directory = os.listdir(os.getcwd() + '/001/L/')
num_of_file = len(directory)

# Display all files in the directory
for file in directory:
    print(file)

# Ask user for the file name with extension
file_name = input("Enter filename with extension: ")
path = os.path.join(os.getcwd(), '001', 'L', file_name)

# Read the image from the specified path
image_read = cv2.imread(path)
output = image_read.copy()

# Step 1: Prepare image for edge detection
image_test = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test = cv2.GaussianBlur(image_test, (7, 7), 1)

# Apply Sobel filter for edge detection (X and Y gradients)
image_test_sobel = cv2.Sobel(image_test, cv2.CV_8U, 1, 1, ksize=5)
cv2.imshow("Sobel (X and Y gradients)", image_test_sobel)

# Step 2: Apply Hough Circle Transform to detect circles
hough_circle = cv2.HoughCircles(image_test_sobel, cv2.HOUGH_GRADIENT, 1.3, 800)
if hough_circle is not None:
    hough_circle = np.round(hough_circle[0, :]).astype("int")
    for (x, y, radius) in hough_circle:
        cv2.circle(output, (x, y), radius, (255, 0, 0), 4)  # Draw circle

# Step 3: Repeat similar process for another Sobel filter (X gradient only)
image_test = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test = cv2.GaussianBlur(image_test, (7, 7), 1)

# Apply Sobel filter (X gradient only)
image_test_sobel_x = cv2.Sobel(image_test, cv2.CV_8U, 1, 0, ksize=5)
cv2.imshow("Sobel (X gradient)", image_test_sobel_x)

# Step 4: Detect circles again using Hough Transform on the new Sobel image
circles = cv2.HoughCircles(image_test_sobel_x, cv2.HOUGH_GRADIENT, 1, 800,
                            param1=50, param2=20, minRadius=0, maxRadius=60)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Draw circle
        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)  # Draw center point

# Display the result with detected circles
cv2.imshow('Detected Circles', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#completed
#chatgpt update
