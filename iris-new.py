import cv2
import numpy as np
import os

# List files in the directory and prompt user for input
directory = os.listdir(os.getcwd() + '/001/L/')
num_of_files = len(directory)

# Display all files in the directory
for file in directory:
    print(file)

# Ask user for the filename with extension
file_name = input("Enter filename with extension: ")
path = os.path.join(os.getcwd(), '001', 'L', file_name)

# Read the image and make a copy for output
image_read = cv2.imread(path)
output = image_read.copy()

# Step 1: Preprocessing - Canny Edge Detection on the image
image_test = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test = cv2.GaussianBlur(image_test, (7, 7), 1)
image_test = cv2.Canny(image_test, 20, 70, apertureSize=3)
cv2.imshow("Canny Edge Detection (Outer)", image_test)

# Step 2: Hough Circle Transform (for outer circle detection)
hough_circle = cv2.HoughCircles(image_test, cv2.HOUGH_GRADIENT, 1.3, 800)
if hough_circle is not None:
    hough_circle = np.round(hough_circle[0, :]).astype("int")
    for (x, y, radius) in hough_circle:
        cv2.circle(output, (x, y), radius, (255, 0, 0), 4)  # Draw detected circle

# Step 3: Canny Edge Detection with different thresholds
image_test1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test1 = cv2.GaussianBlur(image_test1, (7, 7), 1)
image_test1 = cv2.Canny(image_test1, 100, 120, apertureSize=3)
cv2.imshow("Canny Edge Detection (Inner)", image_test1)

# Step 4: Hough Circle Transform (for inner circle detection)
circles = cv2.HoughCircles(image_test1, cv2.HOUGH_GRADIENT, 1, 800,
                            param1=50, param2=20, minRadius=0, maxRadius=60)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Draw detected circle
        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)  # Mark center

# Display the output with circles detected
cv2.imshow('Detected Circles', output)
cv2.waitKey(0)

# Step 5: Load and process another image for comparison (using SIFT)
file_name = '1.jpg'  # Original image file
path = os.path.join(os.getcwd(), '001', 'L', file_name)

image_read = cv2.imread(path)
output = image_read.copy()

# Step 6: Preprocessing the image with Canny Edge Detection (same steps as before)
image_test2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test2 = cv2.GaussianBlur(image_test2, (7, 7), 1)
image_test2 = cv2.Canny(image_test2, 20, 70, apertureSize=3)

# Step 7: Hough Circle Transform (for outer circle detection)
hough_circle = cv2.HoughCircles(image_test2, cv2.HOUGH_GRADIENT, 1.3, 800)
if hough_circle is not None:
    hough_circle = np.round(hough_circle[0, :]).astype("int")
    for (x, y, radius) in hough_circle:
        cv2.circle(output, (x, y), radius, (255, 0, 0), 4)

# Step 8: Canny Edge Detection with different thresholds (for inner detection)
image_test3 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test3 = cv2.GaussianBlur(image_test3, (7, 7), 1)
image_test3 = cv2.Canny(image_test3, 100, 120, apertureSize=3)

# Step 9: Hough Circle Transform (for inner circle detection)
circles = cv2.HoughCircles(image_test3, cv2.HOUGH_GRADIENT, 1, 800,
                            param1=50, param2=20, minRadius=0, maxRadius=60)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

# Step 10: Feature matching using SIFT
original = image_test2
image_to_compare = image_test

# Initialize SIFT detector
sift = cv2.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

# FLANN based matcher parameters
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Find matches between descriptors
matches = flann.knnMatch(desc_1, desc_2, k=2)

# Filter good matches based on distance ratio
good_points = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_points.append(m)

# Calculate match percentage
number_keypoints = min(len(kp_1), len(kp_2))  # Determine the number of keypoints to consider
print("Keypoints in 1st Image:", len(kp_1))
print("Keypoints in 2nd Image:", len(kp_2))
print("Good Matches:", len(good_points))

match_percentage = len(good_points) / number_keypoints * 100
print("Match Percentage:", match_percentage)

# Define threshold for granting access
threshold = 90
if match_percentage >= threshold:
    print("Access Granted")
else:
    print("Access Denied")

# Draw and display the matches
result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
cv2.imshow("Feature Matching Result", result)
cv2.imwrite("feature_matching_result.jpg", result)

# Display the original and test images for comparison
cv2.imshow("Original Image", original)
cv2.imshow("Test Image", image_to_compare)
cv2.waitKey(0)
cv2.destroyAllWindows()
#completed
#review done for final year