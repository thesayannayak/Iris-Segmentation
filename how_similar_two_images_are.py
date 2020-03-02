import cv2
import numpy as np
import os

# Function to get image path from user input
def get_image_path(directory):
    while True:
        file_name = input(f"Please enter a filename from the directory {directory}: ")
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            return file_path
        else:
            print(f"File '{file_name}' does not exist. Please try again.")

# Function to compare images
def compare_images(original, image_to_compare):
    # 1) Check if 2 images are equal
    if original.shape == image_to_compare.shape:
        print("The images have the same size and channels.")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely equal.")
        else:
            print("The images are NOT equal.")
    else:
        print("The images have different dimensions or color channels.")

    # 2) Check for similarities between the 2 images using SIFT
    sift = cv2.SIFT_create()  # Use SIFT for feature extraction
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    # FLANN based matcher parameters
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform knnMatch
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    # Apply Lowe's ratio test to find good matches
    good_points = [m for m, n in matches if m.distance < 0.6 * n.distance]

    # Print statistics about keypoints and matches
    number_keypoints = min(len(kp_1), len(kp_2))
    print(f"Keypoints in 1st image: {len(kp_1)}")
    print(f"Keypoints in 2nd image: {len(kp_2)}")
    print(f"Good matches: {len(good_points)}")
    print(f"Match quality: {len(good_points) / number_keypoints * 100:.2f}%")

    # Visualize the good matches
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
    cv2.imshow("Feature Matching", cv2.resize(result, None, fx=0.4, fy=0.4))
    cv2.imwrite("feature_matching_result.jpg", result)

# Main logic
def main():
    directory = os.path.join(os.getcwd(), '001', 'L')
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    print("List of files in the directory:")
    directory_files = os.listdir(directory)
    for file in directory_files:
        print(file)

    # Get the first image to compare
    original_path = get_image_path(directory)
    original = cv2.imread(original_path)

    if original is None:
        print(f"Failed to load the image {original_path}.")
        return

    # Get the second image to compare
    image_to_compare_path = get_image_path(directory)
    image_to_compare = cv2.imread(image_to_compare_path)

    if image_to_compare is None:
        print(f"Failed to load the image {image_to_compare_path}.")
        return

    # Perform comparison
    compare_images(original, image_to_compare)

    # Display images
    cv2.imshow("Original Image", cv2.resize(original, None, fx=0.4, fy=0.4))
    cv2.imshow("Image to Compare", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    #completed
    #review done for final year
