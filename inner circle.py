import cv2
import numpy as np
import os

# Function to get valid image path from user input
def get_image_path(directory):
    while True:
        file_name = input(f"Please enter a filename from the directory {directory}: ")
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            return file_path
        else:
            print(f"File '{file_name}' does not exist. Please try again.")

# Function to detect circles in the image using HoughCircles
def detect_circles(image):
    # Convert image to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1)
    
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 80, 100, apertureSize=3)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=50, param2=30, minRadius=0, maxRadius=0)
    
    # If no circles are detected, return None
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    else:
        return None

# Function to draw detected circles on the image
def draw_circles(image, circles):
    output_image = image.copy()  # Copy of the original image to draw on
    for (x, y, r) in circles:
        # Draw the circle in the output image, then draw a rectangle at the center
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(output_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return output_image

# Main function to process the image
def main():
    directory = os.path.join(os.getcwd(), '001', 'L')
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    print("List of files in the directory:")
    files = os.listdir(directory)
    for file in files:
        print(file)

    # Get image path from user input
    image_path = get_image_path(directory)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load the image {image_path}.")
        return
    
    # Detect circles in the image
    circles = detect_circles(image)
    
    if circles is not None:
        # Draw circles on the image
        output_image = draw_circles(image, circles)
        
        # Show the results
        cv2.imshow('Detected Circles', output_image)
    else:
        print("No circles were detected.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#completed
#review done for final year