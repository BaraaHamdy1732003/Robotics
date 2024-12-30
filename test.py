import cv2
import numpy as np

# Load the image
image = cv2.imread('photo_2024-12-27_14-45-02.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask

_,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Closing gaps
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)  # Remove small noise

# Find contours from the cleaned binary image
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to track objects
areas = []

for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)

    # Filter out small contours based on area and dimensions
    if area > 100 and w > 10 and h > 10:
        M = cv2.moments(contour)

        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = 0, 0

        areas.append((area, (cX, cY), contour))

# Sort areas to identify the largest, smallest, and middle
areas.sort(reverse=True, key=lambda x: x[0])

# Label largest, smallest, and middle object
if len(areas) > 0:
    max_area, max_center, max_contour = areas[0]
    cv2.putText(image, 'Largest', max_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)

if len(areas) > 1:
    min_area, min_center, min_contour = areas[-1]
    cv2.putText(image, 'Smallest', min_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.drawContours(image, [min_contour], -1, (0, 255, 0), 2)

if len(areas) > 2:
    mid_index = len(areas) // 2
    mid_area, mid_center, mid_contour = areas[mid_index]
    cv2.putText(image, 'Middle', mid_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.drawContours(image, [mid_contour], -1, (0, 255, 0), 2)

# Annotate the total number of valid objects
num_objects = len(areas)
cv2.putText(image, f'Total Objects: {num_objects}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display and save the processed image
cv2.imshow('Processed Image', thresh)
cv2.imwrite('processed_image_fixed.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
