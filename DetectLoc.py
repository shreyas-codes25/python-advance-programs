# import the necessary packages
import numpy as np
import imutils
import cv2

# load the image
image = cv2.imread('sample.jpg', 1)

# convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur the image
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# threshold the image to reveal light regions in the blurred image
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

# perform a connected component analysis on the thresholded image
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)

# Initialize a mask to store only the "large" components
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in range(1, num_labels):
    # construct the label mask and count the number of pixels
    labelMask = (labels == label).astype("uint8") * 255
    numPixels = cv2.countNonZero(labelMask)

    # if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
    if numPixels > 300:
        mask = cv2.add(mask, labelMask)

# find the contours in the mask, then sort them from left to right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Initialize lists to store centroid coordinates and area
centroid_list = []
area_list = []

# Loop over the contours
for i, c in enumerate(cnts):
    # Calculate the area of the contour
    area = cv2.contourArea(c)

    # Find the centroid of the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Draw the bright spot on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)

    # Append centroid coordinates and area to the respective lists
    centroid_list.append((cx, cy))
    area_list.append(area)

# Save the output image as a PNG file
cv2.imwrite("led_detection_results.png", image)

# Open a text file for writing
with open("led_detection_results.txt", "w") as file:
    # Write the number of LEDs detected to the file
    num_leds = len(centroid_list)
    file.write(f"No. of LEDs detected: {num_leds}\n")

    # Loop over the centroids and areas
    for i, (centroid, area) in enumerate(zip(centroid_list, area_list)):
        file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")

# Close the text file
file.close()
