import cv2
import numpy as np
import sys

# Check arguments
# Read sample.png if no arguments given
if(len(sys.argv) < 2):
	filename = "input_image.png"
else:
	filename = sys.argv[1]

#print(filename)

def translation(img):
    # translation of the image
    num_rows, num_cols = img.shape[:2]

    # change last column (5 and 10) values for experiments

    trans_matrix = np.float32([ [1, 0, 30], [0, 1, 40]])
    img_translation = cv2.warpAffine(img, trans_matrix, (num_cols, num_rows))
    return img_translation

def rotation(img, angle):
    # variable (angle) degree rotation
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def resize(img):
    # 50 % resize
    resized = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    return resized

# Read image
img = cv2.imread(filename)
#img = resize(img)
img = rotation(img,45)

#img = translation(img)
# img = rotation(img)

# Convert to gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#ret, thresholded = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate Hu Moments
hu_moments = cv2.HuMoments(cv2.moments(gray)).flatten()

# log scale hu moments
log_scaled_hu_moments = (-np.sign(hu_moments) * np.log10(np.abs(hu_moments)))

print("Hu Moments: ", hu_moments)
print("Logarithmic Scaled Hu Moments: ", log_scaled_hu_moments)
