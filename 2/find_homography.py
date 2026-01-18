import numpy as np # For numerical operations
import os # For file path operations
import cv2 # For computer vision tasks
import matplotlib.pyplot as plt # For plotting images


def load_image(image_path:str) -> np.ndarray:
    """Load an image from a file path.

    Args:
        image_path (str): Path to the image file."""
    # Load the image in grayscale mode
    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"image not found")
    return image


orb = cv2.ORB_create(5000) # Initialize ORB detector

img_target = load_image(os.path.join("data", "image_target.jpg"))
img_ref = load_image(os.path.join("data", "image_ref.jpg"))
kp1, des1 = orb.detectAndCompute(img_target, None)
kp2, des2 = orb.detectAndCompute(img_ref, None)

# 3 Feature Matching
# Using Brute-Force Matcher with Hamming distance
# crossCheck=True ensures that the matches are mutual
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
# Sort matches based on distance (lower distance is better)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:50] # Select top 50 matches

# Visualize Matches
img_matches = cv2.drawMatches(img_target, kp1, img_ref, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 4 Extract location of good matches
points_target = np.zeros((len(good_matches), 2), dtype=np.float32)
points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)

for i, match in enumerate(good_matches):
    points_target[i, :] = kp1[match.queryIdx].pt
    points_ref[i, :] = kp2[match.trainIdx].pt

# 5 Compute Homography using RANSAC
h_matrix, mask = cv2.findHomography(points_target, points_ref, cv2.RANSAC)

# 6 Visualize the detected object in the scene
h, w = img_ref.shape
aligned_img = cv2.warpPerspective(img_target, h_matrix, (w, h))


# Plotting the results
plt.figure(figsize=(20,10))

plt.subplot(1, 3, 1)
plt.title("Matched Keypoints")
plt.imshow(img_matches)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title("Aligned Image")
plt.imshow(aligned_img, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title("Reference Image")
plt.imshow(img_ref, cmap='gray')
plt.axis('off')
plt.show()
