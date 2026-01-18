# Day 2: Homography Estimation & Image Alignment

## 🚀 Objective
Implement a computer vision pipeline to align two images of the same planar surface taken from different angles. This involves feature detection, matching, and calculating the **Homography Matrix** ($3\times3$) using robust statistical methods (RANSAC).

## 🛠️ Tech Stack
- **Python 3.9+**
- **OpenCV (cv2):** For image processing and matrix operations.
- **NumPy:** For linear algebra optimizations.
- **Matplotlib:** For visualization.

## 📝 Exercises & Implementation Steps

### 1. Data Acquisition & Preprocessing
- [x] Capture two images: a `reference` (top-down view) and a `target` (angled/perspective view).
- [x] Convert images to grayscale to reduce dimensionality (3 channels -> 1 channel).

### 2. Feature Detection (ORB)
- [x] Initialize the ORB (Oriented FAST and Rotated BRIEF) detector.
- [x] Detect keypoints (corners/edges) and compute descriptors (binary feature vectors) for both images.
- [x] *Why ORB?* It is rotation invariant and faster/patent-free compared to SIFT/SURF.

### 3. Feature Matching (Hamming Distance)
- [x] Use `BFMatcher` (Brute-Force) with Hamming distance (optimal for binary descriptors).
- [x] Filter matches: Sort by distance and keep only the top N best matches to reduce noise.

### 4. Homography Computation (RANSAC)
- [x] Extract coordinates of matched points.
- [x] Compute the Homography Matrix $H$ using `cv2.findHomography`.
- [x] **Critical Step:** Apply **RANSAC** (Random Sample Consensus) to ignore outliers (false matches) and solve the overdetermined system robustly.

### 5. Perspective Warping
- [x] Apply the calculated matrix $H$ to the `target` image using `cv2.warpPerspective`.
- [x] Visualize the alignment: Overlay the transformed image onto the reference.