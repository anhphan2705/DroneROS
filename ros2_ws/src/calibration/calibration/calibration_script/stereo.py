import cv2 as cv
import numpy as np

def load_calibration_params(file_path):
    """Load calibration parameters from a YAML file."""
    cv_file = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    
    # Load rectification maps and Q matrix
    stereo_map_left_x = cv_file.getNode('stereo_map_left_x').mat()
    stereo_map_left_y = cv_file.getNode('stereo_map_left_y').mat()
    stereo_map_right_x = cv_file.getNode('stereo_map_right_x').mat()
    stereo_map_right_y = cv_file.getNode('stereo_map_right_y').mat()
    translation_vector = cv_file.getNode('translation_vector').mat()
    Q = cv_file.getNode('disparity_to_depth_map').mat()
    
    cv_file.release()
    return (stereo_map_left_x, stereo_map_left_y, stereo_map_right_x, stereo_map_right_y, translation_vector, Q)

# def compute_disparity_map(rectified_left, rectified_right):
#     """Compute the disparity map using StereoBM."""
#     # Adjust parameters for better disparity map
#     n_disp_factor = 14
#     stereo_bm = cv.StereoBM.create(numDisparities=16 * n_disp_factor, blockSize=21)
#     disparity = stereo_bm.compute(rectified_left, rectified_right)
#     return disparity

def compute_disparity_map(rectified_left, rectified_right):
    """Compute the disparity map using StereoSGBM."""
    min_disparity = 0
    num_disparities_factor = 14
    num_disparities = 16 * num_disparities_factor  # Must be divisible by 16
    block_size = 9 # Block size for matching, less = more noise, more = smooth
    p1 = 8 * 3 * block_size**2  # Smoothness parameter 1 # Do not change
    p2 = 32 * 3 * block_size**2  # Smoothness parameter 2 # Do not change
    disp12_max_diff = 1  # Max allowed difference between left and right disparity checks
    uniqueness_ratio = 10  # Margin for the best match
    speckle_window_size = 100  # Max size of small regions considered noise
    speckle_range = 32  # Max disparity variation within a connected component
    preFilterCap = 63  # Truncation value for the prefiltered image pixels
    mode = cv.STEREO_SGBM_MODE_SGBM_3WAY  # Mode for computing disparity

    stereo_sgbm = cv.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=preFilterCap,
        mode=mode
    )

    # Compute disparity map
    disparity = stereo_sgbm.compute(rectified_left, rectified_right)
    return disparity

def normalize_disparity_map(disparity):
    """Normalize disparity map for visualization."""
    disparity_visual = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    return disparity_visual

def overlay_rectified_images(rectified_left, rectified_right, alpha=0.5):
    """Overlay two rectified images for better visualization."""
    # Ensure images are in the same shape
    overlay = cv.addWeighted(rectified_left, alpha, rectified_right, 1 - alpha, 0)
    return overlay

def main():
    # Paths to images and calibration parameters
    left_image_path = "images/test/imageL0000.png"
    right_image_path = "images/test/imageR0000.png"
    calibration_file = "calibrated_params/2024-12-30_18-37-41/stereo_calibration_params.yml"

    # Load calibration parameters
    stereo_map_left_x, stereo_map_left_y, stereo_map_right_x, stereo_map_right_y, translation_vector, Q = load_calibration_params(calibration_file)

    # Read stereo images
    img_left = cv.imread(left_image_path, cv.IMREAD_GRAYSCALE)
    img_right = cv.imread(right_image_path, cv.IMREAD_GRAYSCALE)

    if img_left is None or img_right is None:
        print("Error: Unable to load one or both images.")
        return

    # Rectify images
    rectified_left = cv.remap(img_left, stereo_map_left_x, stereo_map_left_y, cv.INTER_LINEAR)
    rectified_right = cv.remap(img_right, stereo_map_right_x, stereo_map_right_y, cv.INTER_LINEAR)

    # Overlay rectified images
    overlay_image = overlay_rectified_images(rectified_left, rectified_right, alpha=0.5)

    # Compute disparity map
    disparity = compute_disparity_map(rectified_left, rectified_right)

    # Normalize disparity map for visualization
    disparity_visual = normalize_disparity_map(disparity)

    # Display results
    cv.imshow("Rectified Left Image", rectified_left)
    cv.imshow("Rectified Right Image", rectified_right)
    cv.imshow("Overlayed Rectified Images", overlay_image)
    cv.imshow("Disparity Map", disparity_visual)
    
    # Compute the baseline distance
    baseline_distance = np.linalg.norm(translation_vector)
    print(f"Baseline distance between the cameras: {baseline_distance:.2f} mm")
    
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Exit on 'q'
        cv.destroyAllWindows()

    cv.waitKey(0)
    cv.destroyAllWindows()

    # # Save disparity and depth map
    # cv.imwrite("overlayed_rectified_images.png", overlay_image)
    # cv.imwrite("disparity_map.png", disparity_visual)
    # print("Overlayed rectified images saved as overlayed_rectified_images.png")
    # print("Disparity map saved as disparity_map.png")

if __name__ == "__main__":
    main()