import glob
import datetime
import os
import numpy as np
import cv2 as cv

class StereoCalibrator:
    """
    A class for calibrating a stereo camera setup using chessboard images.
    """
    def __init__(self, left_images_dir, right_images_dir, chessboard_size=(10, 7),
                 square_size=25, show_corners=True):
        """
        Initialize the calibrator with directories and calibration parameters.
        
        Args:
            left_images_dir (str): Directory containing left camera images.
            right_images_dir (str): Directory containing right camera images.
            chessboard_size (tuple): Number of inner corners per chessboard row and column.
            square_size (float): Size of a chessboard square (in mm).
            show_corners (bool): Whether to display the detected corners.
        """
        self.left_images_dir = left_images_dir
        self.right_images_dir = right_images_dir
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.show_corners = show_corners

        # Termination criteria for corner refinement
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points template e.g. (0,0,0), (1,0,0), ...,(9,6,0)
        self.object_points_template = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.object_points_template[:, :2] = np.mgrid[0:chessboard_size[0],
                                                      0:chessboard_size[1]].T.reshape(-1, 2)
        self.object_points_template *= square_size

        # Lists for storing points from all images
        self.object_points = []    # Real world 3D points
        self.image_points_left = []  # 2D points from left images
        self.image_points_right = []  # 2D points from right images

        # Image paths and frame size
        self.images_left = []
        self.images_right = []
        self.frame_size = None

        # Calibration parameters (to be computed)
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.rvecs_left = None
        self.tvecs_left = None
        self.rvecs_right = None
        self.tvecs_right = None

        # Stereo calibration results
        self.rotation_matrix = None
        self.translation_vector = None
        self.essential_matrix = None
        self.fundamental_matrix = None
        self.ret_stereo = None

        # Rectification results
        self.rectification_matrix_left = None
        self.rectification_matrix_right = None
        self.projection_matrix_left = None
        self.projection_matrix_right = None
        self.disparity_to_depth_map = None
        self.stereo_map_left = None
        self.stereo_map_right = None

    def load_images(self):
        """
        Load and sort image file paths from the specified directories.
        Also determines the frame size from the first left image.
        """
        self.images_left = sorted(glob.glob(os.path.join(self.left_images_dir, '*.png')))
        self.images_right = sorted(glob.glob(os.path.join(self.right_images_dir, '*.png')))

        if not self.images_left or not self.images_right:
            raise FileNotFoundError("Error: No images found in the specified directories!")
        if len(self.images_left) != len(self.images_right):
            raise ValueError("Error: The number of left and right images do not match!")

        # Determine frame size from the first image
        sample_image = cv.imread(self.images_left[0])
        if sample_image is None:
            raise IOError("Error: Unable to read sample image to determine frame size!")
        height, width, _ = sample_image.shape
        self.frame_size = (width, height)
        print(f"Detected frame size: {self.frame_size}")

    def detect_chessboard_corners(self):
        """
        Detect chessboard corners in each image pair. Refine corner locations,
        add corresponding object points and image points, and optionally display them.
        """
        for img_left_path, img_right_path in zip(self.images_left, self.images_right):
            img_left = cv.imread(img_left_path)
            img_right = cv.imread(img_right_path)
            gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
            gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

            # Find the chessboard corners in both images
            ret_left, corners_left = cv.findChessboardCorners(gray_left, self.chessboard_size, None)
            ret_right, corners_right = cv.findChessboardCorners(gray_right, self.chessboard_size, None)

            if ret_left and ret_right:
                print(f"Chessboard detected for images: {img_left_path} and {img_right_path}")
                self.object_points.append(self.object_points_template)

                # Refine corner positions
                corners_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
                self.image_points_left.append(corners_left)

                corners_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
                self.image_points_right.append(corners_right)

                # Optionally display detected corners
                if self.show_corners:
                    cv.drawChessboardCorners(img_left, self.chessboard_size, corners_left, ret_left)
                    cv.imshow('Left Image', img_left)
                    cv.drawChessboardCorners(img_right, self.chessboard_size, corners_right, ret_right)
                    cv.imshow('Right Image', img_right)
                    cv.waitKey(1000)
            else:
                print(f"Failed to detect chessboard for images: {img_left_path} and {img_right_path}")
        cv.destroyAllWindows()

    def calibrate_cameras(self):
        """
        Calibrate each camera individually using the detected chessboard corners.
        Computes and prints reprojection errors and replaces intrinsic matrices
        with their optimal versions.
        """
        # Calibrate left camera
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, self.rvecs_left, self.tvecs_left = cv.calibrateCamera(
            self.object_points, self.image_points_left, self.frame_size, None, None
        )
        # Get an optimal new camera matrix for left camera
        height_left, width_left, _ = cv.imread(self.images_left[-1]).shape
        optimal_camera_matrix_left, _ = cv.getOptimalNewCameraMatrix(
            self.camera_matrix_left, self.dist_coeffs_left, (width_left, height_left), 1, (width_left, height_left)
        )

        # Calibrate right camera
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, self.rvecs_right, self.tvecs_right = cv.calibrateCamera(
            self.object_points, self.image_points_right, self.frame_size, None, None
        )
        height_right, width_right, _ = cv.imread(self.images_right[-1]).shape
        optimal_camera_matrix_right, _ = cv.getOptimalNewCameraMatrix(
            self.camera_matrix_right, self.dist_coeffs_right, (width_right, height_right), 1, (width_right, height_right)
        )

        # Compute and print reprojection errors
        error_left = self.compute_reprojection_error(self.object_points, self.image_points_left,
                                                     self.rvecs_left, self.tvecs_left,
                                                     self.camera_matrix_left, self.dist_coeffs_left)
        error_right = self.compute_reprojection_error(self.object_points, self.image_points_right,
                                                      self.rvecs_right, self.tvecs_right,
                                                      self.camera_matrix_right, self.dist_coeffs_right)
        print(f"Reprojection error for the left camera: {error_left}")
        print(f"Reprojection error for the right camera: {error_right}")

        # Replace intrinsic matrices with their optimal versions
        self.camera_matrix_left = optimal_camera_matrix_left
        self.camera_matrix_right = optimal_camera_matrix_right

    @staticmethod
    def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
        """
        Compute the total reprojection error across all images.
        """
        total_error = 0
        for i in range(len(object_points)):
            imgpoints2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv.norm(image_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            total_error += error
        return total_error / len(object_points)

    def stereo_calibrate(self):
        """
        Perform stereo calibration by fixing the intrinsics and computing the
        extrinsic parameters (rotation and translation between cameras).
        """
        flags = cv.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        (self.ret_stereo, self.camera_matrix_left, self.dist_coeffs_left,
         self.camera_matrix_right, self.dist_coeffs_right, self.rotation_matrix,
         self.translation_vector, self.essential_matrix, self.fundamental_matrix) = cv.stereoCalibrate(
            self.object_points, self.image_points_left, self.image_points_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            self.frame_size, criteria_stereo, flags
        )
        print(f"Stereo calibration reprojection error: {self.ret_stereo}")

    def stereo_rectify(self, rectify_scale=1):
        """
        Compute the stereo rectification parameters and corresponding undistortion maps.
        
        Args:
            rectify_scale (float): Scale factor for rectification (0=full crop, 1=no crop).
        """
        (self.rectification_matrix_left, self.rectification_matrix_right,
         self.projection_matrix_left, self.projection_matrix_right,
         self.disparity_to_depth_map, _, _) = cv.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            self.frame_size, self.rotation_matrix, self.translation_vector,
            rectify_scale, (0, 0)
        )
        # Compute rectification maps for each camera
        self.stereo_map_left = cv.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.rectification_matrix_left, self.projection_matrix_left,
            self.frame_size, cv.CV_16SC2
        )
        self.stereo_map_right = cv.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right,
            self.rectification_matrix_right, self.projection_matrix_right,
            self.frame_size, cv.CV_16SC2
        )

    def save_parameters(self, output_base_dir="calibrated_params"):
        """
        Save all calibration parameters to a YAML file.
        
        Args:
            output_base_dir (str): Base directory to save the calibration file.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(output_base_dir, current_time)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "stereo_calibration_params.yml")

        cv_file = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
        # Write intrinsic parameters
        cv_file.write('camera_matrix_left', self.camera_matrix_left)
        cv_file.write('dist_coeffs_left', self.dist_coeffs_left)
        cv_file.write('camera_matrix_right', self.camera_matrix_right)
        cv_file.write('dist_coeffs_right', self.dist_coeffs_right)
        # Write extrinsic parameters
        cv_file.write('rotation_matrix', self.rotation_matrix)
        cv_file.write('translation_vector', self.translation_vector)
        # Write rectification parameters
        cv_file.write('rectification_matrix_left', self.rectification_matrix_left)
        cv_file.write('rectification_matrix_right', self.rectification_matrix_right)
        cv_file.write('projection_matrix_left', self.projection_matrix_left)
        cv_file.write('projection_matrix_right', self.projection_matrix_right)
        cv_file.write('disparity_to_depth_map', self.disparity_to_depth_map)
        # Write rectification maps
        stereo_map_left_x, stereo_map_left_y = self.stereo_map_left
        stereo_map_right_x, stereo_map_right_y = self.stereo_map_right
        cv_file.write('stereo_map_left_x', stereo_map_left_x)
        cv_file.write('stereo_map_left_y', stereo_map_left_y)
        cv_file.write('stereo_map_right_x', stereo_map_right_x)
        cv_file.write('stereo_map_right_y', stereo_map_right_y)

        cv_file.release()
        print(f"All parameters saved in {filename}")

    def run_calibration(self):
        """
        Run the entire calibration pipeline.
        """
        self.load_images()
        self.detect_chessboard_corners()
        self.calibrate_cameras()
        self.stereo_calibrate()
        self.stereo_rectify()
        self.save_parameters()


def main():
    left_dir = 'images/stereo_left'
    right_dir = 'images/stereo_right'
    calibrator = StereoCalibrator(left_dir, right_dir,
                                  chessboard_size=(10, 7),
                                  square_size=25,
                                  show_corners=True)
    calibrator.run_calibration()


if __name__ == "__main__":
    main()
