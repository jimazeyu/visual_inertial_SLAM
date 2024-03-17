import numpy as np
from utils import *
import scipy

class ExtendedKalmanFilterSLAM:
    def __init__(self, camera_intrinsics, baseline, num_features, cam_T_imu,
                 landmark_initial_noise, linear_velocity_noise, angular_velocity_noise, measurement_noise,
                 only_mapping=False):
        # if only_mapping is True, the robot pose will not be updated by measurement update
        self.only_mapping = only_mapping

        # Initialize camera intrinsics and baseline
        self.camera_intrinsics = camera_intrinsics
        self.baseline = baseline

        # Initialize noise levels
        self.landmark_initial_noise = landmark_initial_noise
        self.linear_velocity_noise = linear_velocity_noise
        self.angular_velocity_noise = angular_velocity_noise
        self.measurement_noise = measurement_noise

        # Initialize landmarks and their visibility
        self.num_features = num_features
        self.landmarks = np.zeros((3, self.num_features))
        self.landmark_visibility = np.zeros(self.num_features).astype(bool)

        # Initialize robot pose and covariance matrix
        self.robot_pose = np.eye(4)
        self.covariance = np.zeros((3 * self.num_features + 6, 3 * self.num_features + 6))
        self.covariance[:-6, :-6] = self.landmark_initial_noise * np.eye(3 * self.num_features) 

        # Transformation matrix from IMU to camera frame
        self.cam_T_imu = cam_T_imu

        # Transformation matrix to adjust IMU axes to regular axes
        self.imu_T_reg = np.eye(4)
        self.imu_T_reg[1, 1] = -1
        self.imu_T_reg[2, 2] = -1

    def predict(self, u, tau):
        # Update robot pose using motion model
        self.robot_pose = self.robot_pose @ scipy.linalg.expm(tau * axangle2twist(u))

        if not self.only_mapping:
            # Jacobian matrix of the motion model
            F = scipy.linalg.expm(-tau * axangle2adtwist(u))[0]

            # Process noise covariance matrix
            P = np.eye(6)
            P[:3, :3] *= self.linear_velocity_noise
            P[3:, 3:] *= self.angular_velocity_noise

            # Update covariance matrix
            self.covariance[-6:, -6:] = F @ self.covariance[-6:, -6:] @ F.T + P
            self.covariance[:-6, -6:] = self.covariance[:-6, -6:] @ F.T
            self.covariance[-6:, :-6] = F @ self.covariance[-6:, :-6]

    def update(self, features):
        # Calculate 3D coordinates of new landmarks
        u1, v1, u2, v2 = features
        depth = self.baseline * self.camera_intrinsics[0, 0] / (u1 - u2)
        x_new = (u1 - self.camera_intrinsics[0, 2]) * depth / self.camera_intrinsics[0, 0]
        y_new = (v1 - self.camera_intrinsics[1, 2]) * depth / self.camera_intrinsics[1, 1]
        z_new = depth

        # Filter out unseen landmarks and landmarks with too large depth
        new_idx = ~self.landmark_visibility & (features != -1).all(axis=0) & (depth < 100) & (depth > 0)
        new_points_cam = np.array([x_new, y_new, z_new])[:, new_idx]
        new_points_world = self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu @ np.vstack((new_points_cam, np.ones((1, new_points_cam.shape[1]))))

        # Update landmarks and their visibility
        self.landmarks[:, new_idx] = new_points_world[:3, :]
        self.landmark_visibility[new_idx] = True

        # Filter out existing landmarks
        existing_idx = (features != -1).all(axis=0) & self.landmark_visibility
        existing_landmarks = self.landmarks[:, existing_idx]
        selected_landmarks = np.where(existing_idx)[0]
        existing_cam = np.vstack((existing_landmarks, np.ones(existing_landmarks.shape[1])))

        # Camera projection matrix
        Ks = np.zeros((4, 4))
        Ks[:2, :3] = self.camera_intrinsics[:2, :]
        Ks[2:, :3] = self.camera_intrinsics[:2, :]
        Ks[2, 3] = -self.camera_intrinsics[0, 0] * self.baseline

        # Predicted observations
        predicted_obs = Ks @ projection((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) @ existing_cam).T).T

        # Compute errors
        errors = features[:, selected_landmarks] - predicted_obs
        errors = errors.T 

        # Filter out outliers with too large errors
        chosen = np.abs(errors).sum(axis=1) < 20
        errors = errors[chosen, :]
        selected_landmarks = selected_landmarks[chosen]
        existing_cam = existing_cam[:, chosen]
        num_seen = errors.shape[0]
        errors = errors.flatten()

        if num_seen <= 5:
            return

        # Compute measurement Jacobian for each observed feature
        H = np.zeros((4 * num_seen, 3 * self.num_features + 6))

        for i in range(num_seen):
            landmark_idx = selected_landmarks[i]

            P = np.zeros((3, 4))
            P[0, 0] = P[1, 1] = P[2, 2] = 1

            # Compute Jacobian for landmark
            H[4*i: 4*i+4, 3*landmark_idx: 3*landmark_idx+3] = Ks @ projectionJacobian((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) \
                                                    @ existing_cam[:, i][:, np.newaxis]).T)[0] \
                                                    @ np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) @ P.T

            if not self.only_mapping:
                # Compute Jacobian for robot pose
                H[4*i: 4*i+4, 3*self.num_features:] = -Ks @ projectionJacobian((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) \
                                                    @ existing_cam[:, i][:, np.newaxis]).T)[0] \
                                                    @ odot((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg \
                                                    @ self.cam_T_imu) @ existing_cam[:, i][:, np.newaxis]).T)[0]

        # Measurement noise covariance matrix
        R = np.eye(4 * num_seen) * self.measurement_noise

        if self.only_mapping:
            # Measurement update for only mapping
            K = self.covariance[:-6, :-6] @ H[:,:-6].T @ np.linalg.inv(H[:, :-6] @ self.covariance[:-6, :-6] @ H[:,:-6].T + R)

            to_update = np.zeros(3 * selected_landmarks.shape[0]).astype(int)
            to_update[0::3] = selected_landmarks.flatten() * 3
            to_update[1::3] += to_update[0::3] + 1
            to_update[2::3] += to_update[0::3] + 2

            cov_to_update = self.covariance[to_update[:, None], to_update]

            sub_H = H[:, to_update]
            sub_K = K[to_update]

            self.covariance[to_update[:, None], to_update] = (np.eye(cov_to_update.shape[0]) - sub_K @ sub_H) @ cov_to_update
            self.landmarks[:, selected_landmarks] = self.landmarks[:, selected_landmarks] + (K @ errors).reshape(-1, 3).T[:, selected_landmarks]
            
        else:
            # Measurement update for full SLAM
            K = self.covariance @ H.T @ np.linalg.inv(H @ self.covariance @ H.T + R)

            to_update = np.concatenate([
                np.zeros(3 * selected_landmarks.shape[0]),
                np.arange(3 * self.num_features, 3 * self.num_features + 6)
            ]).astype(int)
            to_update[0:-6:3] = selected_landmarks.flatten() * 3
            to_update[1:-6:3] += to_update[0:-6:3] + 1
            to_update[2:-6:3] += to_update[0:-6:3] + 2

            cov_to_update = self.covariance[to_update[:, None], to_update]

            sub_H = H[:, to_update]
            sub_K = K[to_update]

            self.covariance[to_update[:, None], to_update] = (np.eye(cov_to_update.shape[0]) - sub_K @ sub_H) @ cov_to_update
            self.robot_pose = np.dot(self.robot_pose, scipy.linalg.expm(axangle2twist((K @ errors)[-6:][np.newaxis, :])[0]))
            self.landmarks[:, selected_landmarks] = self.landmarks[:, selected_landmarks] + (K @ errors)[:-6].reshape(-1, 3).T[:, selected_landmarks]