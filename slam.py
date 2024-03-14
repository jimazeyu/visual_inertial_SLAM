import numpy as np
from pr3_utils import *
import scipy

class ExtendedKalmanFilterSLAM:

    def __init__(self, camera_intrinsics, baseline, num_features, cam_T_imu,
                 landmark_initial_noise=0.1, linear_velocity_noise=1e-5, angular_velocity_noise=1e-5, measurement_noise=0.1,
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

        # Initialize robot pose and covariance matrix
        self.robot_pose = np.eye(4)
        self.covariance = np.zeros((3 * num_features + 6, 3 * num_features + 6))
        self.covariance[:-6, :-6] = self.landmark_initial_noise * np.eye(3 * num_features) 

        # Initialize landmarks and their visibility
        self.landmarks = np.zeros((3, num_features))
        self.landmark_visibility = np.zeros(num_features).astype(bool)
        self.num_features = num_features

        # Transformation matrix from IMU to camera frame
        self.cam_T_imu = cam_T_imu

        # Transformation matrix to adjust IMU axes to regular axes
        self.imu_T_reg = np.eye(4)
        self.imu_T_reg[1, 1] = -1
        self.imu_T_reg[2, 2] = -1

    def predict(self, u, dt):
        # Update robot pose using motion model
        self.robot_pose = self.robot_pose @ scipy.linalg.expm(dt * axangle2twist(u))

        if not self.only_mapping:
            # Jacobian matrix of the motion model
            F = scipy.linalg.expm(-dt * axangle2adtwist(u))[0]

            # Process noise covariance matrix
            W = np.eye(6)
            W[:3, :3] *= self.linear_velocity_noise
            W[3:, 3:] *= self.angular_velocity_noise

            # Update covariance matrix
            self.covariance[-6:, -6:] = F @ self.covariance[-6:, -6:] @ F.T + W
            self.covariance[:-6, -6:] = self.covariance[:-6, -6:] @ F.T
            self.covariance[-6:, :-6] = F @ self.covariance[-6:, :-6]

    def update(self, features):
        # Initialize or obtain landmarks from observations
        def initialize_and_obtain_landmarks(features, camera_intrinsics, baseline):
            # Filter out invalid features
            features_indices = (features != -1).all(axis=0)
            not_seen = ~self.landmark_visibility
            features_indices_unseen = features_indices & not_seen
            features = features[:, features_indices_unseen]

            ul, vl, ur, vr = features
            depth = baseline * camera_intrinsics[0, 0] / (ul - ur)

            # Calculate 3D coordinates of new landmarks
            new_landmark_points = np.array([
                (ul - camera_intrinsics[0, 2]) * depth / camera_intrinsics[0, 0],
                (vl - camera_intrinsics[1, 2]) * depth / camera_intrinsics[1, 1],
                depth
            ])

            # Filter out landmarks outside valid depth range
            features_indices_unseen[features_indices_unseen.copy()] = (depth < 80) & (depth > 0)
            new_landmark_points = new_landmark_points[:, (depth < 80) & (depth > 0)]

            # Transform new landmarks to world frame
            new_landmark_points = self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu @ np.vstack((new_landmark_points, np.ones((1, new_landmark_points.shape[1]))))

            # Update landmarks and their visibility
            self.landmarks[:, features_indices_unseen] = new_landmark_points[:3, :]
            self.landmark_visibility[features_indices_unseen] = True

            # Filter out existing landmarks
            features_indices = features_indices & ~not_seen
            if np.any(features_indices):
                landmarks_hom = np.vstack((self.landmarks[:, features_indices], np.ones((1, np.sum(features_indices)))))
                depth = np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) @ landmarks_hom

            return self.landmarks[:, features_indices], np.argwhere(features_indices).flatten()
        
        landmarks_from_pose, selected_landmarks = initialize_and_obtain_landmarks(features, self.camera_intrinsics, self.baseline)
        num_features_seen = landmarks_from_pose.shape[1]

        landmarks_from_pose_hom = np.vstack((landmarks_from_pose, np.ones(landmarks_from_pose.shape[1])))

        # Camera projection matrix
        Ks = np.zeros((4, 4))
        Ks[:2, :3] = self.camera_intrinsics[:2, :]
        Ks[2:, :3] = self.camera_intrinsics[:2, :]
        Ks[2, 3] = -self.camera_intrinsics[0, 0] * self.baseline

        # Predicted observations
        predicted_observation = Ks @ projection((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) @ landmarks_from_pose_hom).T).T

        # Compute residual
        residual = features[:, selected_landmarks] - predicted_observation
        residual = residual.T 

        # Filter out outliers based on residual
        chosen_features = np.abs(residual).sum(axis=1) < 20
        residual = residual[chosen_features, :]
        selected_landmarks = selected_landmarks[chosen_features]
        num_features_seen = residual.shape[0]
        landmarks_from_pose = landmarks_from_pose[:, chosen_features]
        landmarks_from_pose_hom = landmarks_from_pose_hom[:, chosen_features]
        residual = residual.flatten()

        # Initialize measurement Jacobian matrix
        H = np.zeros((4 * num_features_seen, 3 * self.num_features + 6))

        # Compute measurement Jacobian for each observed feature
        for i in range(num_features_seen):
            landmark_idx = selected_landmarks[i]

            P = np.zeros((3, 4))
            P[0, 0] = P[1, 1] = P[2, 2] = 1

            # Compute Jacobian for landmark
            out_landmark = Ks @ projectionJacobian((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) \
                                                    @ landmarks_from_pose_hom[:, i][:, np.newaxis]).T)[0] \
                                                    @ np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) @ P.T
            H[4*i: 4*i+4, 3*landmark_idx: 3*landmark_idx+3] = out_landmark

            if not self.only_mapping:
                # Compute Jacobian for robot pose
                out_pose = -Ks @ projectionJacobian((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg @ self.cam_T_imu) \
                                                     @ landmarks_from_pose_hom[:, i][:, np.newaxis]).T)[0] \
                                                    @ circdot((np.linalg.inv(self.robot_pose[0] @ self.imu_T_reg \
                                                    @ self.cam_T_imu) @ landmarks_from_pose_hom[:, i][:, np.newaxis]).T)[0]
                H[4*i: 4*i+4, 3*self.num_features:] = out_pose

        # Measurement noise covariance matrix
        I_x_V = np.eye(4 * num_features_seen) * self.measurement_noise

        if self.only_mapping:
            # Measurement update for mapping
            K = self.covariance[:-6, :-6] @ H[:,:-6].T @ np.linalg.inv(H[:, :-6] @ self.covariance[:-6, :-6] @ H[:,:-6].T + I_x_V)

            idxs_to_be_updated = np.zeros(3 * selected_landmarks.shape[0]).astype(int)

            idxs_to_be_updated[0::3] = selected_landmarks.flatten() * 3
            idxs_to_be_updated[1::3] += idxs_to_be_updated[0::3] + 1
            idxs_to_be_updated[2::3] += idxs_to_be_updated[0::3] + 2

            cov_to_be_updated = self.covariance[idxs_to_be_updated[:, None], idxs_to_be_updated]

            sub_selected_H = H[:, idxs_to_be_updated]
            sub_selected_K = K[idxs_to_be_updated]

            cov_updated = (np.eye(cov_to_be_updated.shape[0]) - sub_selected_K @ sub_selected_H) @ cov_to_be_updated

            self.covariance[idxs_to_be_updated[:, None], idxs_to_be_updated] = cov_updated

            delta = K @ residual

            self.landmarks[:, selected_landmarks] = self.landmarks[:, selected_landmarks] + delta[:-6].reshape(-1, 3).T[:, selected_landmarks]
            
        else:
            # Measurement update for full SLAM
            K = self.covariance @ H.T @ np.linalg.inv(H @ self.covariance @ H.T + I_x_V)

            idxs_to_be_updated = np.concatenate([
                np.zeros(3 * selected_landmarks.shape[0]),
                np.arange(3 * self.num_features, 3 * self.num_features + 6)
            ]).astype(int)

            idxs_to_be_updated[0:-6:3] = selected_landmarks.flatten() * 3
            idxs_to_be_updated[1:-6:3] += idxs_to_be_updated[0:-6:3] + 1
            idxs_to_be_updated[2:-6:3] += idxs_to_be_updated[0:-6:3] + 2

            cov_to_be_updated = self.covariance[idxs_to_be_updated[:, None], idxs_to_be_updated]

            sub_selected_H = H[:, idxs_to_be_updated]
            sub_selected_K = K[idxs_to_be_updated]

            cov_updated = (np.eye(cov_to_be_updated.shape[0]) - sub_selected_K @ sub_selected_H) @ cov_to_be_updated

            self.covariance[idxs_to_be_updated[:, None], idxs_to_be_updated] = cov_updated

            delta = K @ residual

            # Update robot pose and landmarks
            self.robot_pose = np.dot(self.robot_pose, scipy.linalg.expm(axangle2twist(delta[-6:][np.newaxis, :])[0]))

            self.landmarks[:, selected_landmarks] = self.landmarks[:, selected_landmarks] + delta[:-6].reshape(-1, 3).T[:, selected_landmarks]