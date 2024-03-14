import numpy as np
import matplotlib.pyplot as plt
import pr3_utils
import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from slam import ExtendedKalmanFilterSLAM


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Extract relevant configuration
    dataset = os.path.join(get_original_cwd(), cfg.run.dataset)
    mode = cfg.run.mode
    downsample = cfg.run.downsample

    # Load data
    t, features, linear_velocity, angular_velocity, K, b, cam_T_imu = pr3_utils.load_data(dataset)
    total_steps = t.shape[1]
    t = t[0, :]
    features = features[:, ::downsample, :]

    # Initialize EKF SLAM
    if mode == 3:
        only_mapping = False
    else:
        only_mapping = True
    ekf = ExtendedKalmanFilterSLAM(camera_intrinsics=K, baseline=b,
                    num_features=features.shape[1], cam_T_imu=cam_T_imu, only_mapping=only_mapping,
                    landmark_initial_noise=cfg.params.landmark_initial_noise, linear_velocity_noise=cfg.params.linear_velocity_noise,
                    angular_velocity_noise=cfg.params.angular_velocity_noise, measurement_noise=cfg.params.measurement_noise)  

    poses = []
    fig, ax = plt.subplots(figsize=(10, 10))

    # Run SLAM
    for i in range(1, total_steps):
        u = np.concatenate((linear_velocity[:, i], angular_velocity[:, i]))
        # Predict
        ekf.predict(u[np.newaxis, :], t[i] - t[i - 1])
        # Update
        if mode == 2 or mode == 3:
            ekf.update(features[:, :, i])

        poses.append(ekf.robot_pose[0])

        if i % 2 == 1:
            ax.clear()
            poses_to_show = np.array(poses).transpose(1, 2, 0)
            pr3_utils.visualize_trajectory_2d(ax, poses_to_show, ekf.landmarks)
            plt.pause(0.01)

        print(f"Step {i}/{total_steps}")

    plt.show()

if __name__ == "__main__":
    main()