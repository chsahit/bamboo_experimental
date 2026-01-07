import numpy as np
import pyzed.sl as sl
import pickle
from perception.zed.zed_cam import ZedCamera


def get_zed_left_intrinsics(serial_number: int | None = None):
    cam = ZedCamera(serial_number = serial_number)
    cam_info = cam.zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters

    left = calib.left_cam

    fx = left.fx
    fy = left.fy
    cx = left.cx
    cy = left.cy

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # ZED distortion coefficients (keep these!)
    # Order is ZED-specific but compatible with OpenCV's rational model
    dist_coeffs = np.array(left.disto, dtype=np.float64)

    # Resolution intrinsics correspond to
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height

    cam.close()

    return K, dist_coeffs, (width, height)


if __name__ == "__main__":
    sn_to_intrinsics = {}
    available = sl.Camera.get_device_list()
    for i, cam in enumerate(available):
        print(f"camera {i}, serial number {cam.serial_number}")

        K, dist, resolution = get_zed_left_intrinsics(serial_number=cam.serial_number)

        print("Camera matrix K:")
        print(K)

        print("\nDistortion coefficients:")
        print(dist)

        print("\nResolution:")
        print(resolution)
        sn_to_intrinsics[cam.serial_number] = {"K": K, "dist": dist, "resolution": resolution}


    with open("intrinsics.pkl", "wb") as f:
        pickle.dump(sn_to_intrinsics, f, pickle.HIGHEST_PROTOCOL)
    print("saved to intrinsics.pkl")
