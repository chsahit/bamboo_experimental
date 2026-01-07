import pyzed.sl as sl
import numpy as np


class ZedCamera:
    def __init__(self, serial_number: int | None = None):
        self.serial_number = serial_number
        init_params = sl.InitParameters()
        if serial_number is None:
            print("opening camera")
            self.zed = sl.Camera()
        else:
            print(f"opening camera {serial_number}")
            self.zed = sl.Camera(serial_number)
            init_params.set_from_serial_number(serial_number)
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED: {status}")
        if serial_number is not None:
            assert self.zed.get_camera_information().serial_number == serial_number

    def get_rgb_frame(self) -> np.ndarray:
        image = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        if self.zed.grab(runtime_params) <= sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW_LEFT)
        return image.get_data()

    def get_depth_frame(self) -> np.ndarray:
        image = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        if self.zed.grab(runtime_params) <= sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measire(image, sl.MEASURE_DEPTH)
        return image.get_data()

    def get_rgbd_frame(self) -> np.ndarray:
        rgb = self.get_rbg_frame()
        d = self.get_depth_frame()
        return np.concatenate([rgb, d])


    def close(self):
        self.zed.close()
