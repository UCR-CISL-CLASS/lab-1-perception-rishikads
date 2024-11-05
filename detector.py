import numpy as np
# import sys
# import os
# module_path = os.path.abspath(os.path.join('..', 'mmdetection3D'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from mmdet3d.apis import init_model, inference_detector
import mmcv

class Detector:
    def __init__(self):
        # Initialize MMDetection3D model
        config_file = 'mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
        checkpoint_file = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

    def sensors(self):  # pylint: disable=no-self-use
        # Sensor configurations for CARLA environment
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.5, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.5, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},
            {'type': 'sensor.camera.rgb', 'x': 1.0, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'range': 50, 'rotation_frequency': 20, 'channels': 64,
             'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000, 'id': 'LIDAR'},
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
        ]
        return sensors

    def detect(self, sensor_data):
        """
        Run detection on sensor data using the MMDetection3D model.

        Args:
            sensor_data: Dictionary containing sensor data. 
                         Format: { 'LIDAR': (frame_id, np.ndarray), ... }

        Returns:
            Dictionary of detection results.
        """
        # Prepare data according to model requirements (assuming lidar data)
        lidar_data = sensor_data['LIDAR'][1]  # Extract lidar point cloud data

        # Run inference
        result = inference_detector(self.model, lidar_data)
        
        # Extract results and format output
        det_boxes, det_class, det_score = [], [], []

        import pdb; pdb.set_trace()
        for det in result[0]:  # Loop through detected objects
            # if det['scores'] > 0.5:  # Detection confidence threshold
            det_boxes.append(det['bbox_3d'].corners.numpy())
            det_class.append(det['label'])
            det_score.append(det['scores'])

        return {
            'det_boxes': np.array(det_boxes),
            'det_class': np.array(det_class),
            'det_score': np.array(det_score)
        }

    def get_sensor_setup(self):
        """
        Returns the sensor setup for the perception module.

        Returns:
            List of sensor configurations.
        """
        return self.sensors
