import torch
import torchvision
import numpy as np

class Detector:
    def __init__(self):
        # Initialize a pretrained Faster R-CNN model from PyTorch Model Zoo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model with appropriate weights for pretrained detection
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1'
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

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
        # Extract data from the 'Center' camera sensor
        camera_data = sensor_data.get('Center')
        if camera_data is None:
            return {}

        frame_id, image = camera_data  # 'image' is RGBA (H, W, 4)
        image = image[:, :, :3]  # Remove the alpha channel for RGB format
        image = torch.tensor(image).permute(2, 0, 1).float().to(self.device)  # Convert to tensor and move to device
        image = image / 255.0  # Normalize to [0, 1] range
        input_tensor = image.unsqueeze(0)  # Add batch dimension for model input

        # Run inference without gradient calculation
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Prepare lists for detection results
        det_boxes = []
        det_class = []
        det_score = []

        # Process model output
        for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
            if score > 0.5:  # Confidence threshold
                box_np = box.cpu().numpy()
                # Convert bounding box to 8-point 3D format, e.g., (N, 8, 3)
                box_3d = self.convert_to_3d(box_np)
                det_boxes.append(box_3d)
                det_class.append(int(label.cpu().numpy()) - 1)  # Adjust class index as needed
                det_score.append(score.cpu().numpy())

        # Convert lists to numpy arrays in the required format
        return {
            "det_boxes": np.array(det_boxes),  # Shape (N, 8, 3) for 3D bounding boxes
            "det_class": np.array(det_class),
            "det_score": np.array(det_score)
        }

    def convert_to_3d(self, box):
        """
        Convert a 2D bounding box (x1, y1, x2, y2) to a 3D bounding box with 8 points.
        """
        x1, y1, x2, y2 = box
        z = 1.0  # Assume a default depth; adjust based on application if necessary
        box_3d = np.array([
            [x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0],  # Bottom face
            [x1, y1, z], [x2, y1, z], [x2, y2, z], [x1, y2, z]   # Top face
        ])
        return box_3d
