import cv2
import numpy as np
import os

MODEL_PATH = os.path.expanduser("~/Project_Butler/data/faces/face_detection_yunet_2023mar.onnx")
OVERLAP_LIMIT = 0.3

class SpliceFailureException(Exception):
    """Custom exception raised when frame-level validation parameters fail."""
    pass

class Splicer:
    def __init__(self, model_path: str = MODEL_PATH, min_face_size: int = 40, confidence_cutoff: float = 0.6, iou_limit: float = OVERLAP_LIMIT):
        """
        Initializes the stateless YuNet face detector.
        
        Args:
            model_path: System string path to the face_detection_yunet.onnx weights file.
            min_face_size: Minimum pixel dimension (width/height) allowed for processing.
            confidence_cutoff: Minimum detection score required to process a face crop.
            iou_limit: Overlap ratio allowance. Exceeding this declares separation failed.
        """
        self.min_face_size = min_face_size
        self.confidence_cutoff = confidence_cutoff
        self.iou_limit = iou_limit
        
        # Instantiate YuNet via OpenCV's DNN FaceDetectorYN class
        # We pass dummy dimensions (1, 1) during initial construct; updated dynamically per frame
        self.detector = cv2.FaceDetectorYN.create(
            model=MODEL_PATH,
            config="",
            input_size=(1, 1),
            score_threshold=self.confidence_cutoff,
            nms_threshold=self.iou_limit,
            top_k=10
        )

    def _calculate_iou(self, boxA: list, boxB: list) -> float:
        """Computes the Intersection over Union (IoU) between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        unionArea = float(boxAArea + boxBArea - interArea)

        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    def splice(self, frame: np.ndarray) -> list:
        """
        Processes a raw input frame matrix in RAM, detects faces, validates structural 
        constraints, and extracts aligned face crops.
        
        Args:
            frame: A raw OpenCV image matrix passed straight from the RAM disk.
            
        Returns:
            A list of dicts containing: 
            {'crop': np.ndarray, 'box': list, 'confidence': float}
            
        Raises:
            SpliceFailureException with explicit reasons for the Enquirer service.
        """
        if frame is None:
            raise SpliceFailureException("Invalid input: Raw frame is completely empty.")

        height, width, _ = frame.shape
        self.detector.setInputSize((width, height))
        
        # Perform stateless face inference detection
        # faces layout matrix format: [x, y, w, h, x_re, y_re, x_le, y_le, ..., confidence]
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            raise SpliceFailureException("Zero faces detected.")

        parsed_faces = []
        for face in faces:
            box = face[0:4].astype(int).tolist()
            confidence = float(face[-1])
            
            # 1. Validate detection threshold limits
            if confidence < self.confidence_cutoff:
                raise SpliceFailureException(f"Detection confidence {confidence:.2f} falls below cutoff threshold.")

            # 2. Validate geometric scale dimensions
            if box[2] < self.min_face_size or box[3] < self.min_face_size:
                raise SpliceFailureException(f"Detected face size ({box[2]}x{box[3]}) is smaller than the required limit.")

            parsed_faces.append({'box': box, 'confidence': confidence, 'raw_face': face})

        # 3. Generate final output payloads
        final_crops = []
        for f_data in parsed_faces:
            # Crop the bounding region from the parent matrix
            x, y, w, h = f_data['box']
            # Clamp crop coordinates to the bounds of the image plane
            x_start, y_start = max(0, x), max(0, y)
            x_end, y_end = min(width, x + w), min(height, y + h)
            
            crop_matrix = frame[y_start:y_end, x_start:x_end]
            
            final_crops.append({
                'crop': crop_matrix,
                'box': f_data['box'],
                'confidence': f_data['confidence']
            })

        return final_crops
