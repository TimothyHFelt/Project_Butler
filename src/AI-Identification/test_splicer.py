import pytest
import cv2
import os

from splicer import SpliceFailureException, Splicer




# x, y, w, h, then 5 landmarks (right eye, left eye, nose, right mouth, left mouth), then score.
# Landmarks must be plausible: all zeros can't be aligned.
output_dir = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops"
os.makedirs(output_dir, exist_ok=True)

class TestSplicer:

    def _save_crops(self, crops: list, base_filename: str):
        """
        Saves the extracted face crops to disk for visual inspection.
        
        Args:
            crops: List of aligned face crop matrices.
            base_filename: Base filename to use for saving crops.
        """
        for idx, crop in enumerate(crops):
            crop_filename = os.path.join(output_dir, f"{base_filename}_crop_{idx}.png")
            face_matrix = crop['crop']
              # Assuming each item in crops is a dict with 'crop' key
            cv2.imwrite(crop_filename, face_matrix)
        
    def test_splice_success(self):
        splicer = Splicer()
        
        # 1. Define the path
        image_path = "/home/timothyfelt/Project_Butler/data/faces/Tim_daughter.jpeg"
        
        # 2. Read the image into a NumPy array using OpenCV
        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

        # Ensure the image was loaded successfully before proceeding
        assert frame is not None, f"Failed to load image at {image_path}"
        
        # 3. Pass the loaded array to the splice method, not the string path
        result = splicer.splice(frame)
        
        # Assertions
        assert result is not None
        if result is not None:
            self._save_crops(result, "Tim_daughter")
        assert isinstance(result, list) # Note: Your docstring says it returns a list
        assert len(result) > 0 and len(result) < 2  # Expecting a small number of faces detected

    def test_splice_success2(self):
        splicer = Splicer()
        
        # 1. Define the path
        image_path = "/home/timothyfelt/Project_Butler/data/faces/Kashton_RedRocks.jpeg"
        
        # 2. Read the image into a NumPy array using OpenCV
        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

        
        # Ensure the image was loaded successfully before proceeding
        assert frame is not None, f"Failed to load image at {image_path}"
        
        # 3. Pass the loaded array to the splice method, not the string path
        result = splicer.splice(frame)
        
        # Assertions
        assert result is not None
        if result is not None:
            self._save_crops(result, "Kashton_RedRocks")
        assert isinstance(result, list) # Note: Your docstring says it returns a list
        assert len(result) > 0 and len(result) < 2  # Expecting a small number of faces detected

    def test_splice_success3(self):
        splicer = Splicer()
        
        # 1. Define the path
        image_path = "/home/timothyfelt/Project_Butler/data/faces/Kashton_smile.jpeg"
        
        # 2. Read the image into a NumPy array using OpenCV
        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

        
        # Ensure the image was loaded successfully before proceeding
        assert frame is not None, f"Failed to load image at {image_path}"
        
        # 3. Pass the loaded array to the splice method, not the string path
        result = splicer.splice(frame)
        
        # Assertions
        assert result is not None
        if result is not None:
            self._save_crops(result, "Kashton_Smile")
        assert isinstance(result, list) # Note: Your docstring says it returns a list
        assert len(result) > 0 and len(result) < 2  # Expecting a small number of faces detected

    def test_splice_two_people(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/marriage_2_people.jpeg"

        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)


        assert frame is not None, f"Failed to load image at {image_path}"

        result = splicer.splice(frame)

        assert result is not None
        if result is not None:
            self._save_crops(result, "marriage_2_people")
        assert isinstance(result, list)
        assert len(result) > 1 and len(result) < 3  # Expecting more than one face detected

    def test_splice_family(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/halloween_family.jpeg"

        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)


        assert frame is not None, f"Failed to load image at {image_path}"

        result = splicer.splice(frame)

        assert result is not None
        if result is not None:
            self._save_crops(result, "halloween_family")
        assert isinstance(result, list)
        assert len(result) > 1 and len(result) < 4  # Expecting more than one face detected


    def test_splice_no_faces(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/chocolate_fountain.jpeg"

        frame = cv2.imread(image_path) 

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)


        assert frame is not None, f"Failed to load image at {image_path}"

        with pytest.raises(SpliceFailureException) as excinfo:
            splicer.splice(frame)

        assert "Zero faces detected." in str(excinfo.value)

    def test_splice_merged_faces(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/kiss.jpeg"

        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

        assert frame is not None, f"Failed to load image at {image_path}"
        
        result = splicer.splice(frame)

        assert result is not None

        if result is not None:
            self._save_crops(result, "kiss")

        assert isinstance(result, list)
        assert len(result) > 1 and len(result) < 3 

    def test_fisheye(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/fisheye.jpeg"

        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

        assert frame is not None, f"Failed to load image at {image_path}"

        result = splicer.splice(frame)

        assert result is not None
        if result is not None:
            self._save_crops(result, "fisheye")
        assert isinstance(result, list)
        assert len(result) > 0 and len(result) < 2  # Expecting a small number of faces detected

    def test_iou_Perfect_Overlap(self):
        splicer = Splicer()
                # Test Case 1: Perfect Overlap (100% Identical)
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        assert splicer._calculate_iou(box1, box2) == pytest.approx(1.0)

    def test_iou_Partial_Overlap(self):
        splicer = Splicer()
        # Test Case 2: Partial Overlap
        box1 = [10, 10, 50, 50]
        box2 = [30, 30, 50, 50]
        expected_iou = (30 * 30) / float((50 * 50) + (50 * 50) - (30 * 30))
        assert splicer._calculate_iou(box1, box2) == pytest.approx(expected_iou)

    def test_iou_No_Overlap(self):
        splicer = Splicer()
        # Test Case 3: No Overlap
        box1 = [10, 10, 50, 50]
        box2 = [100, 100, 50, 50]
        assert splicer._calculate_iou(box1, box2) == pytest.approx(0.0)

    def test_iou_One_Inside_Other(self):
        splicer = Splicer()
        # Test Case 4: One Box Inside Another
        box1 = [10, 10, 50, 50]
        box2 = [20, 20, 10, 10]
        expected_iou = (10 * 10) / float((50 * 50))
        assert splicer._calculate_iou(box1, box2) == pytest.approx(expected_iou)

    def test_iou_Zero_Area_Box(self):
        splicer = Splicer()
        # Test Case 5: Zero Area Box
        box1 = [10, 10, 0, 0]
        box2 = [20, 20, 10, 10]
        assert splicer._calculate_iou(box1, box2) == pytest.approx(0.0)

    def test_iou_touching_edges(self):
        splicer = Splicer()
        # Test Case 6: Touching Edges
        box1 = [10, 10, 50, 50]
        box2 = [60, 10, 50, 50]
        assert splicer._calculate_iou(box1, box2) == pytest.approx(0.0)