import numpy as np
import pytest
import cv2
import os

from splicer import SpliceFailureException, Splicer




# x, y, w, h, then 5 landmarks (right eye, left eye, nose, right mouth, left mouth), then score.
# Landmarks must be plausible: all zeros can't be aligned.
output_dir = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops"
os.makedirs(output_dir, exist_ok=True)

class TestSplicer:
    def test_splice_success(self):
        splicer = Splicer()
        
        # 1. Define the path
        image_path = "/home/timothyfelt/Project_Butler/data/faces/IMG_6844.jpeg"
        
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
        assert isinstance(result, list) # Note: Your docstring says it returns a list
        assert len(result) > 0 and len(result) < 2  # Expecting a small number of faces detected

    def test_splice_success2(self):
        splicer = Splicer()
        
        # 1. Define the path
        image_path = "/home/timothyfelt/Project_Butler/data/faces/IMG_7988.jpeg"
        
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
        assert isinstance(result, list) # Note: Your docstring says it returns a list
        assert len(result) > 0 and len(result) < 2  # Expecting a small number of faces detected
    

    def test_splice_two_people(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/IMG_6811.jpeg"

        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)


        assert frame is not None, f"Failed to load image at {image_path}"

        result = splicer.splice(frame)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 1 and len(result) < 3  # Expecting more than one face detected

    def test_splice_no_faces(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/IMG_9272.jpeg"

        frame = cv2.imread(image_path) 

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)


        assert frame is not None, f"Failed to load image at {image_path}"

        with pytest.raises(SpliceFailureException) as excinfo:
            splicer.splice(frame)

        assert "Zero faces detected." in str(excinfo.value)

    def test_splice_merged_faces(self):
        splicer = Splicer()
        image_path = "/home/timothyfelt/Project_Butler/data/faces/Kiss_overlap.jpeg"

        frame = cv2.imread(image_path)

        scale_factor = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

        exception_caught = False
        assert frame is not None, f"Failed to load image at {image_path}"
        try:
            result = splicer.splice(frame)
        except SpliceFailureException as e:
            exception_caught = True
            assert "Detection confidence" in str(e) or "Overlapping faces detected." in str(e)

        if exception_caught:
            # If the code raises the error, we verify the exception was caught successfully
            self.assertTrue(exception_caught, "Expected SpliceFailureException due to overlapping boxes.")

            if result is not None:
                self.assertEqual(len(result), 2, "Expected 2 faces to be returned due to overlapping boxes.")
        else:
            assert result is not None
            for index, face_array in enumerate(result):
                filename = f"{output_dir}/face_{index + 1}.jpg"
                face_array = face_array.get('face') or face_array.get('image') or face_array.get('crop')

                success = cv2.imwrite(filename, face_array)
                if success:
                    print(f"Successfully saved face crop to: {filename}")
                else:
                    print(f"Failed to save image at: {filename}")
            assert isinstance(result, list)

            assert len(result) > 1 and len(result) < 3  # Expecting more than one face detected