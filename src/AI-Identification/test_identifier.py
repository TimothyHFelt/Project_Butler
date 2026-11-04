import pytest
import cv2
import os

from identifier import Identifier

class TestIdentifier:
    def test_extract_embedding(self):
        identifier = Identifier(model_path="/home/timothyfelt/Project_Butler/data/faces/face_recognition_sface_2021dec.onnx")
        image_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Kashton_RedRocks_crop_0.png"

        crop = cv2.imread(image_path)

        assert crop is not None, f"Failed to load image at {image_path}"

        embedding = identifier.extract_embedding(crop)

        assert embedding is not None
        assert embedding.shape == (1, 128)  # SFace embeddings are 128-dimensional

    def test_identify_known_face(self):


        identifier = Identifier(model_path="/home/timothyfelt/Project_Butler/data/faces/face_recognition_sface_2021dec.onnx")
        image_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Kashton_RedRocks_crop_0.png"
        image2_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kashton_fishtank_crop_0.png"
        image3_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Kashton_Smile_crop_0.png"



        crop = cv2.imread(image_path)
        crop2 = cv2.imread(image2_path)
        crop3 = cv2.imread(image3_path)
        assert crop is not None, f"Failed to load image at {image_path}"
        assert crop2 is not None, f"Failed to load image at {image2_path}"
        assert crop3 is not None, f"Failed to load image at {image3_path}"

        [matched_identity, match_score, runner_up_score] = identifier.identify(crop, {"Kashton": [identifier.extract_embedding(crop2), identifier.extract_embedding(crop3)]})
        assert matched_identity is not None
        assert matched_identity == "Kashton"  # Expecting the identity to match "Kashton"
        assert match_score > 0.363  # Assuming the match threshold is 0.363

        # Simulate a reference database with one known embedding

    def test_identify_partially_occluded_face(self):
        identifier = Identifier(model_path="/home/timothyfelt/Project_Butler/data/faces/face_recognition_sface_2021dec.onnx")
        image_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/marriage_2_people_crop_1.png"
        occluded_image_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kiss_crop_1.png"
        second_crop_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/halloween_family_crop_0.png"
        third_crop_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Tim_daughter_crop_0.png"
        crop = cv2.imread(image_path)
        occluded_crop = cv2.imread(occluded_image_path)
        second_crop = cv2.imread(second_crop_path)
        third_crop = cv2.imread(third_crop_path)

        assert crop is not None, f"Failed to load image at {image_path}"
        assert occluded_crop is not None, f"Failed to load image at {occluded_image_path}"
        assert second_crop is not None, f"Failed to load image at {second_crop_path}"

        # Simulate a reference database with one known embedding
        [matched_identity, match_score, runner_up_score] = identifier.identify(second_crop, {"Timothy": [identifier.extract_embedding(crop), identifier.extract_embedding(occluded_crop), identifier.extract_embedding(third_crop)]})

        assert matched_identity == "Timothy"
        assert match_score > 0.363  # Assuming the match threshold is 0.363

    def test_identify_Timothy_from_multiple_faces(self):
        identifier = Identifier(model_path="/home/timothyfelt/Project_Butler/data/faces/face_recognition_sface_2021dec.onnx")
        kenzie_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/marriage_2_people_crop_0.png"
        kenzie1_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/halloween_family_crop_1.png"
        kenzie2_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kiss_crop_0.png"
        kenzie_crop = cv2.imread(kenzie_path)
        kenzie1_crop = cv2.imread(kenzie1_path)
        kenzie2_crop = cv2.imread(kenzie2_path)

        kashton_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Kashton_Smile_crop_0.png"
        kashton1_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Kashton_RedRocks_crop_0.png"
        kashton2_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kashton_fishtank_crop_0.png"
        kashton_crop = cv2.imread(kashton_path)
        kashton1_crop = cv2.imread(kashton1_path)
        kashton2_crop = cv2.imread(kashton2_path)

        image_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Tim_daughter_crop_0.png"
        tim1_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/halloween_family_crop_0.png"
        tim2_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kiss_crop_1.png"
        tim3_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/marriage_2_people_crop_1.png"
        image_crop = cv2.imread(image_path)
        tim1_crop = cv2.imread(tim1_path)
        tim2_crop = cv2.imread(tim2_path)
        tim3_crop = cv2.imread(tim3_path)

        assert kenzie_crop is not None, f"Failed to load image at {kenzie_path}"
        assert kenzie1_crop is not None, f"Failed to load image at {kenzie1_path}"
        assert kenzie2_crop is not None, f"Failed to load image at {kenzie2_path}"
        assert kashton_crop is not None, f"Failed to load image at {kashton_path}"
        assert kashton1_crop is not None, f"Failed to load image at {kashton1_path}"
        assert kashton2_crop is not None, f"Failed to load image at {kashton2_path}"
        assert image_crop is not None, f"Failed to load image at {image_path}" 
        assert tim1_crop is not None, f"Failed to load image at {tim1_path}"
        assert tim2_crop is not None, f"Failed to load image at {tim2_path}"
        assert tim3_crop is not None, f"Failed to load image at {tim3_path}"

        [matched_identity, match_score, runner_up_score] = identifier.identify(image_crop, {"Kenzie": [identifier.extract_embedding(kenzie_crop), identifier.extract_embedding(kenzie1_crop), identifier.extract_embedding(kenzie2_crop)],
                                                                                          "Kashton": [identifier.extract_embedding(kashton_crop), identifier.extract_embedding(kashton1_crop), identifier.extract_embedding(kashton2_crop)],
                                                                                          "Timothy": [identifier.extract_embedding(tim1_crop), identifier.extract_embedding(tim2_crop), identifier.extract_embedding(tim3_crop)]})
        assert matched_identity is not None
        assert matched_identity == "Timothy"  # Expecting the identity to match "Timothy"
        assert match_score > 0.363  # Assuming the match threshold is 0.363

    def test_identify_Kashton_from_multiple_face(self):
        identifier = Identifier(model_path="/home/timothyfelt/Project_Butler/data/faces/face_recognition_sface_2021dec.onnx")
        kenzie_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/marriage_2_people_crop_0.png"
        kenzie1_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/halloween_family_crop_1.png"
        kenzie2_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kiss_crop_0.png"
        kenzie_crop = cv2.imread(kenzie_path)
        kenzie1_crop = cv2.imread(kenzie1_path)
        kenzie2_crop = cv2.imread(kenzie2_path)

        image_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Kashton_Smile_crop_0.png"
        kashton1_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Kashton_RedRocks_crop_0.png"
        kashton2_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kashton_fishtank_crop_0.png"
        image_crop = cv2.imread(image_path)
        kashton1_crop = cv2.imread(kashton1_path)
        kashton2_crop = cv2.imread(kashton2_path)

        tim_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/Tim_daughter_crop_0.png"
        tim1_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/halloween_family_crop_0.png"
        tim2_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/kiss_crop_1.png"
        tim3_path = "/home/timothyfelt/Project_Butler/data/faces/extracted_crops/marriage_2_people_crop_1.png"
        tim_crop = cv2.imread(tim_path)
        tim1_crop = cv2.imread(tim1_path)
        tim2_crop = cv2.imread(tim2_path)
        tim3_crop = cv2.imread(tim3_path)

        assert kenzie_crop is not None, f"Failed to load image at {kenzie_path}"
        assert kenzie1_crop is not None, f"Failed to load image at {kenzie1_path}"
        assert kenzie2_crop is not None, f"Failed to load image at {kenzie2_path}"
        assert image_crop is not None, f"Failed to load image at {image_path}"
        assert kashton1_crop is not None, f"Failed to load image at {kashton1_path}"
        assert kashton2_crop is not None, f"Failed to load image at {kashton2_path}"
        assert tim_crop is not None, f"Failed to load image at {tim_path}"
        assert tim1_crop is not None, f"Failed to load image at {tim1_path}"
        assert tim2_crop is not None, f"Failed to load image at {tim2_path}"
        assert tim3_crop is not None, f"Failed to load image at {tim3_path}"

        [matched_identity, match_score, runner_up_score] = identifier.identify(image_crop, {"Kenzie": [identifier.extract_embedding(kenzie_crop), identifier.extract_embedding(kenzie1_crop), identifier.extract_embedding(kenzie2_crop)],
                                                                                          "Kashton": [identifier.extract_embedding(kashton1_crop), identifier.extract_embedding(kashton2_crop)],
                                                                                          "Timothy": [identifier.extract_embedding(tim_crop), identifier.extract_embedding(tim1_crop), identifier.extract_embedding(tim2_crop), identifier.extract_embedding(tim3_crop)]})
        assert matched_identity is not None
        assert matched_identity == "Kashton"  # Expecting the identity to match "Kashton"
        assert match_score > 0.363  # Assuming the match threshold is 0