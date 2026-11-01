import cv2
import numpy as np

# 1. Define paths to your downloaded models
YUNET_MODEL = "../data/faces/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "../data/faces/face_recognition_sface_2021dec.onnx"

# 2. Initialize YuNet Face Detector
# We initialize with dummy dimensions (1, 1); they will update dynamically per image.
detector = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL,
    config="",
    input_size=(1, 1),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=500
)

# 3. Initialize SFace Face Recognizer
recognizer = cv2.FaceRecognizerSF.create(
    model=SFACE_MODEL,
    config=""
)

def get_face_feature(img_path):
    """Loads an image, downscales it for YuNet, detects, aligns, and extracts features."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return None

    # --- ADDED: Scale down large smartphone images so YuNet can detect the face ---
    max_dimension = 1024
    height, width, _ = img.shape
    if max(height, width) > max_dimension:
        scale = max_dimension / float(max(height, width))
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        # Recalculate dimensions after scaling
        height, width, _ = img.shape

    # Update YuNet input dimensions dynamically to the new scaled size
    detector.setInputSize((width, height))
    
    # Detect faces
    _, faces = detector.detect(img)
    
    if faces is None or len(faces) == 0:
        print(f"No face detected in {img_path} (Scaled Size: {width}x{height})")
        return None
    
    # Align and crop the first detected face, then extract its vector identity fingerprint
    aligned_face = recognizer.alignCrop(img, faces)
    feature = recognizer.feature(aligned_face)
    return feature


def verify_user(image_a_path, image_b_path):
    """
    Compares two images using commercially-safe open-source models natively.
    Handles different backgrounds, lightings, and environments.
    """
    try:
        # Extract identity fingerprints
        feat1 = get_face_feature(image_a_path)
        feat2 = get_face_feature(image_b_path)
        
        if feat1 is None or feat2 is None:
            print("Verification failed because features couldn't be extracted.")
            return False, 0.0
            
        # Match using Cosine Similarity
        cosine_score = recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        
        # SFace standard threshold: Cosine similarity >= 0.363 means same person
        is_same_person = bool(cosine_score >= 0.363)
        
        print(f"--- Verification Result ---")
        print(f"Is same person? {is_same_person}")
        print(f"Similarity Score: {cosine_score:.4f}")
        
        return is_same_person, cosine_score

    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return False, 0.0

# --- Test the Prototype ---
if __name__ == "__main__":
    # Your target test files inside the Project_Butler tree
    Kashton1 = "../data/faces/IMG_6844.jpeg" 
    Kashton2 = "../data/faces/IMG_7988.jpeg"
    
    verify_user(Kashton1, Kashton2)
