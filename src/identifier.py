import cv2
import numpy as np

class Identifier:
    def __init__(self, model_path: str, match_threshold: float = 0.363, runner_up_margin: float = 0.10):
        """
        Initializes the stateless SFace embedding and vector matching engine.
        
        Args:
            model_path: System string path to the face_recognition_sface.onnx weights file.
            match_threshold: SFace cosine reference value (initial value: 0.363).
            runner_up_margin: Minimum separation score required between best and next-best match.
        """
        self.match_threshold = match_threshold
        self.runner_up_margin = runner_up_margin
        
        # Instantiate SFace via OpenCV's native FaceRecognizerSF class
        # It requires the model file path and an optional config string
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=model_path,
            config=""
        )

    def extract_embedding(self, aligned_crop: np.ndarray) -> np.ndarray:
        """
        Extracts a 128-float32 embedding vector from a 112x112 aligned face crop.
        
        This method is purely stateless and acts entirely in RAM.
        """
        # SFace expects a specific image matrix format; feature extraction runs local inference
        embedding = self.recognizer.feature(aligned_crop)
        
        # OpenCV returns a 1x128 or 1x512 matrix. We flatten it to a clean 1D array
        return embedding.flatten()

    def _calculate_cosine_similarity(self, vecA: np.ndarray, vecB: np.ndarray) -> float:
        """Computes the cosine similarity between two 128-float32 vectors."""
        dot_product = np.dot(vecA, vecB)
        normA = np.linalg.norm(vecA)
        normB = np.linalg.norm(vecB)
        
        if normA == 0 or normB == 0:
            return 0.0
            
        return float(dot_product / (normA * normB))

    def identify(self, crop: np.ndarray, subjects_references: dict) -> tuple:
        """
        Takes a raw crop, extracts its vector, and performs a best-of-5 evaluation
        against reference embeddings supplied dynamically by the Runner.
        
        Args:
            crop: Aligned face crop matrix from the Splicer.
            subjects_references: A dict mapping a subject_name to a list of its stored 
                                 embedding vectors. Format: {'Alice': [vec1, vec2, ...]}
                                 
        Returns:
            A tuple of: (matched_identity_string or "Unknown", match_score, runner_up_score)
        """
        # Step 1: Extract feature vector from the current face crop
        current_embedding = self.extract_embedding(crop)
        
        if not subjects_references:
            # If database cache is completely empty, it is mathematically Unknown
            return "Unknown", 0.0, 0.0

        subject_scores = {}

        # Step 2: Compute Best-of-5 cosine similarity for each individual subject profile
        for subject_name, vectors in subjects_references.items():
            max_similarity = -1.0
            for ref_vector in vectors:
                similarity = self._calculate_cosine_similarity(current_embedding, ref_vector)
                if similarity > max_similarity:
                    max_similarity = similarity
            
            subject_scores[subject_name] = max_similarity

        # Step 3: Sort subjects to find the absolute best match and the runner-up
        sorted_scores = sorted(subject_scores.items(), key=lambda item: item[1], reverse=True)
        
        best_subject, best_score = sorted_scores[0]
        runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        if best_score < self.match_threshold:
            return "Unknown", best_score, runner_up_score
            
        # Rule B: Sibling Lockout Protection.
        # Define a line of "Absolute Certainty". If the AI scores a 0.55 or higher, 
        # it is confident enough to bypass the runner-up margin entirely.
        CERTAIN_MATCH_THRESHOLD = 0.55  
        
        if best_score < CERTAIN_MATCH_THRESHOLD:
            # We are in the "Fuzzy Match Zone" (between 0.363 and 0.55). 
            # Here, we MUST enforce the margin check to prevent dangerous misidentifications.
            if (best_score - runner_up_score) < self.runner_up_margin:
                # The score is too close to a lookalike. Flag as Unknown to trigger human triage!
                return "Unknown", best_score, runner_up_score

        # If it passes Rule A and clears Rule B (or completely bypassed it via certainty),
        # return the verified identity to the Runner to drop the lock relay.
        return best_subject, best_score, runner_up_score

