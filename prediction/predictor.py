import os
import joblib
import numpy as np

class CropPredictor:
    """
    Service for making crop predictions using the trained XGBoost model.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Set default path relative to this file's location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models")
            
        self.model_path = os.path.join(model_dir, "crop_model.joblib")
        self.scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.encoder_path = os.path.join(model_dir, "label_encoder.joblib")
        
        self.model = None
        self.scaler = None
        self.encoder = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the serialized model, scaler, and label encoder."""
        if not all(os.path.exists(p) for p in [self.model_path, self.scaler_path, self.encoder_path]):
            raise FileNotFoundError("Model artifacts not found. Please run the training script first.")
            
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.encoder = joblib.load(self.encoder_path)

    def predict(self, feature_vector):
        """
        Predicts the best crop based on the input feature vector.
        
        Args:
            feature_vector (list or np.array): [N, P, K, temp, humidity, pH, rainfall]
            
        Returns:
            dict: Predicted crop name and top 3 suggestions with probabilities.
        """
        # Ensure 2D array for scaler and model
        features = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Get probabilities for all classes
        probs = self.model.predict_proba(scaled_features)[0]
        
        # Get top 5 indices
        top_indices = np.argsort(probs)[-5:][::-1]
        
        # Decode crop names
        top_crops = self.encoder.inverse_transform(top_indices)
        top_probs = probs[top_indices]
        
        results = []
        for crop, prob in zip(top_crops, top_probs):
            results.append({
                "crop": crop,
                "confidence": round(float(prob) * 100, 2)
            })
            
        return {
            "prediction": results[0]["crop"],
            "confidence": results[0]["confidence"],
            "top_suggestions": results
        }
