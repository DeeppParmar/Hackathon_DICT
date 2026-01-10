"""
MURA (Musculoskeletal Radiographs) Inference Module
Detects abnormalities in musculoskeletal X-rays
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add MURA dataset path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'DenseNet-MURA'))

try:
    from densenet import densenet169
except ImportError:
    # Fallback if densenet module not available
    densenet169 = None

class MURAPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.output_is_probability = False
        base_models_dir = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'DenseNet-MURA', 'models'
        )
        preferred = os.path.join(base_models_dir, 'XR_WRIST', 'model.pth')
        if os.path.isfile(preferred):
            self.model_path = preferred
        else:
            found = None
            for root, _, files in os.walk(base_models_dir):
                if 'model.pth' in files:
                    found = os.path.join(root, 'model.pth')
                    break
            self.model_path = found or preferred
        self.load_model()
    
    def load_model(self):
        """Load the MURA DenseNet model"""
        try:
            if densenet169 is None:
                # Use torchvision DenseNet as fallback
                import torchvision.models as models
                try:
                    self.model = models.densenet169(weights=None)
                except TypeError:
                    self.model = models.densenet169(pretrained=False)
                self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
                self.output_is_probability = False
            else:
                self.model = densenet169(pretrained=False)
                self.output_is_probability = True
            
            if os.path.isfile(self.model_path):
                state = torch.load(self.model_path, map_location=self.device)
                if isinstance(state, dict):
                    if 'model_state_dict' in state:
                        state = state['model_state_dict']
                    elif 'state_dict' in state:
                        state = state['state_dict']
                if isinstance(state, dict):
                    state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
                load_res = self.model.load_state_dict(state, strict=False)
                if getattr(load_res, 'missing_keys', None):
                    if len(load_res.missing_keys) > 0:
                        raise RuntimeError(f"MURA checkpoint missing keys: {load_res.missing_keys[:10]}")
                print(f"Loaded MURA model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading MURA model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image for MURA"""
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)
    
    def predict(self, image_path):
        """Predict abnormality in musculoskeletal X-ray"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                out = output
                if isinstance(out, (list, tuple)):
                    out = out[0]
                out = out.view(-1)[0]
                if self.output_is_probability:
                    probability = float(out.cpu().numpy())
                else:
                    probability = float(torch.sigmoid(out).cpu().numpy())
            
            is_abnormal = probability > 0.5
            
            return {
                'model': 'MURA',
                'description': 'Musculoskeletal radiographs abnormality detection',
                'is_abnormal': bool(is_abnormal),
                'abnormality_probability': float(probability),
                'normal_probability': float(1 - probability),
                'prediction': 'Abnormal' if is_abnormal else 'Normal'
            }
        
        except Exception as e:
            raise Exception(f"Error in MURA prediction: {str(e)}")

    def predict_for_frontend(self, image_path):
        """Predict and format results for frontend React component"""
        try:
            raw_result = self.predict(image_path)

            prob = float(raw_result.get('abnormality_probability', 0.0))
            is_abnormal = bool(raw_result.get('is_abnormal', False))

            if is_abnormal:
                status = 'critical' if prob > 0.75 else 'warning'
                primary = {
                    'disease': 'Musculoskeletal Abnormality',
                    'confidence': int(prob * 100),
                    'status': status,
                    'description': 'Abnormality detected in musculoskeletal radiograph. Please consult an orthopedic specialist for confirmation.',
                    'regions': []
                }
                secondary = {
                    'disease': 'Normal Tissue',
                    'confidence': int((1.0 - prob) * 100),
                    'status': 'healthy',
                    'description': 'Some regions appear within normal limits.',
                    'regions': []
                }
                return [primary, secondary]

            return [{
                'disease': 'Healthy Scan (Bone)',
                'confidence': int((1.0 - prob) * 100),
                'status': 'healthy',
                'description': 'No significant musculoskeletal abnormality detected.',
                'regions': []
            }, {
                'disease': 'Abnormality Risk',
                'confidence': int(prob * 100),
                'status': 'healthy',
                'description': 'Low abnormality probability.',
                'regions': []
            }]

        except Exception as e:
            return [{
                'disease': 'Analysis Error',
                'confidence': 0,
                'status': 'warning',
                'description': f'Error analyzing image for musculoskeletal abnormality: {str(e)}',
                'regions': []
            }]

