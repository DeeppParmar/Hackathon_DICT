
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CheXNet'))

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = 14

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        try:
            self.densenet121 = torchvision.models.densenet121(weights=None)
        except TypeError:
            self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class CheXNetPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'CheXNet', 'model.pth.tar'
        )
        self.load_model()
    
    def load_model(self):
        try:
            self.model = DenseNet121(N_CLASSES)
            
            if os.path.isfile(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix and handle key format conversion
                # Old torchvision: norm.1, conv.1  | New torchvision: norm1, conv1
                import re
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Remove 'module.' prefix
                    if k.startswith('module.'):
                        name = k[7:]
                    else:
                        name = k
                    
                    # Convert old-style keys (norm.1, conv.1) to new-style (norm1, conv1)
                    # Pattern: denselayer1.norm.1 -> denselayer1.norm1
                    name = re.sub(r'\.norm\.(\d+)', r'.norm\1', name)
                    name = re.sub(r'\.conv\.(\d+)', r'.conv\1', name)
                    
                    new_state_dict[name] = v
                
                load_res = self.model.load_state_dict(new_state_dict, strict=False)
                
                if getattr(load_res, 'missing_keys', None):
                    critical_missing = [k for k in load_res.missing_keys 
                                       if 'num_batches_tracked' not in k]
                    if len(critical_missing) > 0:
                        print(f"Warning: CheXNet checkpoint had {len(critical_missing)} missing keys")
                        # Print first few for debugging
                        print(f"First missing keys: {critical_missing[:5]}")
                
                print(f"Loaded CheXNet model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ CheXNet model ready on device: {self.device}")
        except Exception as e:
            print(f"Error loading CheXNet model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def preprocess_image(self, image_path):
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    def predict(self, image_path):
        try:
            image_tensor = self.preprocess_image(image_path)
            
            input_var = image_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_var)
                probabilities = output.cpu().numpy()[0]
            
            results = {}
            detected_diseases = []
            
            for i, class_name in enumerate(CLASS_NAMES):
                prob = float(probabilities[i])
                results[class_name] = prob
                if prob > 0.5:  # Threshold for positive detection
                    detected_diseases.append({
                        'disease': class_name,
                        'probability': prob
                    })
            
            detected_diseases.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'model': 'CheXNet',
                'description': 'Chest X-ray disease detection',
                'all_probabilities': results,
                'detected_diseases': detected_diseases,
                'num_detections': len(detected_diseases)
            }
        
        except Exception as e:
            raise Exception(f"Error in CheXNet prediction: {str(e)}")
    
    def predict_for_frontend(self, image_path):
        try:
            raw_result = self.predict(image_path)
            
            disease_descriptions = {
                'Atelectasis': 'Collapsed or airless lung tissue detected. May indicate lung compression or obstruction.',
                'Cardiomegaly': 'Enlarged heart silhouette observed. Could indicate cardiac enlargement or pericardial effusion.',
                'Effusion': 'Fluid accumulation in the pleural space detected. Possible pleural effusion.',
                'Infiltration': 'Abnormal substances in lung tissue identified. May indicate inflammation or fluid.',
                'Mass': 'Abnormal mass or nodule detected in lung tissue. Requires further evaluation.',
                'Nodule': 'Small nodule detected in lung parenchyma. Monitor for changes.',
                'Pneumonia': 'Inflammation of lung tissue detected. Signs of infection present.',
                'Pneumothorax': 'Air in pleural space detected. Lung collapse possible.',
                'Consolidation': 'Lung tissue consolidation observed. May indicate pneumonia or other conditions.',
                'Edema': 'Fluid accumulation in lung tissue detected. Possible pulmonary edema.',
                'Emphysema': 'Destructive lung changes detected. Possible chronic obstructive pulmonary disease.',
                'Fibrosis': 'Scarring of lung tissue detected. Chronic lung disease indicators present.',
                'Pleural_Thickening': 'Thickening of pleural membranes observed. Chronic inflammation possible.',
                'Hernia': 'Diaphragmatic hernia or other hernia detected in chest cavity.'
            }
            
            analysis_results = []
            
            all_probs = raw_result.get('all_probabilities', {})
            
        # Sort by probability
            sorted_diseases = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            detected_above_threshold = [d for d, prob in sorted_diseases if prob > 0.5]
            
            if not detected_above_threshold:
                analysis_results.append({
                    'disease': 'Healthy Scan',
                    'confidence': int((1.0 - sorted_diseases[0][1]) * 100) if sorted_diseases else 95,
                    'status': 'healthy',
                    'description': 'No significant abnormalities detected. Lung fields appear clear. Normal cardiac silhouette.',
                    'regions': []
                })
                for disease, prob in sorted_diseases[:3]:
                    analysis_results.append({
                        'disease': disease.replace('_', ' '),
                        'confidence': int(prob * 100),
                        'status': 'healthy',
                        'description': f'No significant {disease.replace("_", " ").lower()} indicators detected.'
                    })
            else:
                for disease, prob in sorted_diseases:
                    if prob > 0.5:
                        status = 'critical' if prob > 0.75 else 'warning'
                        confidence = int(prob * 100)
                        description = disease_descriptions.get(disease, f'{disease.replace("_", " ")} indicators detected.')
                        
                        regions = []
                        if 'Lobe' not in description:
                            if disease in ['Pneumonia', 'Consolidation']:
                                regions = ['Right Lower Lobe', 'Left Lower Lobe']
                            elif disease in ['Effusion', 'Pleural_Thickening']:
                                regions = ['Pleural Space']
                            elif disease == 'Cardiomegaly':
                                regions = ['Mediastinum', 'Cardiac Silhouette']
                        
                        analysis_results.append({
                            'disease': disease.replace('_', ' '),
                            'confidence': confidence,
                            'status': status,
                            'description': description,
                            'regions': regions
                        })
                        
                        if len([r for r in analysis_results if r['status'] != 'healthy']) >= 3:
                            break
                
                for disease, prob in sorted_diseases:
                    if prob <= 0.5 and len(analysis_results) < 3:
                        analysis_results.append({
                            'disease': disease.replace('_', ' '),
                            'confidence': int(prob * 100),
                            'status': 'healthy',
                            'description': f'No significant {disease.replace("_", " ").lower()} indicators detected.'
                        })
            
            # Ensure we have at least 1 result, max 3
            return analysis_results[:3]
        
        except Exception as e:
            raise Exception(f"Error formatting CheXNet results for frontend: {str(e)}")

