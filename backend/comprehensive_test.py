"""Comprehensive test of the medical imaging API"""
import requests
import os

BASE_URL = 'http://localhost:5000/api/analyze'

tests = [
    # TB positive images
    ('TB Positive 1', 'data/tb/Tuberculosis/Tuberculosis-1.png', 'tuberculosis', 'Tuberculosis'),
    ('TB Positive 2', 'data/tb/Tuberculosis/Tuberculosis-10.png', 'tuberculosis', 'Tuberculosis'),
    ('TB Positive 3', 'data/tb/Tuberculosis/Tuberculosis-100.png', 'tuberculosis', 'Tuberculosis'),
    
    # Normal (TB negative) images
    ('Normal 1', 'data/tb/Normal/Normal-1.png', 'chexnet', None),  # Should go to CheXNet after TB check
    ('Normal 2', 'data/tb/Normal/Normal-2.png', 'chexnet', None),
    
    # Pneumonia images
    ('Pneumonia 1', 'data/pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg', 'chexnet', None),
    
    # Bone images  
    ('Elbow', 'data/MURA/train/XR_ELBOW/patient00011/study1_negative/image1.png', 'mura', None),
    ('Shoulder', 'data/MURA/train/XR_SHOULDER/patient00001/study1_positive/image1.png', 'mura', None),
    ('Wrist', 'data/MURA/train/XR_WRIST/patient00006/study1_positive/image1.png', 'mura', None),
]

print('='*70)
print('COMPREHENSIVE MEDICAL IMAGING API TEST')
print('='*70)

correct = 0
total = 0

for name, path, expected_model, expected_disease in tests:
    if not os.path.exists(path):
        print(f'SKIP: {name} - File not found: {path}')
        continue
    
    total += 1
    try:
        with open(path, 'rb') as f:
            r = requests.post(BASE_URL, files={'image': f})
        
        model = r.headers.get('X-Model-Used', 'unknown')
        data = r.json()
        disease = data[0]['disease'] if data else 'N/A'
        conf = data[0]['confidence'] if data else 0
        
        # Check if model matches expected
        model_ok = model == expected_model
        # Check if disease matches expected (if specified)
        disease_ok = expected_disease is None or expected_disease in disease
        
        if model_ok and disease_ok:
            status = '✅ OK'
            correct += 1
        else:
            status = '❌ WRONG'
        
        print(f'{status}: {name}')
        print(f'       Model: {model} (expected: {expected_model})')
        print(f'       Result: {disease} ({conf}%)')
        if expected_disease and expected_disease not in disease:
            print(f'       Expected disease: {expected_disease}')
        print()
        
    except Exception as e:
        print(f'❌ ERROR: {name} - {e}')
        print()

print('='*70)
print(f'ACCURACY: {correct}/{total} ({100*correct/total:.1f}%)')
print('='*70)
