"""Test TB model predictions"""
import sys
import numpy.core.multiarray
sys.modules['numpy._core'] = sys.modules['numpy.core']
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

from models.tuberculosis_inference import TuberculosisPredictor

predictor = TuberculosisPredictor()

print('Testing TB Detection Model')
print('='*60)

# Test multiple TB images
tb_images = [
    'data/tb/Tuberculosis/Tuberculosis-1.png',
    'data/tb/Tuberculosis/Tuberculosis-2.png',
    'data/tb/Tuberculosis/Tuberculosis-3.png',
]

print('\n--- TB POSITIVE IMAGES ---')
for tb_path in tb_images:
    try:
        result = predictor.predict(tb_path)
        status = 'CORRECT' if result['is_tuberculosis'] else 'WRONG'
        print(f'{status}: {tb_path}')
        print(f'  Prediction: {result["prediction"]}')
        print(f'  TB prob: {result["tuberculosis_probability"]:.4f}')
        print(f'  Normal prob: {result["normal_probability"]:.4f}')
    except Exception as e:
        print(f'ERROR: {tb_path} - {e}')

# Test multiple Normal images
normal_images = [
    'data/tb/Normal/Normal-1.png',
    'data/tb/Normal/Normal-2.png',
    'data/tb/Normal/Normal-3.png',
]

print('\n--- NORMAL IMAGES ---')
for normal_path in normal_images:
    try:
        result = predictor.predict(normal_path)
        status = 'CORRECT' if not result['is_tuberculosis'] else 'WRONG'
        print(f'{status}: {normal_path}')
        print(f'  Prediction: {result["prediction"]}')
        print(f'  TB prob: {result["tuberculosis_probability"]:.4f}')
        print(f'  Normal prob: {result["normal_probability"]:.4f}')
    except Exception as e:
        print(f'ERROR: {normal_path} - {e}')

print('\n' + '='*60)
