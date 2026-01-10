"""
Flask Backend for Early Disease Detection
Handles image uploads and routes them to appropriate models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import importlib.util
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import traceback

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.chexnet_inference import CheXNetPredictor
from models.mura_inference import MURAPredictor
from models.tuberculosis_inference import TuberculosisPredictor
from models.rsna_inference import RSNPredictor
from models.unet_inference import UNetPredictor

app = Flask(__name__)
# Configure CORS to allow frontend (running on port 8080)
CORS(app, origins=["http://localhost:8080", "http://127.0.0.1:8080"], supports_credentials=True, expose_headers=["X-Model-Used"])

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model predictors (lazy loading)
predictors = {
    'chexnet': None,
    'mura': None,
    'tuberculosis': None,
    'rsna': None,
    'unet': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_predictor(model_name):
    """Lazy load model predictors"""
    if predictors[model_name] is None:
        try:
            if model_name == 'chexnet':
                predictors[model_name] = CheXNetPredictor()
            elif model_name == 'mura':
                predictors[model_name] = MURAPredictor()
            elif model_name == 'tuberculosis':
                predictors[model_name] = TuberculosisPredictor()
            elif model_name == 'rsna':
                predictors[model_name] = RSNPredictor()
            elif model_name == 'unet':
                predictors[model_name] = UNetPredictor()
        except Exception as e:
            print(f"Error loading {model_name} model: {str(e)}")
            raise
    return predictors[model_name]

def model_checkpoint_available(model_name: str) -> bool:
    base = os.path.join(os.path.dirname(__file__), 'datasets')
    if model_name == 'chexnet':
        return os.path.isfile(os.path.join(base, 'CheXNet', 'model.pth.tar'))
    if model_name == 'mura':
        mura_models = os.path.join(base, 'DenseNet-MURA', 'models')
        if not os.path.isdir(mura_models):
            return False
        for root, _, files in os.walk(mura_models):
            if 'model.pth' in files:
                return True
        return False
    if model_name == 'tuberculosis':
        if importlib.util.find_spec('tensorflow') is None:
            return False
        tb_base = os.path.join(base, 'TuberculosisNet')
        candidates = [
            os.path.join(tb_base, 'models', 'Baseline'),
            os.path.join(tb_base, 'TB-Net')
        ]
        for d in candidates:
            if os.path.isdir(d):
                files = os.listdir(d)
                has_meta = any(f.endswith('.meta') for f in files)
                has_index = any(f.endswith('.index') for f in files)
                has_data = any('.data-' in f for f in files)
                if has_meta and has_index and has_data:
                    return True
        return False
    if model_name == 'unet':
        return os.path.isfile(os.path.join(base, 'UNet', 'MODEL.pth'))
    if model_name == 'rsna':
        return os.path.isdir(os.path.join(base, 'rsna18'))
    return False

def unavailable_model_result(model_name: str):
    return [{
        'disease': 'Model Unavailable',
        'confidence': 0,
        'status': 'warning',
        'description': f'Required checkpoint files for {model_name} were not found under backend/datasets.',
        'regions': []
    }]

def model_error_result(model_name: str, message: str):
    return [{
        'disease': 'Model Error',
        'confidence': 0,
        'status': 'warning',
        'description': f'{model_name}: {message}',
        'regions': []
    }]

def infer_scan_type_from_image(filepath: str, ext: str) -> str:
    if ext == 'dcm':
        if importlib.util.find_spec('pydicom') is not None:
            try:
                import pydicom
                ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                modality = str(getattr(ds, 'Modality', '')).upper()
                if modality == 'CT':
                    return 'ct'
                if modality in {'CR', 'DX'}:
                    return 'chest'
            except Exception:
                pass
        return 'unknown'
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'unknown'
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32) / 255.0

    left = img_f[:, :128]
    right = np.fliplr(img_f[:, 128:])
    l = left.reshape(-1)
    r = right.reshape(-1)
    denom = (np.std(l) * np.std(r))
    symmetry = float(np.corrcoef(l, r)[0, 1]) if denom > 1e-6 else 0.0
    if not np.isfinite(symmetry):
        symmetry = 0.0

    edges = cv2.Canny(img, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    chest_score = 0.7 * max(symmetry, 0.0) + 0.3 * (1.0 - edge_density)
    bone_score = 0.7 * edge_density + 0.3 * (1.0 - max(symmetry, 0.0))

    if chest_score > 0.58 and chest_score > bone_score + 0.05:
        return 'chest'
    if bone_score > 0.58 and bone_score > chest_score + 0.05:
        return 'bone'
    return 'unknown'

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Early Disease Detection API is running'
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'models': [
            {
                'name': 'chexnet',
                'available': model_checkpoint_available('chexnet'),
                'description': 'Chest X-ray disease detection (14 diseases)',
                'diseases': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                           'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                           'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            },
            {
                'name': 'mura',
                'available': model_checkpoint_available('mura'),
                'description': 'Musculoskeletal radiographs abnormality detection'
            },
            {
                'name': 'tuberculosis',
                'available': model_checkpoint_available('tuberculosis'),
                'description': 'Tuberculosis detection from chest X-rays'
            },
            {
                'name': 'rsna',
                'available': model_checkpoint_available('rsna'),
                'description': 'RSNA Intracranial Hemorrhage Detection'
            },
            {
                'name': 'unet',
                'available': model_checkpoint_available('unet'),
                'description': 'Medical image segmentation'
            }
        ]
    })

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """Predict endpoint for a specific model"""
    if model_name not in predictors:
        return jsonify({'error': f'Model {model_name} not found'}), 404

    if not model_checkpoint_available(model_name):
        resp = jsonify(unavailable_model_result(model_name))
        resp.headers['X-Model-Used'] = model_name
        return resp, 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictor and make prediction
        predictor = get_predictor(model_name)
        result = predictor.predict(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        resp = jsonify({
            'success': True,
            'model': model_name,
            'result': result
        })
        resp.headers['X-Model-Used'] = model_name
        return resp
    
    except Exception as e:
        # Clean up on error
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint for frontend - intelligently routes between TB and CheXNet models"""
    if 'image' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Handle both 'image' (frontend) and 'file' (alternative) field names
    file = request.files.get('image') or request.files.get('file')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported: PNG, JPG, JPEG, DICOM (.dcm)'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        scan_type = (request.form.get('scan_type') or request.args.get('scan_type') or 'auto').strip().lower()
        requested_model = (request.form.get('model') or request.args.get('model') or '').strip().lower()
        filename_lower = filename.lower()
        ext = os.path.splitext(filename_lower)[1].lstrip('.')

        if scan_type == 'auto':
            inferred_scan_type = infer_scan_type_from_image(filepath, ext)
            if inferred_scan_type == 'unknown':
                os.remove(filepath)
                resp = jsonify([{
                    'disease': 'Scan Type Required',
                    'confidence': 0,
                    'status': 'warning',
                    'description': 'Unable to confidently determine scan type automatically. Please select Chest X-Ray or Bone X-Ray in the UI.',
                    'regions': []
                }])
                resp.headers['X-Model-Used'] = 'none'
                return resp, 422
            scan_type = inferred_scan_type

        bone_keywords = ['wrist', 'hand', 'elbow', 'shoulder', 'humerus', 'finger', 'forearm', 'ankle', 'foot', 'knee', 'hip', 'bone', 'mura']
        ct_keywords = ['ct', 'brain', 'head', 'intracranial', 'rsna']

        inferred_model = None
        if ext == 'dcm':
            inferred_model = 'rsna'
        elif any(k in filename_lower for k in bone_keywords):
            inferred_model = 'mura'
        elif any(k in filename_lower for k in ct_keywords):
            inferred_model = 'rsna'

        if scan_type == 'bone':
            selected_model = 'mura'
        elif scan_type == 'ct':
            selected_model = 'rsna'
        elif scan_type == 'chest':
            selected_model = None
        else:
            selected_model = requested_model if requested_model in predictors else inferred_model

        if selected_model == 'rsna':
            if not model_checkpoint_available('rsna'):
                os.remove(filepath)
                resp = jsonify(unavailable_model_result('rsna'))
                resp.headers['X-Model-Used'] = 'rsna'
                return resp
            predictor = get_predictor('rsna')
            result = predictor.predict_for_frontend(filepath)
            os.remove(filepath)
            resp = jsonify(result)
            resp.headers['X-Model-Used'] = 'rsna'
            return resp

        if selected_model == 'mura':
            if not model_checkpoint_available('mura'):
                os.remove(filepath)
                resp = jsonify(unavailable_model_result('mura'))
                resp.headers['X-Model-Used'] = 'mura'
                return resp
            try:
                predictor = get_predictor('mura')
                result = predictor.predict_for_frontend(filepath)
                os.remove(filepath)
                resp = jsonify(result)
                resp.headers['X-Model-Used'] = 'mura'
                return resp
            except Exception as e:
                os.remove(filepath)
                resp = jsonify(model_error_result('mura', str(e)))
                resp.headers['X-Model-Used'] = 'mura'
                return resp, 500
        
        # Try TB model first (fast inference)
        try:
            if not model_checkpoint_available('tuberculosis'):
                raise Exception('Tuberculosis model checkpoint not found')
            tb_predictor = get_predictor('tuberculosis')
            tb_raw = tb_predictor.predict(filepath)
            
            # If TB is detected with high confidence, return TB results
            if 'is_tuberculosis' in tb_raw and tb_raw['is_tuberculosis']:
                tb_prob = tb_raw.get('tuberculosis_probability', 0.0)
                if tb_prob > 0.55:
                    result = tb_predictor.predict_for_frontend(filepath)
                    os.remove(filepath)
                    resp = jsonify(result)
                    resp.headers['X-Model-Used'] = 'tuberculosis'
                    return resp
        except Exception as tb_error:
            app.logger.warning(f"TB model failed, falling back to CheXNet: {tb_error}")

        if scan_type != 'chest':
            try:
                if model_checkpoint_available('mura'):
                    mura_predictor = get_predictor('mura')
                    mura_raw = mura_predictor.predict(filepath)
                    mura_prob = float(mura_raw.get('abnormality_probability', 0.0))
                    if mura_prob > 0.65:
                        result = mura_predictor.predict_for_frontend(filepath)
                        os.remove(filepath)
                        resp = jsonify(result)
                        resp.headers['X-Model-Used'] = 'mura'
                        return resp
            except Exception as mura_error:
                app.logger.warning(f"MURA model check failed, falling back to CheXNet: {mura_error}")
        
        # Use CheXNet for comprehensive analysis (default)
        if not model_checkpoint_available('chexnet'):
            os.remove(filepath)
            resp = jsonify(unavailable_model_result('chexnet'))
            resp.headers['X-Model-Used'] = 'chexnet'
            return resp
        try:
            predictor = get_predictor('chexnet')
            result = predictor.predict_for_frontend(filepath)
        except Exception as e:
            os.remove(filepath)
            resp = jsonify(model_error_result('chexnet', str(e)))
            resp.headers['X-Model-Used'] = 'chexnet'
            return resp, 500
        
        # Clean up uploaded file
        os.remove(filepath)
        
        resp = jsonify(result)
        resp.headers['X-Model-Used'] = 'chexnet'
        return resp
    
    except Exception as e:
        # Clean up on error
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(filepath):
            os.remove(filepath)
        
        app.logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """Predict using all models"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = {}
        
        # Run predictions for all models
        for model_name in predictors.keys():
            try:
                if not model_checkpoint_available(model_name):
                    results[model_name] = {'error': unavailable_model_result(model_name)[0]['description']}
                    continue
                predictor = get_predictor(model_name)
                results[model_name] = predictor.predict(filepath)
            except Exception as e:
                results[model_name] = {
                    'error': str(e)
                }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        # Clean up on error
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print("Starting Early Disease Detection API...")
    print("Available models: chexnet, mura, tuberculosis, rsna, unet")
    print("Frontend endpoint: /api/analyze")
    print("Backend running on: http://0.0.0.0:5000")
    print("CORS enabled for: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

