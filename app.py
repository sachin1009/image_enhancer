from flask import Flask, request, render_template, jsonify, send_file
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from rembg import remove
import uuid
import time

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

def get_unique_filename(extension):
    """Generate a unique filename based on timestamp and UUID"""
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique_id}.{extension}"

def process_image(image_path, params, remove_bg=False):
    """Process the image with the given parameters"""
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Remove background if requested
        if remove_bg:
            # Convert to RGB if it's not (e.g., if it's RGBA)
            if img.mode != 'RGB' and img.mode != 'RGBA':
                img = img.convert('RGB')
            
            # Remove background
            img = remove(img)
        
        # Apply enhancements based on parameters
        if 'brightness' in params:
            brightness = float(params['brightness'])
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        if 'contrast' in params:
            contrast = float(params['contrast'])
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        if 'sharpness' in params:
            sharpness = float(params['sharpness'])
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness)
        
        if 'color' in params:
            color = float(params['color'])
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(color)
        
        # Apply noise reduction if specified
        if 'noise_reduction' in params and float(params['noise_reduction']) > 0:
            # Convert PIL image to OpenCV format for advanced processing
            if img.mode == 'RGBA':
                # Save the alpha channel
                r, g, b, a = img.split()
                img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                
                # Apply noise reduction
                strength = int(float(params['noise_reduction']) * 10)
                img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, strength, strength, 7, 21)
                
                # Convert back to PIL and restore alpha
                img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                img.putalpha(a)
            else:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # Apply noise reduction
                strength = int(float(params['noise_reduction']) * 10)
                img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, strength, strength, 7, 21)
                img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Scale up the image if requested (basic upscaling)
        if 'upscale' in params and params['upscale'] == 'true':
            # Double the resolution using Lanczos resampling
            width, height = img.size
            img = img.resize((width * 2, height * 2), Image.LANCZOS)
        
        # Save the processed image
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], get_unique_filename('png'))
        if img.mode == 'RGBA':
            img.save(output_path, format='PNG')
        else:
            img.convert('RGB').save(output_path, format='PNG')
        
        return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400
    
    if file:
        # Save the uploaded file
        filename = get_unique_filename(file.filename.split('.')[-1])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get processing parameters
        params = {
            'brightness': request.form.get('brightness', '1.0'),
            'contrast': request.form.get('contrast', '1.0'),
            'sharpness': request.form.get('sharpness', '1.0'),
            'color': request.form.get('color', '1.0'),
            'noise_reduction': request.form.get('noise_reduction', '0'),
            'upscale': request.form.get('upscale', 'false')
        }
        
        # Check if background removal is requested
        remove_bg = request.form.get('remove_bg', 'false') == 'true'
        
        # Process the image
        processed_path = process_image(filepath, params, remove_bg)
        
        if processed_path:
            # Convert the processed image to base64 for display
            with open(processed_path, 'rb') as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'success': True, 
                'image': encoded_img,
                'download_path': f'/download/{os.path.basename(processed_path)}'
            })
        else:
            return jsonify({'error': 'Image processing failed'}), 500
    
    return jsonify({'error': 'Unknown error occurred'}), 500

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), 
                     as_attachment=True, 
                     download_name=filename)

# Cleanup old files periodically (optional for production)
@app.before_request
def cleanup_old_files():
    # Add cleanup logic here if needed
    pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)