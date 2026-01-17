from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import base64
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 创建必要的目录
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULT_FOLDER']).mkdir(exist_ok=True)

# 加载YOLOv8模型
model = None
MODEL_PATH = 'yolov8n.pt'

def load_model():
    global model
    if model is None:
        print(f"加载模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("模型加载完成")

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return {'error': '没有上传文件'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': '没有选择文件'}, 400

    if not allowed_file(file.filename):
        return {'error': '不支持的文件格式'}, 400

    # 保存上传的文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 执行检测
    load_model()
    results = model(filepath, conf=0.25, iou=0.45)

    # 保存检测结果图片
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

    # 绘制检测结果
    result_img = results[0].plot()
    import cv2
    cv2.imwrite(result_path, result_img)

    # 获取检测信息
    detections = []
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            detections.append({
                'class': class_name,
                'confidence': f'{conf:.2f}'
            })

    return {
        'success': True,
        'result_image': f'/results/{result_filename}',
        'detections': detections,
        'total_objects': len(detections)
    }

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("YOLOv8 Web 检测服务启动中...")
    print("请访问: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
