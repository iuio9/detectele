from flask import Flask, request, render_template, send_from_directory, jsonify, session
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolov8-detection-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

socketio = SocketIO(app, cors_allowed_origins="*")

# 创建必要的目录
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULT_FOLDER']).mkdir(exist_ok=True)

# 全局变量
current_model = None
current_model_path = None
inference_stop_flag = False

# 允许的文件扩展名
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def load_model(model_path):
    """加载YOLO模型"""
    global current_model, current_model_path

    if current_model is None or current_model_path != model_path:
        socketio.emit('log', {'message': f'正在加载模型: {model_path}'})
        current_model = YOLO(model_path)
        current_model_path = model_path
        socketio.emit('log', {'message': '模型加载完成'})
    return current_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/list_models', methods=['GET'])
def list_models():
    """列出可用的模型文件"""
    models = []
    base_path = os.getcwd()

    # 查找当前目录和子目录中的.pt文件
    for root, dirs, files in os.walk(base_path):
        # 跳过某些目录
        if any(skip in root for skip in ['node_modules', '.git', '__pycache__', 'venv']):
            continue
        for file in files:
            if file.endswith('.pt'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                models.append({'path': rel_path, 'name': file})

    # 默认模型
    if os.path.exists('yolov8n.pt'):
        models.insert(0, {'path': 'yolov8n.pt', 'name': 'yolov8n.pt (推荐)'})

    return jsonify(models)

@app.route('/api/results/<path:filename>')
def get_result_file(filename):
    """获取结果文件"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@socketio.on('start_inference')
def handle_inference(data):
    """处理推理请求"""
    global inference_stop_flag
    inference_stop_flag = False

    try:
        mode = data.get('mode', 'image')
        model_path = data.get('model_path', 'yolov8n.pt')
        conf_thresh = float(data.get('conf_thresh', 0.25))
        iou_thresh = float(data.get('iou_thresh', 0.45))
        img_size = int(data.get('img_size', 640))
        save_results = data.get('save_results', True)

        emit('status', {'status': '准备中', 'color': '#333'})
        emit('log', {'message': '开始推理任务...'})

        # 加载模型
        model = load_model(model_path)

        if mode == 'image':
            # 图片模式
            files = data.get('files', [])
            if not files:
                emit('error', {'message': '没有上传图片'})
                return

            total_files = len(files)
            results_list = []

            for idx, file_data in enumerate(files):
                if inference_stop_flag:
                    emit('log', {'message': '推理已停止'})
                    break

                filename = file_data['name']
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                emit('status', {'status': f'推理中 ({idx+1}/{total_files})', 'color': '#2D7A67'})
                emit('progress', {'progress': int((idx / total_files) * 100)})
                emit('log', {'message': f'正在处理: {filename}'})

                # 执行检测
                results = model(file_path, conf=conf_thresh, iou=iou_thresh, imgsz=img_size)

                # 统计检测结果
                detection_counts = {}
                total_detections = 0
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1

                    emit('log', {'message': f'检测到 {total_detections} 个目标'})
                    for cls_name, count in detection_counts.items():
                        emit('log', {'message': f'  - {cls_name}: {count} 个'})
                else:
                    emit('log', {'message': '未检测到任何目标'})

                # 保存结果图片
                if save_results:
                    result_filename = f"result_{filename}"
                    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

                    # 绘制检测结果（plot()返回BGR格式，直接保存）
                    result_img = results[0].plot()
                    cv2.imwrite(result_path, result_img)

                    # 获取检测信息（用于返回）
                    detections = []
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls_id]
                            detections.append({
                                'class': class_name,
                                'confidence': f'{conf:.2f}'
                            })

                    results_list.append({
                        'filename': result_filename,
                        'path': f'/api/results/{result_filename}',
                        'detections': detections,
                        'total': len(detections)
                    })

                    # 发送当前图片结果
                    emit('image_result', {
                        'path': f'/api/results/{result_filename}',
                        'index': idx,
                        'total': total_files
                    })

            emit('progress', {'progress': 100})
            emit('status', {'status': '已完成', 'color': '#2D7A67'})
            emit('log', {'message': f'推理完成，共处理 {len(results_list)} 张图片'})
            emit('inference_complete', {'results': results_list})

        elif mode == 'folder':
            # 文件夹模式
            folder_files = data.get('files', [])
            if not folder_files:
                emit('error', {'message': '文件夹为空'})
                return

            total_files = len(folder_files)
            results_list = []

            for idx, file_data in enumerate(folder_files):
                if inference_stop_flag:
                    emit('log', {'message': '推理已停止'})
                    break

                filename = file_data['name']
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                emit('status', {'status': f'推理中 ({idx+1}/{total_files})', 'color': '#2D7A67'})
                emit('progress', {'progress': int((idx / total_files) * 100)})
                emit('log', {'message': f'正在处理: {filename}'})

                # 执行检测
                results = model(file_path, conf=conf_thresh, iou=iou_thresh, imgsz=img_size)

                # 统计检测结果
                detection_counts = {}
                total_detections = 0
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1

                    emit('log', {'message': f'检测到 {total_detections} 个目标'})
                    for cls_name, count in detection_counts.items():
                        emit('log', {'message': f'  - {cls_name}: {count} 个'})
                else:
                    emit('log', {'message': '未检测到任何目标'})

                # 保存结果
                if save_results:
                    result_filename = f"result_{filename}"
                    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

                    result_img = results[0].plot()
                    cv2.imwrite(result_path, result_img)

                    # 获取检测信息（用于返回）
                    detections = []
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls_id]
                            detections.append({
                                'class': class_name,
                                'confidence': f'{conf:.2f}'
                            })

                    results_list.append({
                        'filename': result_filename,
                        'path': f'/api/results/{result_filename}',
                        'detections': detections,
                        'total': len(detections)
                    })
                    emit('image_result', {
                        'path': f'/api/results/{result_filename}',
                        'index': idx,
                        'total': total_files
                    })

            emit('progress', {'progress': 100})
            emit('status', {'status': '已完成', 'color': '#2D7A67'})
            emit('log', {'message': f'文件夹推理完成，共处理 {len(results_list)} 张图片'})
            emit('inference_complete', {'results': results_list})

    except Exception as e:
        emit('error', {'message': str(e)})
        emit('status', {'status': '出错', 'color': 'red'})
        emit('log', {'message': f'错误: {str(e)}'})

@socketio.on('stop_inference')
def handle_stop():
    """停止推理"""
    global inference_stop_flag
    inference_stop_flag = True
    emit('log', {'message': '正在停止推理...'})
    emit('status', {'status': '正在停止...', 'color': '#f57900'})

@socketio.on('upload_files')
def handle_upload(data):
    """处理文件上传"""
    try:
        files = data.get('files', [])
        uploaded_files = []

        print(f"[上传] 接收到 {len(files)} 个文件")

        for idx, file_data in enumerate(files):
            filename = secure_filename(file_data['name'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 保存base64编码的文件
            import base64
            file_content = base64.b64decode(file_data['data'].split(',')[1])
            with open(filepath, 'wb') as f:
                f.write(file_content)

            uploaded_files.append({'name': filename, 'path': filepath})

            if (idx + 1) % 5 == 0:
                print(f"[上传] 已保存 {idx + 1}/{len(files)} 个文件")

        print(f"[上传] 批次上传完成，共 {len(uploaded_files)} 个文件")
        emit('upload_complete', {'files': uploaded_files})

    except Exception as e:
        print(f"[上传错误] {str(e)}")
        emit('error', {'message': f'文件上传失败: {str(e)}'})

if __name__ == '__main__':
    print("=" * 60)
    print("YOLOv8 Web 检测服务启动中...")
    print("请访问: http://localhost:5000")
    print("=" * 60)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
