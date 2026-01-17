# YOLOv8 Web 目标检测展示系统

这是一个基于 Flask 的 YOLOv8 目标检测 Web 应用，提供简单直观的界面用于展示目标检测功能。

## 功能特性

- 简洁美观的 Web 界面
- 支持图片上传
- 实时目标检测
- 可视化检测结果
- 显示检测到的物体类别和置信度

## 安装依赖

```bash
pip install -r web_requirements.txt
```

## 运行应用

```bash
python web_app.py
```

应用将在 `http://localhost:5000` 启动。

## 使用方法

1. 打开浏览器访问 `http://localhost:5000`
2. 点击"选择图片"按钮上传待检测的图片
3. 点击"开始检测"按钮
4. 等待检测完成，查看结果

## 支持的图片格式

- PNG
- JPG/JPEG
- GIF
- BMP
- WebP

## 技术栈

- **后端**: Flask
- **深度学习框架**: Ultralytics YOLOv8
- **图像处理**: OpenCV
- **前端**: HTML5 + CSS3 + JavaScript

## 文件结构

```
detectele/
├── web_app.py              # Flask 应用主文件
├── templates/
│   └── index.html          # 前端页面
├── uploads/                # 上传的图片存储目录（自动创建）
├── results/                # 检测结果存储目录（自动创建）
├── yolov8n.pt             # YOLOv8 预训练模型
└── web_requirements.txt    # Web 应用依赖
```

## 注意事项

- 确保 `yolov8n.pt` 模型文件在项目根目录
- 上传文件大小限制为 16MB
- 首次运行时会自动创建 `uploads` 和 `results` 目录
