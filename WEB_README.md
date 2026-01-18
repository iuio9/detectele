# YOLOv8 Web 目标检测展示系统

这是一个基于 Flask + WebSocket 的 YOLOv8 目标检测 Web 应用，界面布局和功能完全模仿 PyQt5 桌面版，提供专业的目标检测体验。

## 功能特性

- **完整的PyQt5界面复刻**：左右分栏布局，与桌面版完全一致
- **多种推理模式**：支持图片模式和文件夹模式
- **实时进度反馈**：WebSocket实时推送检测进度和日志
- **参数可调**：置信度阈值、IoU阈值、图像尺寸均可自定义
- **图片浏览器**：支持浏览所有检测结果，上一张/下一张切换
- **终端输出**：实时显示检测日志和状态信息
- **模型自动发现**：自动扫描并列出所有可用的.pt模型文件
- **暗色主题**：专业的VSCode风格界面设计

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
