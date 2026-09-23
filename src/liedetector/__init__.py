# 谎言检测器（透明图形版）视频验证模块。
# 求解算法移植自参考项目 komari，现改为 DIS 稠密光流 + 粒子滤波在线跟踪（shape_session/shape_tracking），不使用神经网络。
# 弹窗定位用多尺度模板匹配（detector.py）；原 ONNX/YOLO + ByteTrack 检测线（detector 旧类、solver.py、tracker.py）已移除。
