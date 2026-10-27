# -*- coding: utf-8 -*-
"""D:\\lie 视觉管线移植：YOLO+OC-SORT 跟踪 → 旋转/运动学证据 → 六态状态机。

除以下三类适配外与 D:\\lie 原始实现逐文件一致：
1. import 路径改为本包（src.liedetector.vision.*）
2. config.py 资源路径指向宿主项目（weights/best.pt、cache/trt_calib，ocsort.yaml/models 随包）
3. Detector.reset_tracking 会话复位（Pipeline 构造自动触发，warm Detector 跨解题会话复用所需）
appearance / neural_evidence 两条证据通道保留代码但默认关闭（与 D:\\lie 默认一致）。
"""
