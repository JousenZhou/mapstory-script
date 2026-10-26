# -*- coding: utf-8 -*-
"""D:\\lie 视觉管线移植：YOLO+OC-SORT 跟踪 → 旋转/运动学证据 → 六态状态机。

除以下适配外与 D:\\lie 原始实现逐文件一致：
1. import 路径改为本包（src.liedetector.vision.*）
2. config.py 资源路径指向宿主项目（weights/best.pt、cache/trt_calib，ocsort.yaml/models 随包）
3. Detector.reset_tracking 会话复位（Pipeline 构造自动触发，warm Detector 跨会话复用所需）
4. flow_tracker.py + flow_session.py：宿主新增的光流主架构（D:\\lie 无此模块）——
   倒计时白目标锁定后身份交给 LK 光流像素携带（环带采样描边点），YOLO 降级为周期校验
   （吸附杀漂移）与丢失重获（try_recover 四路证据 + 光流末位锚）。FLOW_PRIMARY=False
   时 solver/验证页签走原 Pipeline 路径，行为与 D:\\lie 完全一致。
appearance / neural_evidence 两条证据通道保留代码但默认关闭（与 D:\\lie 默认一致）。
"""
