import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque
import torch

class TennisTracker:
    def __init__(self, frame_size, 
                 heatmap_thresh=0.4, 
                 min_ball_size=8,
                 max_lost=30,
                 reid_distance=80):
        """
        修复后的网球跟踪器
        :param frame_size: 视频帧尺寸 (width, height)
        """
        # 初始化跟踪参数
        self.width, self.height = frame_size
        self.is_tracking = False
        self.lost_count = 0
        self.max_lost = max_lost
        self.reid_distance = reid_distance
        
        # 检测参数
        self.heatmap_thresh = heatmap_thresh
        self.min_ball_size = min_ball_size
        
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self._init_kalman()
        
        # 轨迹管理
        self.history = deque(maxlen=30)
        self.smooth_pos = None
        self.predicted_pos = None

    def _init_kalman(self):
        """初始化卡尔曼滤波器参数"""
        dt = 1.0
        self.kf.F = np.array([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
        self.kf.P *= 100
        self.kf.Q = np.eye(4) * 0.1
        self.kf.R = np.eye(2) * 5

    def update(self, heatmap):
        """核心更新方法（只接受heatmap参数）"""
        # 持续预测
        if self.is_tracking:
            self.kf.predict()
            self.predicted_pos = self.kf.x[:2].flatten()
            
        # 检测目标
        candidates = self._detect_candidates(heatmap)
        detection = self._select_target(candidates)
        
        # 更新跟踪状态
        if detection is not None:
            self._update_tracking(detection)
        elif self.is_tracking:
            self.lost_count += 1
            
        # 处理目标丢失
        if self.lost_count > self.max_lost:
            self._reset_tracking()
            
        return self.status

    def _detect_candidates(self, heatmap):
        """检测候选目标"""
        _, thresh = cv2.threshold(heatmap, self.heatmap_thresh, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_ball_size:
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                candidates.append((cx, cy))
        return candidates

    def _select_target(self, candidates):
        """选择跟踪目标"""
        if not candidates:
            return None
            
        if self.is_tracking and self.predicted_pos is not None:
            # 使用欧氏距离匹配
            return min(candidates, key=lambda c: np.linalg.norm(np.array(c)-self.predicted_pos))
        return candidates[0]  # 选择第一个候选

    def _update_tracking(self, detection):
        """更新跟踪状态"""
        x, y = detection
        if self.is_tracking:
            distance = np.linalg.norm(np.array(detection) - self.predicted_pos)
            if distance < self.reid_distance:
                self.kf.update(np.array([x, y]))
                self.lost_count = 0
                self._update_history((x, y))
            else:
                self.lost_count += 1
        else:
            self._init_tracking(x, y)

    def _init_tracking(self, x, y):
        """初始化跟踪"""
        self.kf.x = np.array([x, y, 0, 0], dtype=np.float32).reshape(4, 1)
        self.is_tracking = True
        self.lost_count = 0
        self.history.clear()
        self._update_history((x, y))

    def _update_history(self, pos):
        """更新轨迹"""
        self.history.append(pos)
        # 指数平滑
        # self.smooth_pos = pos if self.smooth_pos is None else 0.8*self.smooth_pos + 0.2*np.array(pos)
        self.smooth_pos = np.array(pos) if self.smooth_pos is None else 0.8 * np.array(self.smooth_pos) + 0.2 * np.array(pos)

    def _reset_tracking(self):
        """重置跟踪器"""
        self.is_tracking = False
        self.history.clear()
        self.smooth_pos = None
        self.predicted_pos = None

    @property
    def status(self):
        """生成状态信息"""
        valid = self.is_tracking and len(self.history) > 0
        pos = self.smooth_pos if valid else self.predicted_pos
        
        # 计算动态框大小
        box_size = 20 + 2*np.linalg.norm(self.kf.x[2:]) if valid else 20
        box_size = np.clip(box_size, 15, 50)
        
        if pos is not None:
            x, y = pos
            x = np.clip(x, 0, self.width)
            y = np.clip(y, 0, self.height)
            box = (
                int(x - box_size/2),
                int(y - box_size/2),
                int(x + box_size/2),
                int(y + box_size/2)
            )
        else:
            box = (0, 0, 0, 0)
            
        return {
            "tracking": valid,
            "predicted": self.lost_count > 0,
            "box": box,
            "position": (int(x), int(y)) if pos is not None else (0, 0),
            "history": list(self.history)[-10:]  # 返回最近10个点
        }

def visualize(frame, status, width, height):
    """可视化函数"""
    display = frame.copy()
    
    # 绘制跟踪框
    if status["box"][2] > 0 and status["box"][3] > 0:
        color = (0, 255, 0) if not status["predicted"] else (0, 0, 255)
        
        # 半透明填充
        overlay = display.copy()
        cv2.rectangle(overlay, 
                      (status["box"][0], status["box"][1]),
                      (status["box"][2], status["box"][3]), 
                      color, -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        
        # 边框
        cv2.rectangle(display, 
                     (status["box"][0], status["box"][1]),
                     (status["box"][2], status["box"][3]), 
                     color, 2)
        
        # 轨迹
        history = status.get("history", [])
        for i in range(1, len(history)):
            cv2.line(display,
                    tuple(map(int, history[i-1])),
                    tuple(map(int, history[i])),
                    (255, 255, 0), 2)
        
        # 状态文本
        text = "Tracking" if status["tracking"] else "Predicting"
        cv2.putText(display, text, 
                   (status["box"][0], status["box"][1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(display, "Searching...", 
                   (width//2-100, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    
    return display