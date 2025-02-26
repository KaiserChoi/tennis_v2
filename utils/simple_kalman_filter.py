import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


class SimpleKalmanFilter(object):
    def __init__(self, dim_x=4, dim_z=2):
        # 创建卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.filtered_positions = []

    def initiate(self, measurements):
        # 初始化状态向量 (位置 x, y 和速度 vx, vy)
        self.kf.x = np.array([measurements[0],  # 初始 x
                        measurements[1],  # 初始 y
                        0.,                  # 初始速度 vx
                        0.])                 # 初始速度 vy
        
        # 状态转移矩阵 F (假设速度在 dt 内不变)
        dt = 1 / 30  # 假设每秒采样一次
        self.kf.F = np.array([[1, 0, dt, 0],  # x = x + vx * dt
                        [0, 1, 0, dt],  # y = y + vy * dt
                        [0, 0, 1, 0],    # vx = vx
                        [0, 0, 0, 1]])   # vy = vy

        # 观测矩阵 H (直接测量 x, y)
        self.kf.H = np.array([[1, 0, 0, 0],  # 测量的 x 是状态向量的第 1 项
                        [0, 1, 0, 0]]) # 测量的 y 是状态向量的第 2 项

        # 过程噪声协方差矩阵 Q (可以根据实际情况调整)
        self.kf.Q = np.eye(4) * 0.1  # 过程噪声较小

        # 测量噪声协方差矩阵 R (假设测量噪声较小)
        self.kf.R = np.eye(2) * 0.1  # 测量噪声

        # 初始协方差矩阵 P (较大的值，表示初始不确定)
        self.kf.P *= 1000

    def update(self, z):
        self.kf.update(z)

    def predict(self):
        self.kf.predict()
        
        out = self.kf.x[:2]
        self.filtered_positions.append(out)
        return out
    
    def xyxy_to_xyah(self, bbox):
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            h = 1
        return [xc, yc, w / h, h]
    
    def xyah_to_xyxy(self, bbox):
        xc, yc, a, h = bbox
        x1 = xc - a * h / 2
        y1 = yc - h / 2
        x2 = xc + a * h / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]