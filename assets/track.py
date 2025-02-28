from collections import deque
from filterpy.kalman import KalmanFilter
import numpy as np
import cv2

class TrackedObject:
    def __init__(self, obj_id, initial_pos, appearance_feature=None):
        # 基础跟踪属性
        self.id = obj_id
        self.kf = self._create_kalman_filter(initial_pos)
        self.history = deque([initial_pos], maxlen=100)
        self.lost_count = 0
        self.active = True
        self.last_seen = 0
        
        # 稳定跟踪属性
        self.confirmed = False          # 是否已确认
        self.detection_count = 0        # 连续检测次数
        self.confirmation_threshold = 90# 需要90帧确认
        
        # 重识别属性
        self.appearance = appearance_feature if appearance_feature is not None else np.array([])
        self.max_age = 30               # 最大保留时间
        
        # 运动状态
        self.last_position = initial_pos
        self.velocity = np.zeros(2)

    @staticmethod
    def _create_kalman_filter(initial_pos):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        kf.P *= 100
        kf.R = np.eye(2) * 5
        kf.Q = np.eye(4) * 0.1
        
        # 修正维度：将初始位置转换为列向量
        kf.x = np.zeros((4, 1))  # 初始化为4x1的列向量
        kf.x[:2] = np.array(initial_pos).reshape(2, 1)  # 转换为2x1形状
        return kf

    def update(self, position, new_feature=None):
        """更新目标状态"""
        # 基础更新
        self.kf.update(position)
        self.history.append(position)
        self.lost_count = 0
        self.last_seen = 0
        self.last_position = position
        
        # 稳定跟踪逻辑
        self.detection_count += 1
        if self.detection_count >= self.confirmation_threshold:
            self.confirmed = True
            
        # 更新外观特征
        if new_feature is not None:
            self._update_appearance(new_feature)
            
        # 更新速度
        if len(self.history) >= 2:
            self.velocity = position - self.history[-2]

    def _update_appearance(self, new_feature, alpha=0.9):
        """更新外观特征"""
        if self.appearance.size == 0:
            self.appearance = new_feature
        else:
            self.appearance = alpha*self.appearance + (1-alpha)*new_feature

    def predict(self):
        """返回预测位置并更新状态"""
        self.kf.predict()
        self.last_seen += 1
        return self.kf.x[:2].flatten()

    def mark_missed(self):
        """标记为丢失"""
        self.lost_count += 1
        self.last_seen += 1
        # 未确认目标立即失效
        if not self.confirmed:
            self.active = False

    def should_remove(self):
        """判断是否应该移除"""
        # 已确认目标：允许丢失一定帧数
        if self.confirmed:
            return self.lost_count > self.max_age
        # 未确认目标：超过确认阈值或连续丢失
        return self.lost_count > 5 or self.detection_count + self.lost_count > self.confirmation_threshold

    def get_bbox(self, base_size=15, speed_scale=0.3):
        """获取动态边界框"""
        speed = np.linalg.norm(self.velocity)
        size = base_size * (1 + speed * speed_scale)
        x, y = self.last_position
        return (
            int(x - size/2),
            int(y - size/2),
            int(x + size/2),
            int(y + size/2)
        )
    
class Tracker:
    def __init__(self):
        self.objects = []
        self.lost_objects = []  # 用于重识别的缓冲池
        self.next_id = 1
        self.dist_threshold = 50    # 运动匹配阈值
        self.reid_threshold = 0.7   # 重识别阈值
        
        # 稳定跟踪参数
        self.confirmation_threshold = 90
        
        # GUI 状态
        self.reset_requested = False

    def reset(self):
        """重置所有跟踪状态"""
        self.objects = []
        self.lost_objects = []
        self.next_id = 1
        self.reset_requested = False

    def process_frame(self, frame, heatmap):
        """处理单帧"""
        if self.reset_requested:
            self.reset()
        
        # 获取当前检测
        current_detections = self._get_detections_with_features(frame, heatmap)
        
        # === 跟踪匹配流程 ===
        # 阶段1：匹配现有目标
        active_objects = [obj for obj in self.objects if obj.active]
        predictions = {obj.id: obj.predict() for obj in active_objects}
        matched, unmatched = self._match_detections(predictions, current_detections)
        
        # 阶段2：尝试重识别
        rematched = self._reidentify(current_detections)
        matched.update(rematched)
        
        # 更新匹配目标
        for obj_id, det in matched.items():
            obj = next(o for o in self.objects if o.id == obj_id)
            obj.update(det['position'], det['feature'])
            
        # 处理未匹配检测
        matched_values = list(matched.values())
        matched_tuples = {tuple(v) for v in matched_values}  # 使用集合提高查询效率
        unmatched_detections = [
            d for d in current_detections 
            if tuple(d) not in matched_tuples
        ]
        
        # 创建新目标
        for pos in unmatched_detections:
            new_obj = TrackedObject(self.next_id, pos)
            self.objects.append(new_obj)
            self.next_id += 1

        # 维护目标状态
        self._update_object_states()  
        
        return [obj for obj in self.objects if obj.confirmed]

    def _update_object_states(self):
        """更新目标状态并清理"""
        # 标记丢失目标
        for obj in self.objects:
            if obj.lost_count > 0:
                obj.mark_missed()
                
        # 移动丢失目标到缓冲池
        new_objects = []
        for obj in self.objects:
            if obj.should_remove() and obj.confirmed:
                self.lost_objects.append(obj)
            elif obj.active:
                new_objects.append(obj)
        self.objects = new_objects
        
        # 清理过期目标
        self.lost_objects = [o for o in self.lost_objects if not o.should_remove()]

    def _reidentify(self, detections):
        """重识别匹配"""
        matched = {}
        for obj in self.lost_objects:
            best_sim = -1
            best_det = None
            for det in detections:
                if self._is_valid_match(obj, det):
                    sim = self.cosine_similarity(obj.appearance, det['feature'])
                    if sim > best_sim and sim > self.reid_threshold:
                        best_sim = sim
                        best_det = det
            if best_det:
                matched[obj.id] = best_det
                obj.active = True
                self.lost_objects.remove(obj)
                self.objects.append(obj)
        return matched

    def _is_valid_match(self, obj, det):
        """时空有效性检查"""
        time_constraint = obj.last_seen < obj.max_age
        distance = np.linalg.norm(obj.predict() - det['position'])
        dist_constraint = distance < 2 * obj.last_seen * self.dist_threshold
        return time_constraint and dist_constraint

    def _extract_appearance(self, frame, center):
        """提取目标区域外观特征"""
        x, y = int(center[0]), int(center[1])
        # 示例使用颜色直方图（可根据需要替换为CNN特征）
        patch = frame[max(0,y-10):min(y+10,frame.shape[0]), 
                     max(0,x-10):min(x+10,frame.shape[1])]
        if patch.size == 0:
            return np.zeros(256)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def _get_detections_with_features(self, frame, heatmap):
        """从热力图中提取检测目标（带鲁棒性改进）"""
        # 确保输入数据在正确范围内
        heatmap = np.clip(heatmap, 0, 1)
        
        # 自适应阈值处理（提高噪声鲁棒性）
        thresh_value = max(0.5, np.percentile(heatmap, 95))  # 取前5%的亮度作为阈值
        _, thresh = cv2.threshold(heatmap, thresh_value, 1.0, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)

        # 形态学操作（可选，消除小噪声点）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, 
            connectivity=8, 
            ltype=cv2.CV_32S
        )

        detections = []
        for i in range(1, num_labels):  # 跳过背景标签0
            # 获取当前连通域的掩码
            mask = (labels == i)
            
            # 有效性检查：跳过面积过小的区域
            if np.sum(mask) < 5:  # 像素数小于5视为噪声
                continue

            # 加权中心计算（带置信度）
            y_coords, x_coords = np.where(mask)
            weights = heatmap[mask]
            total_weight = weights.sum()
            
            if total_weight < 1e-6:  # 防止除以零
                continue

            # 计算加权平均坐标
            center_x = np.dot(x_coords, weights) / total_weight
            center_y = np.dot(y_coords, weights) / total_weight
            
            # 置信度计算（基于热图强度）
            confidence = total_weight / np.sum(mask)  # 平均激活值

            # 有效性过滤（根据实际场景调整阈值）
            if confidence < 0.3:
                continue


            # 原有坐标计算...
            # 新增特征提取
            feature = self._extract_appearance(frame, (center_x, center_y))
            detections.append({
                'position': np.array([center_x, center_y]),
                'feature': feature
            })

        return detections

    
    def _data_association(self, predictions, detections):
        """简单最近邻数据关联"""
        matched = {}
        remaining_detections = set(range(len(detections)))
        
        for obj_id, pred_pos in predictions.items():
            if not remaining_detections:
                break
            
            # 将预测位置转换为平面数组
            pred_pos_flat = pred_pos.flatten()
            distances = [
                np.linalg.norm(d - pred_pos_flat) 
                for d in detections
            ]
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < self.dist_threshold:
                matched[obj_id] = detections[closest_idx]
                remaining_detections.remove(closest_idx)
        
        return matched
    
    def _extract_appearance(self, frame, center):
        """提取目标区域外观特征"""
        x, y = int(center[0]), int(center[1])
        # 示例使用颜色直方图（可根据需要替换为CNN特征）
        patch = frame[max(0,y-10):min(y+10,frame.shape[0]), 
                        max(0,x-10):min(x+10,frame.shape[1])]
        if patch.size == 0:
            return np.zeros(256)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _match_detections(self, predictions, detections):
        """双重匹配逻辑：运动+外观"""
        matched = {}
        remaining_dets = detections.copy()  # 使用检测副本而不是索引
        
        # 第一轮：运动匹配
        for obj_id, (pred_pos, _) in predictions.items():
            if not remaining_dets:
                break
            
            # 计算距离
            distances = [
                (idx, np.linalg.norm(det['position'] - pred_pos.flatten()))
                for idx, det in enumerate(remaining_dets)
            ]
            
            if distances:
                best_idx = min(distances, key=lambda x: x[1])[0]
                if distances[best_idx][1] < self.dist_threshold:
                    matched[obj_id] = remaining_dets[best_idx]
                    del remaining_dets[best_idx]

        # 第二轮：外观匹配
        for obj_id, (_, obj_feat) in predictions.items():
            if obj_id in matched or not remaining_dets or obj_feat is None:
                continue
            
            # 计算相似度
            similarities = [
                (idx, self._cosine_similarity(obj_feat, det['feature']))
                for idx, det in enumerate(remaining_dets)
            ]
            
            if similarities:
                best_idx, best_sim = max(similarities, key=lambda x: x[1])
                if best_sim > self.reid_threshold:
                    matched[obj_id] = remaining_dets[best_idx]
                    del remaining_dets[best_idx]

        unmatched = remaining_dets
        return matched, unmatched


    def _cosine_similarity(self, feat1, feat2):
        """计算余弦相似度"""
        if feat1 is None or feat2 is None:
            return 0
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1)*np.linalg.norm(feat2)+1e-8)

    def _create_new_objects(self, unmatched, detections, frame):
        """创建新目标"""
        for det in unmatched:
            new_obj = TrackedObject(
                obj_id=self.next_id,
                initial_pos=det['position'],
                appearance_feature=det['feature']
            )
            self.objects.append(new_obj)
            self.next_id += 1
        
    def reset(self):
        """完全重置跟踪器状态"""
        self.objects = []
        self.lost_buffer = []
        self.next_id = 1

    def update(self, position):
        self.last_position = position
        self.detection_count += 1
        if self.detection_count >= 90:
            self.confirmed = True
        self.lost_count = 0
    
    def mark_missed(self):
        if not self.confirmed:
            self.active = False
        self.lost_count += 1



class TrackerGUI:
    def __init__(self):
        self.reset_flag = False
        self.button_rect = (20, 20, 100, 40)  # (x, y, width, height)
        cv2.namedWindow("Tracking")
        cv2.setMouseCallback("Tracking", self.mouse_handler)

    def mouse_handler(self, event, x, y, flags, param):
        """处理鼠标点击事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = self.button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.reset_flag = True

    def draw_controls(self, frame):
        """绘制虚拟按钮"""
        bx, by, bw, bh = self.button_rect
        # 绘制按钮背景
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), -1)
        # 绘制按钮文字
        cv2.putText(frame, "Reset", (bx+10, by+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def check_reset(self):
        """返回重置标志并清除状态"""
        if self.reset_flag:
            self.reset_flag = False
            return True
        return False