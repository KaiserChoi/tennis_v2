import cv2
import numpy as np

def undistort_image(image, strength=-0.14, tangential_strength=-0.05):
    """
    用于修正畸变的函数。
    
    参数：
    image: 输入图像
    strength: 径向畸变矫正强度，范围 [-2.0, 2.0]
    tangential_strength: 切向畸变矫正强度，范围 [-2.0, 2.0]
    
    返回：
    undistorted: 修正后的图像
    """
    # 获取图像的宽度和高度
    h, w = image.shape[:2]
    
    # 假设相机矩阵的焦距大约为图像宽度的 0.7 倍，光心在图像中心
    fx = w * 0.65  # 可以根据需要调整
    fy = h * 0.7
    cx = w // 2
    cy = h // 2
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    
    # 设置畸变系数
    k1 = strength
    k2 = strength * 0.1
    p1 = -tangential_strength * 0.1  # 切向畸变系数 p1
    p2 = tangential_strength * 0.05   # 切向畸变系数 p2
    dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float32)  # 包含径向和切向畸变
    
    # 执行畸变矫正
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    return undistorted
