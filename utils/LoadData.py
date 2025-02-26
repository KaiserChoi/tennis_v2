import pandas as pd
import numpy as np
import cv2
import os

combination = [[0,1,0,2], [1,0,2,0], [0, 2, 0, 1], [2, 0, 1, 0]]

def load_data(file_path):
    features,  labels = [], []
    files = get_files(file_path)
    for file in files:
        df = pd.read_csv(file).dropna(subset=['x-coordinate', 'y-coordinate', 'status'])
        feature,  label = process_data_by_stage(df)
        features.extend(feature)
        labels.extend(label)
        
    return np.array(features), np.array(labels)

def process_data_by_stage(df):
    # Initialize variables
    groups = []
    features = []
    labels = []
    current_group = []
    last_status = df.iloc[0]['status']

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        status = row['status']
        # Append the current row to the current group
        current_group.append(row)
        if (last_status == 0 and status in [1, 2]):
            # Append the current group to groups list if it's not empty
            if current_group:
                groups.append(pd.DataFrame(current_group))
                current_group = []
        # Update last_status
        last_status = status
    
    # Append the last group if not empty
    if current_group:
        groups.append(pd.DataFrame(current_group))
    # Generate random slices from the data
    random_data_slices = random_slices(df, len(groups), 5, 40)
    groups.extend(random_data_slices)

    for clip in groups:
        feature, label = resize_feature_label(clip) # , df
        if feature is not None and label is not None:
            features.append(feature)
            labels.append(label)
    return features, labels


def random_slices(df, num_slices, min_len, max_len):
    slices = []
    np.random.seed(42)  # For reproducibility
    
    while len(slices) < num_slices:
        start_index = np.random.randint(0, len(df) - min_len)
        max_end_index = min(start_index + max_len, len(df))
        end_index = np.random.randint(start_index + min_len, max_end_index)
        
        # # Ensure that the start and end status are not both 1 or 2
        # if not ((df.iloc[start_index]['status'] in [1, 2]) and (df.iloc[end_index - 1]['status'] in [1, 2])):
        #     slices.append(df.iloc[start_index:end_index])
        # Check if all status in the slice are 0
        slice_df = df.iloc[start_index:end_index]
        if (slice_df['status'] == 0).all():
            slices.append(slice_df)
    return slices

# def resize_feature_label(clip, raw):
#     label = np.zeros([32, 1])
#     coord = clip[['x-coordinate', 'y-coordinate']].values
#     coord = coord.astype(np.int16)

#     resized_coord = cv2.resize(coord, (2, 32), interpolation=cv2.INTER_NEAREST)
#     # print(resized_coord)
#     for l in range(1, 3):
#         status_indices = raw[raw['status'].isin([l])]
#         label_xy = status_indices[['x-coordinate', 'y-coordinate']].values
#         for xy in label_xy:
#             # 找出与 target 匹配的子数组的索引
#             indices = np.where(np.all(resized_coord == xy, axis=1))
#             for indice in indices:
#                 label[indice] = l

#     # print(label)
#     return resized_coord, label[-1]

def resize_feature_label(clip, size=32):
    coord = clip[['x-coordinate', 'y-coordinate']].values
    coord = coord.astype(np.int16)

    if coord.shape[0] > 4:
        resized_coord = cv2.resize(coord[:-2], (2, size-2), interpolation=cv2.INTER_LINEAR)
        resized_coord = np.append(resized_coord, coord[-2:], axis=0)
        label = clip.iloc[-1]['status'].astype(np.int16)
    else:
        print('warn!!!')
        return None, None
    return resized_coord, label

def resize_feature(clip, size=32):
    coord = clip.astype(np.int16)
    resized_coord = cv2.resize(coord[:-2], (2, size-2), interpolation=cv2.INTER_LINEAR)
    resized_coord = np.append(resized_coord, coord[-2:], axis=0)
    return resized_coord



def process_data_by_window(file_path):
    """
    滑动窗口提取特征和标签。窗口大小为30，例如:第16帧，则取1-15和17到30帧。
    每次滑动生成一个样本，label为锚点的第16帧的status。
    数据结构: row = [x, y, status]
    返回: features, labels
    """
    # 加载数据
    df = pd.read_csv(file_path)
    # 去掉所有包含 NaN 的行
    df = df.dropna()

    # 窗口大小
    window_size = 30
    half_window = window_size // 2

    # 存储特征和标签
    features = []
    labels = []

    # 遍历数据，生成滑动窗口
    for index in range(15, len(df) - 14):
        # 确定窗口范围 (排除超出边界的数据)
        start_idx = max(0, index - half_window)
        end_idx = min(len(df), index + half_window)

        # 滑动窗口中排除锚点
        window = df.iloc[start_idx:index].values.tolist() + df.iloc[index:end_idx].values.tolist()

        # 提取窗口特征 (x, y 坐标)
        feature = [[row[2], row[3]] for row in window]  # 假设 'x-coordinate' 在第0列，'y-coordinate' 在第1列

        # 提取锚点的标签
        label = df.iloc[index]['status']

        # 添加到列表
        features.append(feature)
        labels.append(label)

    return features, labels


def get_files(directory):
    """
    遍历指定目录下的所有 'clip' 文件夹，读取并处理每个文件夹中的 'label.csv' 文件。
    """
    # 存储所有读取到的 DataFrame
    files_paths = []

    # 遍历目录中的所有文件和文件夹
    for root, dirs, files in os.walk(directory):
        # 检查文件夹名是否以 'clip' 开始
        if root.split(os.sep)[-1].startswith('clip'):
            label_file_path = os.path.join(root, 'labels.csv')
            # 检查 'label.csv' 是否存在于这个目录
            if os.path.exists(label_file_path):
                files_paths.append(label_file_path)
                # print(f"Read labels from {label_file_path}")
            else:
                print(f"Label file not found in {root}")

    return files_paths

def process_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 插值处理坐标
    interpolated_coords = interpolation(df[['x-coordinate', 'y-coordinate']].values)
    df['x-coordinate'], df['y-coordinate'] = zip(*interpolated_coords)

    # 处理缺失的status，先尝试前向填充，然后后向填充
    df['status'] = df['status'].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate differences between subsequent rows
    df['dx'] = df['x-coordinate'].diff().fillna(0) + 1e-15
    df['dy'] = df['y-coordinate'].diff().fillna(0) + 1e-15
    
    # Apply the angle calculation
    df['direction'] = df.apply(lambda row: calculate_direction(row['dx'], row['dy']), axis=1)


    # 准备特征和标签
    features = df[['y-coordinate', 'direction']]
    labels = df['status']
    
    return features.values, labels.values

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolation(coords):
  coords =coords.copy()
  x, y = [x[0] if x is not None else np.nan for x in coords], [x[1] if x is not None else np.nan for x in coords]

  xxx = np.array(x) # x coords
  yyy = np.array(y) # y coords

  nons, yy = nan_helper(xxx)
  xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
  nans, xx = nan_helper(yyy)
  yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

  newCoords = [*zip(xxx,yyy)]

  return newCoords


def calculate_velocity(coords):
    V = []
    for i in range(len(coords)-1):
        p1 = coords[i]
        p2 = coords[i+1]
        # 计算速度，假设时间间隔为1
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        v = (vx**2 + vy**2)**0.5
        V.append(v)
    V.append(V[-1])  # 为最后一个点添加速度值
    return V

def calculate_direction(dx, dy):
    # Handle the stationary case
    if dx == 0 and dy == 0:
        return None  # No movement
    angle = np.arctan2(-dy, dx)
    if angle < 0:
        angle += 360  # Normalize angle to be in the range [0, 360)
    return angle



if __name__ == '__main__':
    directory_path = 'E:/TennisProject-main/train' # E:/TennisProject-main/train/game1/clip1/labels.csv

    features,  labels = load_data(directory_path)
    #print(len(features[0][1]))
    print(np.array(features).shape)
    print(np.array(labels).shape)


