import numpy as np
import pandas as pd

class TrajectoryInterpolator:
    def __init__(self, interpolater_range=7):
        self.interpolater_range = interpolater_range

    def update_detection_history(self, trajectory, start_idx, end_idx):
        """
        Update detection history and check if we need to interpolate missing frames.
        """
        trajectory = pd.DataFrame(trajectory, columns=['x', 'y'])

        ### 处理开头的 None 值：用最近有效值填充
        check = 0
        while pd.isna(trajectory.loc[check, 'x']):
            # 当发现 None 时，填充为下一个有效值
            if check + 1 < len(trajectory) and not pd.isna(trajectory.loc[check + 1, 'x']):
                # 用最近有效值填充前面的所有 None
                for i in range(check):
                    trajectory.loc[i, 'x'] = trajectory.loc[check + 1, 'x']
                    trajectory.loc[i, 'y'] = trajectory.loc[check + 1, 'y']
            check += 1

        if start_idx <= 0:
            start_idx = start_idx + 1

        for i in range(start_idx, end_idx):
            prev_idx = i - 1
            next_idx = i + 1
            
            # Find valid previous and next frames to interpolate

            while prev_idx > 0 and pd.isna(trajectory.loc[prev_idx, 'x']):
                prev_idx -= 1
            while next_idx < len(trajectory) - 1 and pd.isna(trajectory.loc[next_idx, 'x']):
                next_idx += 1

            # Linear interpolation formula
            t1 = prev_idx
            t2 = next_idx
            t = i

            x_interp = trajectory.loc[prev_idx, 'x'] + (trajectory.loc[next_idx, 'x'] - trajectory.loc[prev_idx, 'x']) * (t - t1) / (t2 - t1)
            y_interp = trajectory.loc[prev_idx, 'y'] + (trajectory.loc[next_idx, 'y'] - trajectory.loc[prev_idx, 'y']) * (t - t1) / (t2 - t1)
            
            trajectory.loc[i, 'x'] = x_interp
            trajectory.loc[i, 'y'] = y_interp
        return trajectory