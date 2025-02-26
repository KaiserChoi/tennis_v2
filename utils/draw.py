import cv2
import math

class Draw_video:
    def __init__(self):
        self.last_draw = 0

    def draw_30_frame(self, frame_history, x, y, ies):
        

        drawed_frame = []
        visited = []
        i = 0

        for i in range(self.last_draw, len(frame_history)):
            frame = frame_history[i]
            # If at that frame a bounce is detected, mark the bounce as visited
            if i in ies.keys():
                visited.append(i)

            # Draw crosses at the specified (x, y) positions every frame onwards from bounce detection
            for item in visited:

                if ies[item] == 'shot':
                    color = (0, 255, 0) 
                elif ies[item] == 'bounce':
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                print(item, x[item], y[item])
                self.draw_cross(frame, x[item], y[item], color=color)

        self.last_draw = len(frame_history)
        return drawed_frame

    def draw_cross(frame, x, y, size=10, color=(255, 0, 0), thickness=2):

        if not math.isnan(x):
            x, y = int(x), int(y)  # Convert coordinates to integers
            cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness)
            cv2.line(frame, (x - size, y + size), (x + size, y - size), color, thickness)
            return frame
        return None