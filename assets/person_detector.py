import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import time

class PersonDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.min_score = 0.85
        self.player1, self.player2 = [], []
        self.height = None
        self.width = None
        
    def preprocess_image(self, image):
        frame_tensor = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_tensor).unsqueeze(0).float().to(self.device)
        
        return frame_tensor

    def detect_person(self, image):
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
            
        bboxes = []
        probs = []
        
        # print(predictions)
        
        for box, label, score in zip(predictions[0]['boxes'][:], predictions[0]['labels'], predictions[0]['scores']):
            if score > self.min_score and label == 1:
                bbox = box.detach().cpu().numpy()
                score = score.detach().cpu().numpy()
                bboxes.append(bbox)
                probs.append(score)
                
        return bboxes, probs
    
    def filter_players(self, image):
        bboxes, probs = self.detect_person(image)
        
        if len(bboxes) == 0:
            return [], []
        
        mid = self.height / 2
        
        player1 = []
        player2 = []
        
        for bbox in bboxes:
            if bbox[3] < mid:
                player2.append(bbox)
            else:
                player1.append(bbox)
        
        return player1, player2
    
    def track_players(self, frame, frame_number):
        self.height, self.width = frame.shape[:2]
        persons_top, persons_bottom = self.filter_players(frame)
        
        persons_top = persons_top if persons_top else [[0, 0, 0, 0]]
        persons_bottom = persons_bottom if persons_bottom else [[0, 0, 0, 0]]
        
        x1, y1, x2, y2 = map(int, persons_bottom[0])
        x3, y3, x4, y4 = map(int, persons_top[0])
            
        return [frame_number, x1, y1, x2, y2, x3, y3, x4, y4]
    
if __name__ == '__main__':
    detector = PersonDetector()
    video_path = 'resources/samtennisvids.mp4'
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('resources/output.mp4', fourcc, fps, (width, height))

    results = []
    frame_number = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.track_players(frame, frame_number)
        results.append(result)
        # print(f"Frame {frame_number}: {result}")
        
        # Draw the bounding boxes
        x1, y1, x2, y2, x3, y3, x4, y4 = result[1:]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        out.write(frame)
        
        frame_number += 1
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    columns = [
        'frame_number', 
        'player1_x1', 'player1_y1', 'player1_x2', 'player1_y2',  # Player 2 (top)
        'player2_x3', 'player2_y3', 'player2_x4', 'player2_y4'   # Player 1 (bottom)
    ]

    result_df = pd.DataFrame(results, columns=columns)
    result_df.to_csv('resources/results.csv', index=False)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds. fps: {361/(end_time - start_time):.2f}")