import json
import pandas as pd
import yaml
import torch
import os.path as osp
from PIL import Image
from torch import nn

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def prepare_json(frame_number, player1, player2, ball_coordinate, ball_event):
    trajectory = pd.DataFrame(ball_coordinate, columns=['x', 'y'])
    # Prepare the JSON object with relevant information
    json_data = {
        "frame_range": [frame_number - 29, frame_number],  # Range of 30 frames
        
        "players": {
            "player1": [
                {"frame": f + frame_number, "x1": player[0], "y1": player[1], "x2": player[2], "y2": player[3]}
                for f, player in enumerate(player1) if f >= 0 and f < 30  # We directly use all 30 frames
            ],
            "player2": [
                {"frame": f + frame_number, "x1": player[0], "y1": player[1], "x2": player[2], "y2": player[3]}
                for f, player in enumerate(player2) if f >= 0 and f < 30  # We directly use all 30 frames
            ]
        },
        "ball_data": {
            "trajectory": [{"x": row['x'], "y": row['y']} for index, row in trajectory.iterrows()],
            "events": ball_event  # Include all historical events
        }
    }
    # Print or return the JSON object
    return json.dumps(json_data, indent=4)

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _top1(scores):
    batch, seq, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, seq, -1), 1)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
