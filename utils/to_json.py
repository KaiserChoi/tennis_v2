import json
import pandas as pd

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
