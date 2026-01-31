import cv2
from ultralytics import YOLO
from typing import Generator, Tuple

def load_detection_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    return model

def read_video_frames_streaming(vid_path: str) -> Generator[Tuple[int, int, int, int], None, None]:
    cap = cv2.VideoCapture(vid_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        yield frame, frame_count, total_frames, frame_height, frame_width, fps
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def get_video_properties(vid_path: str) -> dict:
    cap = cv2.VideoCapture(vid_path)
    props = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'frame_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'fps': cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return props
