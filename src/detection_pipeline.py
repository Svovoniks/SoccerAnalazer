
from typing import Tuple, Dict, List, Generator
import numpy as np
from ultralytics import YOLO
from src.annotator import AnnotatorManager
import supervision as sv
from src.utils import load_detection_model, read_video_frames_streaming, get_video_properties
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cv2


@dataclass
class VideoStatistics:
    total_frames: int = 0
    frames_with_detection: int = 0
    detection_percentage: float = 0.0
    average_ball_speed: float = 0.0
    max_ball_speed: float = 0.0
    average_confidence: float = 0.0
    total_detections: int = 0
    processing_time: float = 0.0
    heatmap_image: str = ""
    grid_data: list = None
    
    def __post_init__(self):
        if self.grid_data is None:
            self.grid_data = [[0 for _ in range(20)] for _ in range(20)]
    
    def to_dict(self) -> Dict:
        return {
            "total_frames": int(self.total_frames),
            "frames_with_detection": int(self.frames_with_detection),
            "detection_percentage": float(round(self.detection_percentage, 2)),
            "average_ball_speed": float(round(self.average_ball_speed, 2)),
            "max_ball_speed": float(round(self.max_ball_speed, 2)),
            "average_confidence": float(round(self.average_confidence, 2)),
            "total_detections": int(self.total_detections),
            "processing_time": float(round(self.processing_time, 2)),
            "heatmap_image": self.heatmap_image,
            "grid_data": self.grid_data
        }


class DetectionPipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.annotator_manager = AnnotatorManager()

    def get_detections(self, detection_model: YOLO, frame: np.ndarray, use_slicer: bool = False) -> sv.Detections:

        def inference_callback(frame: np.ndarray) -> sv.Detections:
            result = detection_model(frame, verbose=False)[0]
            return sv.Detections.from_ultralytics(result)

        if use_slicer:
            slicer = sv.InferenceSlicer(callback=inference_callback)
            detections = slicer(frame)
        else:
            detections = inference_callback(frame)

        ball_detections = detections[detections.class_id == 1]

        return ball_detections
        
    def initialize_model(self):

        if self.model is None:
            print("Loading detection model...")
            self.model = load_detection_model(self.model_path)
        return self.model
    
    def detect_frame_objects(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:

        if self.model is None:
            self.initialize_model()
        
        return self.get_detections(self.model, frame)
    
    
    def process_video_streaming(self, video_path: str, output_path: str) -> Generator[Tuple[np.ndarray, sv.Detections], None, None]:
        if self.model is None:
            self.initialize_model()
        
        for frame, frame_idx, total_frames, height, width, fps in read_video_frames_streaming(video_path):
            ball_detections = self.get_detections(self.model, frame)
            yield frame, ball_detections, frame_idx, total_frames, height, width, fps
    
    def detect_in_video_with_stats(self, video_path: str, output_path: str) -> Tuple[List[np.ndarray], VideoStatistics]:
        import time
        
        start_time = time.time()
        self.initialize_model()
        
        # Get video properties without loading all frames
        video_props = get_video_properties(video_path)
        total_frames = video_props['total_frames']
        frame_height = video_props['frame_height']
        frame_width = video_props['frame_width']
        fps = video_props['fps']
        
        print("Processing video in streaming mode...")
        stats = VideoStatistics(total_frames=total_frames)
        
        ball_positions: List[np.ndarray] = []
        confidences: List[float] = []
        
        # Initialize video writer for streaming output
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        try:
            # Process frames one at a time with streaming and progress bar
            for frame, frame_idx, ball_detections, _, _, _, _ in tqdm(self._process_video_streaming_with_detections(video_path), total=total_frames, desc="Processing frames", unit="frame"):
                annotated_frame = self.annotator_manager.annotate_ball(frame, ball_detections)
                out.write(annotated_frame)
                
                # Collect statistics
                if len(ball_detections.xyxy) > 0:
                    stats.frames_with_detection += 1
                    stats.total_detections += len(ball_detections.xyxy)
                    
                    # Calculate centroid of ball detection and track in grid
                    for box in ball_detections.xyxy:
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        ball_positions.append(np.array([x_center, y_center]))
                        
                        # Convert position to grid coordinates (20x20)
                        grid_x = int((x_center / frame_width) * 20)
                        grid_y = int((y_center / frame_height) * 20)
                        
                        # Clamp to grid bounds
                        grid_x = min(19, max(0, grid_x))
                        grid_y = min(19, max(0, grid_y))
                        
                        stats.grid_data[grid_y][grid_x] += 1
                    
                    # Collect confidence scores
                    if hasattr(ball_detections, 'confidence') and ball_detections.confidence is not None:
                        confidences.extend(ball_detections.confidence.tolist())
        finally:
            out.release()
            cv2.destroyAllWindows()
        
        # Calculate detection percentage
        if stats.total_frames > 0:
            stats.detection_percentage = (stats.frames_with_detection / stats.total_frames) * 100
        
        # Calculate ball speed (distance between consecutive detections)
        if len(ball_positions) > 1:
            speeds = []
            for i in range(1, len(ball_positions)):
                distance = np.linalg.norm(ball_positions[i] - ball_positions[i-1])
                speeds.append(distance)
            
            if speeds:
                stats.average_ball_speed = np.mean(speeds)
                stats.max_ball_speed = np.max(speeds)
        
        # Calculate average confidence
        if confidences:
            stats.average_confidence = np.mean(confidences) * 100
        
        # Generate heatmap visualization
        stats.heatmap_image = self._generate_heatmap(stats.grid_data)
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        print(f"Done! Output saved to: {output_path}")
        
        # Return empty list for compatibility (we've written directly to file)
        return [], stats
    
    def _process_video_streaming_with_detections(self, video_path: str):
        for frame, frame_idx, total_frames, height, width, fps in read_video_frames_streaming(video_path):
            ball_detections = self.get_detections(self.model, frame)
            yield frame, frame_idx, ball_detections, total_frames, height, width, fps
    
    def _generate_heatmap(self, grid_data: list) -> str:
        # Convert grid data to numpy array
        heatmap_array = np.array(grid_data, dtype=float)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # Create heatmap
        im = ax.imshow(heatmap_array, cmap='hot', origin='upper', interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Frames')
        
        # Set labels and title
        ax.set_xlabel('Horizontal Position')
        ax.set_ylabel('Vertical Position')
        ax.set_title('Ball Position Heatmap')
        
        # Set grid
        ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Remove tick labels for minor ticks
        ax.tick_params(which='minor', size=0)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"

    def detect_in_video(self, video_path: str, output_path: str):
        self.initialize_model()
        
        # Get video properties
        video_props = get_video_properties(video_path)
        frame_width = video_props['frame_width']
        frame_height = video_props['frame_height']
        fps = video_props['fps']
        
        print("Processing video in streaming mode...")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        try:
            # Process and write frames one at a time
            frame_count = 0
            for frame, frame_idx, ball_detections, total_frames, _, _, _ in self._process_video_streaming_with_detections(video_path):
                annotated_frame = self.annotator_manager.annotate_ball(frame, ball_detections)
                out.write(annotated_frame)
                frame_count += 1
                
                if frame_count % 100 == 0 or frame_count == 1:
                    print(f"Processed {frame_count}/{total_frames} frames")
        finally:
            out.release()
            cv2.destroyAllWindows()
        
        print(f"Done! Output saved to: {output_path}")
