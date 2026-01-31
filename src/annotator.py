

import numpy as np
import supervision as sv


class AnnotatorManager:

    def __init__(self):
        self.triangle_annotator = sv.TriangleAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()


    def annotate_ball(self, frame: np.ndarray, ball_detections: sv.Detections) -> np.ndarray:
        if ball_detections is not None and len(ball_detections.xyxy) > 0:
            frame = self.box_annotator.annotate(frame, ball_detections[0])
            frame = self.label_annotator.annotate(frame, ball_detections[0], labels=["Ball"])
            return self.triangle_annotator.annotate(frame, ball_detections[0])
        return frame
