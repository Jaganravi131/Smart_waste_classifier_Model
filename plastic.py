from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# Define a custom callback function to print detections to terminal
def print_detections(predictions: dict, video_frame: VideoFrame):
    # Extract predictions
    if "predictions" in predictions:
        print("\n--- New Frame Detections ---")
        for pred in predictions["predictions"]:
            class_name = pred["class"]
            confidence = pred["confidence"]
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
    
    # You can still display the video with bounding boxes
    # by importing and using render_boxes alongside your custom function
    from inference.core.interfaces.stream.sinks import render_boxes
    render_boxes(predictions, video_frame)

# Initialize the pipeline with your custom callback
api_key = "FBhInBlzNw08RleCe8iQ"

pipeline = InferencePipeline.init(
    model_id="cleaned_dataset/6",  # Replace with your model ID
    video_reference=0,  # Default webcam
    on_prediction=print_detections,
    api_key=api_key,
)

pipeline.start()
pipeline.join()