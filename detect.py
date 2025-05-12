'''
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

api_key = "FBhInBlzNw08RleCe8iQ"  # Replace with the API key you found

pipeline = InferencePipeline.init(
    model_id="garbage-classification-3/2",  # Replace with the actual model ID from Universe
    video_reference=0,  # 0 is typically the default webcam
    on_prediction=render_boxes,
    api_key=api_key,
)

pipeline.start()
pipeline.join()
'''
'''
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
    model_id="garbage-classification-3/2",  # Replace with your model ID
    video_reference=0,  # Default webcam
    on_prediction=print_detections,
    api_key=api_key,
)

pipeline.start()
pipeline.join()
'''
'''
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import numpy as np

# Define a custom callback function that combines results from multiple models
def ensemble_predictions(predictions: dict, video_frame: VideoFrame):
    print("\n--- New Frame Detections (Ensemble) ---")
    
    if "predictions" in predictions:
        # Process and display predictions
        for pred in predictions["predictions"]:
            class_name = pred["class"]
            confidence = pred["confidence"]
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
    
    # Still display the video with bounding boxes
    from inference.core.interfaces.stream.sinks import render_boxes
    render_boxes(predictions, video_frame)

# Initialize multiple models in a pipeline
api_key = "FBhInBlzNw08RleCe8iQ"

# Create a pipeline with multiple models
pipeline = InferencePipeline.init(
    model_id=[
        "waste-management/cleaned_dataset/version",  # Your primary waste model
        "another-workspace/another-waste-model/version"  # Second waste model
    ],
    video_reference=0,  # Default webcam
    on_prediction=ensemble_predictions,
    api_key=api_key,
)

pipeline.start()
pipeline.join()
''' 

'''
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import numpy as np

# Define a custom callback function that combines results from specialized models
def ensemble_waste_predictions(predictions: dict, video_frame: VideoFrame):
    print("\n--- New Frame Detections (Specialized Ensemble) ---")
    
    # Dictionary to store highest confidence predictions by class
    best_predictions = {}
    
    # Process predictions from all models
    if "predictions" in predictions:
        for pred in predictions["predictions"]:
            class_name = pred["class"]
            confidence = pred["confidence"]
            
            # Only keep the highest confidence prediction for each class
            if class_name not in best_predictions or confidence > best_predictions[class_name]["confidence"]:
                best_predictions[class_name] = pred
    
    # Display the best predictions
    for class_name, pred in best_predictions.items():
        confidence = pred["confidence"]
        # Only show predictions with confidence above threshold
        if confidence > 0.40:  # Adjust threshold as needed
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
            
            # You could add logic here to trigger different actions based on waste type
            if "plastic" in class_name.lower():
                print("  → Place in plastic recycling bin")
            elif "metal" in class_name.lower():
                print("  → Place in metal recycling bin")
            elif "paper" in class_name.lower():
                print("  → Place in paper recycling bin")
            elif "glass" in class_name.lower():
                print("  → Place in glass recycling bin")
            elif "biodegradable" in class_name.lower():
                print("  → Place in compost bin")
            else:
                print("  → Place in general waste bin")
    
    # Display the video with bounding boxes
    from inference.core.interfaces.stream.sinks import render_boxes
    render_boxes(predictions, video_frame)

# Initialize the pipeline with multiple specialized models
api_key = "FBhInBlzNw08RleCe8iQ"  # Use the API key from your first model

# Create a pipeline with specialized waste detection models
pipeline = InferencePipeline.init(
    model_id=[
        "garbage-classification-3/2",  # General waste model
        "plastic-detection-2mrgf/1",  # Specialized for plastics
        "atik-ayristirma/4",  # Specialized for metals
        "garbage_detection-wvzwv/9"  # Specialized for paper/glass
    ],
    video_reference=0,  # Default webcam
    on_prediction=ensemble_waste_predictions,
    api_key=api_key,
)

# Start the pipeline
print("Starting waste detection ensemble. Press Ctrl+C to stop.")
pipeline.start()
pipeline.join()
'''

'''
from inference import InferencePipeline
import cv2
import threading
import time

# Dictionary to store predictions from different models
ensemble_predictions = {}
lock = threading.Lock()

# Custom callback functions for each specialized model
def process_general_waste(predictions, video_frame):
    with lock:
        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                class_name = pred["class"]
                confidence = pred["confidence"]
                if class_name not in ensemble_predictions or confidence > ensemble_predictions[class_name]["confidence"]:
                    ensemble_predictions[class_name] = {"confidence": confidence, "source": "general"}

def process_plastic(predictions, video_frame):
    with lock:
        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                class_name = pred["class"]
                confidence = pred["confidence"] * 1.1  # Give plastic model slightly higher weight
                if class_name not in ensemble_predictions or confidence > ensemble_predictions[class_name]["confidence"]:
                    ensemble_predictions[class_name] = {"confidence": confidence, "source": "plastic"}

# Function to display results
def display_results():
    while True:
        time.sleep(1)  # Update console every second
        print("\n--- Ensemble Detection Results ---")
        with lock:
            local_predictions = ensemble_predictions.copy()
        
        for class_name, details in local_predictions.items():
            confidence = details["confidence"]
            source = details["source"]
            if confidence > 0.4:  # Confidence threshold
                print(f"Detected: {class_name} (Confidence: {confidence:.2f}, Model: {source})")

# Start display thread
display_thread = threading.Thread(target=display_results, daemon=True)
display_thread.start()

# Initialize pipelines for different models
api_key = "FBhInBlzNw08RleCe8iQ"  # Use your API key

# General waste model
general_pipeline = InferencePipeline.init(
    model_id="garbage-classification-3/2",  # Replace with actual model ID
    video_reference=0,  # Default webcam
    on_prediction=process_general_waste,
    api_key=api_key,
)

# Plastic specialized model
plastic_pipeline = InferencePipeline.init(
    model_id="garbage_detection-wvzwv/9",  # Replace with actual model ID
    video_reference=0,  # Same webcam
    on_prediction=process_plastic,
    api_key=api_key,
)

# Start pipelines
print("Starting waste detection ensemble. Press Ctrl+C to stop.")
general_pipeline.start()
plastic_pipeline.start()

try:
    # Keep main thread alive
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping pipelines...")
    # Cleanup will happen automatically when the program exits
'''

'''
from inference import InferencePipeline
import cv2
import threading
import time

# Dictionary to store predictions from different models
ensemble_predictions = {}
lock = threading.Lock()
current_frame = None  # Store the current frame for display

# Custom callback functions for each specialized model
def process_general_waste(predictions, video_frame):
    global current_frame
    with lock:
        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                class_name = pred["class"]
                confidence = pred["confidence"]
                if class_name not in ensemble_predictions or confidence > ensemble_predictions[class_name]["confidence"]:
                    ensemble_predictions[class_name] = {"confidence": confidence, "source": "general"}
        
        # Store the frame with bounding boxes for display
        from inference.core.interfaces.stream.sinks import render_boxes
        frame_with_boxes = video_frame.image.copy()
        render_boxes(predictions, video_frame)
        current_frame = video_frame.image

def process_plastic(predictions, video_frame):
    with lock:
        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                class_name = pred["class"]
                confidence = pred["confidence"] * 1.1  # Give plastic model slightly higher weight
                if class_name not in ensemble_predictions or confidence > ensemble_predictions[class_name]["confidence"]:
                    ensemble_predictions[class_name] = {"confidence": confidence, "source": "plastic"}

# Function to display video feed and results
def display_results():
    while True:
        # Display the video frame if available
        if current_frame is not None:
            cv2.imshow("Waste Detection", current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Print detection results
        print("\n--- Ensemble Detection Results ---")
        with lock:
            local_predictions = ensemble_predictions.copy()
        
        for class_name, details in local_predictions.items():
            confidence = details["confidence"]
            source = details["source"]
            if confidence > 0.4:  # Confidence threshold
                print(f"Detected: {class_name} (Confidence: {confidence:.2f}, Model: {source})")
                
                # Add logic to trigger different actions based on waste type
                class_lower = class_name.lower()
                if "plastic" in class_lower:
                    print("  → ACTION: Place in plastic recycling bin")
                    # You could add code here to control a servo for plastic sorting
                    # Example: send_command_to_arduino("PLASTIC")
                elif "metal" in class_lower:
                    print("  → ACTION: Place in metal recycling bin")
                    # Example: send_command_to_arduino("METAL")
                elif "paper" in class_lower:
                    print("  → ACTION: Place in paper recycling bin")
                    # Example: send_command_to_arduino("PAPER")
                elif "glass" in class_lower:
                    print("  → ACTION: Place in glass recycling bin")
                    # Example: send_command_to_arduino("GLASS")
                elif "cardboard" in class_lower:
                    print("  → ACTION: Place in cardboard recycling")
                    # Example: send_command_to_arduino("CARDBOARD")
                elif "biodegradable" in class_lower or "organic" in class_lower:
                    print("  → ACTION: Place in compost bin")
                    # Example: send_command_to_arduino("COMPOST")
                else:
                    print("  → ACTION: Place in general waste bin")
                    # Example: send_command_to_arduino("GENERAL")
                
        time.sleep(0.1)  # Short sleep to prevent high CPU usage

# Optional: Function to send commands to Arduino (similar to the automated sorting example)
def send_command_to_arduino(command):
    # This is a placeholder - you would implement actual serial communication here
    # Similar to the code from the automated sorting with computer vision example
    print(f"  → Sending command to sorting mechanism: {command}")
    # Example implementation:
    # import serial
    # arduino = serial.Serial('COM3', 9600, timeout=1)
    # arduino.write(f"{command}\n".encode())

# Start display thread
display_thread = threading.Thread(target=display_results, daemon=True)
display_thread.start()

# Initialize pipelines for different models
api_key = "FBhInBlzNw08RleCe8iQ"  # Use your API key

# General waste model
general_pipeline = InferencePipeline.init(
    model_id="garbage-classification-3/2",  # Replace with actual model ID
    video_reference=0,  # Default webcam
    on_prediction=process_general_waste,
    api_key=api_key,
)

# Plastic specialized model
plastic_pipeline = InferencePipeline.init(
    model_id="garbage_detection-wvzwv/9",  # Replace with actual model ID
    video_reference=0,  # Same webcam
    on_prediction=process_plastic,
    api_key=api_key,
)

# Start pipelines
print("Starting waste detection ensemble. Press 'q' in video window to stop.")
general_pipeline.start()
plastic_pipeline.start()

try:
    # Keep main thread alive
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping pipelines...")
    # Cleanup will happen automatically when the program exits
    '''


from inference import InferencePipeline
import cv2
import threading
import time

# Dictionary to store predictions from different models
ensemble_predictions = {}
lock = threading.Lock()
current_frame = None  # Store the current frame for display

# Custom callback functions for each model
def process_general_waste(predictions, video_frame):
    global current_frame
    with lock:
        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                class_name = pred["class"]
                confidence = pred["confidence"]
                if class_name not in ensemble_predictions or confidence > ensemble_predictions[class_name]["confidence"]:
                    ensemble_predictions[class_name] = {
                        "confidence": confidence, 
                        "source": "general",
                        "bbox": [pred.get("x"), pred.get("y"), pred.get("width"), pred.get("height")]
                    }
        
        # Store the frame with bounding boxes for display
        from inference.core.interfaces.stream.sinks import render_boxes
        render_boxes(predictions, video_frame)
        current_frame = video_frame.image

def process_specialized_waste(predictions, video_frame):
    with lock:
        if "predictions" in predictions:
            for pred in predictions["predictions"]:
                class_name = pred["class"]
                confidence = pred["confidence"] * 1.1  # Give specialized model slightly higher weight
                if class_name not in ensemble_predictions or confidence > ensemble_predictions[class_name]["confidence"]:
                    ensemble_predictions[class_name] = {
                        "confidence": confidence, 
                        "source": "specialized",
                        "bbox": [pred.get("x"), pred.get("y"), pred.get("width"), pred.get("height")]
                    }

# Function to display video feed and results
def display_results():
    while True:
        # Display the video frame if available
        if current_frame is not None:
            # Create a copy to avoid modifying the original
            display_frame = current_frame.copy()
            
            # Add text for detected items
            with lock:
                local_predictions = ensemble_predictions.copy()
            
            y_offset = 30
            for class_name, details in local_predictions.items():
                confidence = details["confidence"]
                if confidence > 0.4:  # Confidence threshold
                    # Determine bin type based on waste class
                    class_lower = class_name.lower()
                    if "plastic" in class_lower:
                        action = "Place in plastic recycling bin"
                        color = (0, 255, 0)  # Green
                    elif "metal" in class_lower:
                        action = "Place in metal recycling bin"
                        color = (255, 0, 0)  # Blue
                    elif "paper" in class_lower:
                        action = "Place in paper recycling bin"
                        color = (0, 0, 255)  # Red
                    elif "glass" in class_lower:
                        action = "Place in glass recycling bin"
                        color = (255, 255, 0)  # Cyan
                    elif "cardboard" in class_lower:
                        action = "Place in cardboard recycling"
                        color = (0, 165, 255)  # Orange
                    else:
                        action = "Place in general waste bin"
                        color = (128, 128, 128)  # Gray
                    
                    # Add text to the frame
                    text = f"{class_name}: {confidence:.2f} - {action}"
                    cv2.putText(display_frame, text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 30
                    
                    # Print to console as well
                    print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
                    print(f"  → ACTION: {action}")
            
            # Show the frame
            cv2.imshow("Waste Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        time.sleep(0.05)  # Short sleep to prevent high CPU usage

# Start display thread
display_thread = threading.Thread(target=display_results, daemon=True)
display_thread.start()

# Initialize pipelines for different models
api_key = "FBhInBlzNw08RleCe8iQ"  # Use your API key

# General waste model
general_pipeline = InferencePipeline.init(
    model_id="garbage-classification-3/2",  # Replace with actual model ID
    video_reference=0,  # Default webcam
    on_prediction=process_general_waste,
    api_key=api_key,
)

# Specialized waste model (e.g., plastic detection)
specialized_pipeline = InferencePipeline.init(
    model_id="garbage_detection-wvzwv/9",  # Replace with actual model ID
    video_reference=0,  # Same webcam
    on_prediction=process_specialized_waste,
    api_key=api_key,
)

# Start pipelines
print("Starting waste detection ensemble. Press 'q' in video window to stop.")
general_pipeline.start()
specialized_pipeline.start()

try:
    # Keep main thread alive
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping pipelines...")
    # Cleanup will happen automatically when the program exits
finally:
    cv2.destroyAllWindows()