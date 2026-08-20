from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import os
from datetime import datetime, timedelta
from collections import deque, Counter
import threading
import time
import argparse

# Use the new graph analysis module to log predictions for later plotting
from graph_analysis import GraphAnalyzer

# Matplotlib for real-time plotting
import matplotlib.pyplot as plt

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Default CSV log path
CSV_PATH = os.path.join(LOG_DIR, "detections.csv")

# Initialize GraphAnalyzer with default path; will be reinitialized if overridden via CLI
analyzer = GraphAnalyzer(csv_path=CSV_PATH)

# In-memory buffer for real-time plotting (thread-safe with lock)
# Store tuples: (timestamp(datetime), class(str), confidence(float))
recent_events = deque(maxlen=5000)
events_lock = threading.Lock()

# Default parameters for real-time plot (can be overridden via CLI)
PLOT_WINDOW_SECONDS = 30
PLOT_UPDATE_INTERVAL = 0.5  # seconds


def print_detections(predictions: dict, video_frame: VideoFrame):
    """Callback for inference pipeline: print, log, and store events for plotting."""
    ts = datetime.utcnow()

    # Print detections
    if "predictions" in predictions and predictions["predictions"]:
        print("\n--- New Frame Detections ---")
        for pred in predictions["predictions"]:
            class_name = pred.get("class")
            confidence = float(pred.get("confidence", 0.0))
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

            # Append to in-memory buffer for plotting
            with events_lock:
                recent_events.append((ts, class_name, confidence))
    else:
        # mark frame with no detections
        with events_lock:
            recent_events.append((ts, None, None))

    # Log predictions to CSV (GraphAnalyzer handles empty / missing data)
    analyzer.log_predictions(predictions, timestamp=ts)

    # Display the video with bounding boxes
    from inference.core.interfaces.stream.sinks import render_boxes
    render_boxes(predictions, video_frame)


def plotting_thread(stop_event: threading.Event):
    """Background thread that updates matplotlib plots in near-real time."""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    while not stop_event.is_set():
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=PLOT_WINDOW_SECONDS)

        with events_lock:
            events = list(recent_events)

        # Filter events within window
        window_events = [e for e in events if e[0] >= cutoff]

        # Counts per class in window
        class_counts = Counter([c for (_, c, _) in window_events if c])

        # Top classes (sorted)
        classes = list(class_counts.keys())
        counts = [class_counts[c] for c in classes]

        ax1.clear()
        ax1.set_title(f"Detections in last {PLOT_WINDOW_SECONDS}s")
        if classes:
            ax1.bar(classes, counts, color='tab:green')
            ax1.set_ylabel('Count')
        else:
            ax1.text(0.5, 0.5, 'No detections', ha='center', va='center')

        # Rolling average confidence for the most frequent class
        ax2.clear()
        ax2.set_title('Recent confidence for top class')
        if classes:
            top_class = classes[0]
            # Build time-series (per event) for the top class
            times = [ (e[0] - cutoff).total_seconds() for e in window_events if e[1] == top_class and e[2] is not None ]
            confs = [ e[2] for e in window_events if e[1] == top_class and e[2] is not None ]
            if times:
                ax2.plot(times, confs, marker='o', linestyle='-')
                ax2.set_xlim(0, PLOT_WINDOW_SECONDS)
                ax2.set_ylim(0, 1)
                ax2.set_xlabel('Seconds ago (window start = 0)')
                ax2.set_ylabel('Confidence')
            else:
                ax2.text(0.5, 0.5, f'No confidence entries for {top_class}', ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center')

        fig.tight_layout()
        try:
            plt.pause(PLOT_UPDATE_INTERVAL)
        except Exception:
            # In case of backend errors, just sleep briefly
            time.sleep(PLOT_UPDATE_INTERVAL)

    plt.close(fig)


def parse_video_reference(value: str):
    """Parse video reference: if integer string, return int, else return the string path."""
    try:
        return int(value)
    except Exception:
        return value


def main():
    global analyzer, PLOT_WINDOW_SECONDS, PLOT_UPDATE_INTERVAL, recent_events

    parser = argparse.ArgumentParser(description='Run real-time trash detection with optional live plots')
    parser.add_argument('--no-plot', action='store_true', help='Disable real-time plotting')
    parser.add_argument('--plot-window', type=int, default=PLOT_WINDOW_SECONDS, help='Seconds of recent data to show in plot')
    parser.add_argument('--plot-interval', type=float, default=PLOT_UPDATE_INTERVAL, help='Plot update interval in seconds')
    parser.add_argument('--csv-path', type=str, default=CSV_PATH, help='Path to write detections CSV')
    parser.add_argument('--max-events', type=int, default=5000, help='Max events to keep in memory for plotting')
    parser.add_argument('--video', type=str, default='0', help='Video reference (camera index or file path)')
    parser.add_argument('--model-id', type=str, default='cleaned_dataset/6', help='Model ID to run')
    parser.add_argument('--api-key', type=str, default=None, help='API key for inference service')
    args = parser.parse_args()

    # Apply CLI overrides
    PLOT_WINDOW_SECONDS = args.plot_window
    PLOT_UPDATE_INTERVAL = args.plot_interval
    recent_events = deque(maxlen=args.max_events)

    # Reinitialize analyzer with CLI csv path
    analyzer = GraphAnalyzer(csv_path=args.csv_path)

    # Video reference: try int else path
    video_ref = parse_video_reference(args.video)

    # API key handling: prefer CLI, then the environment.
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        parser.error("provide --api-key or set the ROBOFLOW_API_KEY environment variable")

    # Initialize the pipeline with your custom callback
    pipeline = InferencePipeline.init(
        model_id=args.model_id,
        video_reference=video_ref,
        on_prediction=print_detections,
        api_key=api_key,
    )

    stop_event = threading.Event()
    thread = None
    if not args.no_plot:
        thread = threading.Thread(target=plotting_thread, args=(stop_event,), daemon=True)
        thread.start()

    try:
        print("Starting pipeline. Press Ctrl+C to stop.")
        pipeline.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_event.set()
        try:
            pipeline.join(timeout=2.0)
        except Exception:
            pass


if __name__ == '__main__':
    main()