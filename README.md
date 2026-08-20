# Trash detector model - Graph analysis

This small addition logs model predictions to CSV and provides a small CLI to plot the detection history.

Files added:

- `graph_analysis.py` - simple logger (GraphAnalyzer) and helpers to load the CSV.
- `plot_cli.py` - CLI to produce two plots: counts per class and rolling average confidence for a class.
- `requirements.txt` - minimal dependencies (pandas, matplotlib).

How it works:

1. `plastic.py` now imports `GraphAnalyzer` and writes detections to `logs/detections.csv` automatically when the inference callback runs.
2. Use `plot_cli.py` to visualize results. Examples:

```powershell
python .\plot_cli.py --csv logs\detections.csv --plot counts
python .\plot_cli.py --csv logs\detections.csv --plot rolling --class Plastic --window 20
```

Notes:

- The CSV columns are: `timestamp,class,confidence,x,y,width,height,source`.
- The code creates the `logs` folder next to `plastic.py` if it doesn't exist.
- If you want additional plots (time-series counts, per-minute aggregation), I can add them.

Running `plastic.py` with options

`plastic.py` now accepts command-line flags to control plotting, CSV path, video source, and model settings. Examples:

Run with live plots (default):

```powershell
python .\plastic.py
```

Disable live plots (headless):

```powershell
python .\plastic.py --no-plot
```

Change the plot window to 60 seconds and update interval to 1s:

```powershell
python .\plastic.py --plot-window 60 --plot-interval 1.0
```

Save logs to a custom CSV and use a specific camera index or video file:

```powershell
python .\plastic.py --csv-path .\logs\my_detections.csv --video 1
python .\plastic.py --csv-path .\logs\my_detections.csv --video C:\path\to\video.mp4
```

Override model ID and API key from CLI:

```powershell
python .\plastic.py --model-id your_workspace/your_model/1 --api-key YOUR_API_KEY
```

Memory for in-memory events (affects how much recent history is shown):

```powershell
python .\plastic.py --max-events 10000
```

