"""CLI to plot detection logs created by `graph_analysis.GraphAnalyzer`.

Usage examples:
  python plot_cli.py --csv logs/detections.csv --plot counts
  python plot_cli.py --csv logs/detections.csv --plot rolling --class Plastic --window 20
"""
import argparse
import os
import sys
import matplotlib.pyplot as plt

from graph_analysis import GraphAnalyzer, aggregate_counts_by_class, moving_avg_confidence


def plot_counts(csv_path: str):
    ga = GraphAnalyzer(csv_path)
    df = ga.load_dataframe()
    counts = aggregate_counts_by_class(df)
    plt.figure(figsize=(8, 6))
    counts.plot(kind='bar')
    plt.title('Detections per Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_rolling(csv_path: str, class_name: str, window: int):
    ga = GraphAnalyzer(csv_path)
    df = ga.load_dataframe()
    series = moving_avg_confidence(df, class_name, window=window)
    if series.empty:
        print(f"No entries for class '{class_name}' in {csv_path}")
        return
    plt.figure(figsize=(10, 4))
    series.plot()
    plt.title(f'Rolling average confidence for {class_name} (window={window})')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to detections CSV')
    parser.add_argument('--plot', choices=['counts', 'rolling'], required=True)
    parser.add_argument('--class', dest='class_name', help='Class name for rolling plot')
    parser.add_argument('--window', type=int, default=10, help='Rolling window size')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        sys.exit(1)

    if args.plot == 'counts':
        plot_counts(args.csv)
    elif args.plot == 'rolling':
        if not args.class_name:
            print("--class is required for rolling plot")
            sys.exit(1)
        plot_rolling(args.csv, args.class_name, args.window)


if __name__ == '__main__':
    main()
