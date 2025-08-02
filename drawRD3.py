"""
Simple plotting utility to display Rate-Distortion curves (RD) comparison
between codecs. Supports variable number of points per codec.
"""
import argparse
import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

_backends = ["matplotlib", "plotly"]

def parse_json_file(filepath, metric):
    filepath = Path(filepath)
    name = filepath.name.split(".")[0]
    with filepath.open("r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file "{filepath}"')
            raise err

    if "results" in data:
        results = data["results"]
    else:
        results = data

    if metric not in results:
        raise ValueError(
            f'Error: metric "{metric}" not available.'
            f' Available metrics: {", ".join(results.keys())}'
        )

    try:
        # Allow bpp and metric to have different lengths (skip mismatched points)
        bpp = results["bpp"]
        ys = results[metric]
        min_len = min(len(bpp), len(ys))
        return {
            "name": data.get("name", name),
            "xs": bpp[:min_len],  # Truncate to common length
            "ys": ys[:min_len],
        }
    except KeyError:
        raise ValueError(f'Invalid file "{filepath}"')

def matplotlib_plt(scatters, title, ylabel, output_file, limits=None, show=False, figsize=None):
    if figsize is None:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)
    
    styles = {
        'factorized1': {'color': 'blue', 'marker': 'o'},
        'hyperprior': {'color': 'red', 'marker': 'o'},
        'joint': {'color': 'orange', 'marker': 'o'},
        'bmshj2018-factorized': {'color': 'green', 'marker': 'o'},
        'mbt2018-mean': {'color': 'purple', 'marker': 'o'},
        'mbt2018': {'color': 'pink', 'marker': 'o'}
    }
    
    for sc in scatters:
        style = styles.get(sc["name"], {})
        ax.scatter(
            sc["xs"],
            sc["ys"],
            s=50,
            label=sc["name"],
            **style
        )
        ax.plot(
            sc["xs"],
            sc["ys"],
            linestyle='-',
            color=style.get('color', None),
            alpha=0.5
        )
    
    ax.set_xlabel("Bit-rate [bpp]")
    ax.set_ylabel(ylabel)
    ax.grid()
    if limits is not None:
        ax.axis(limits)
    ax.legend(loc="lower right")

    if title:
        ax.set_title(title)

    if show:
        plt.show()

    if output_file:
        os.makedirs("/output", exist_ok=True)
        fig.savefig(os.path.join("/output", output_file), dpi=300)

def plotly_plt(scatters, title, ylabel, output_file, limits=None, show=False, figsize=None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SystemExit("Unable to import plotly, install with: pip install pandas plotly")

    fig = go.Figure()
    
    styles = {
        'factorized1': {'color': 'blue', 'symbol': 'circle'},
        'hyperprior': {'color': 'red', 'symbol': 'circle'},
        'joint': {'color': 'orange', 'symbol': 'circle'},
        'bmshj2018-factorized': {'color': 'green', 'symbol': 'circle'},
        'mbt2018-mean': {'color': 'purple', 'symbol': 'circle'},
        'mbt2018': {'color': 'pink', 'symbol': 'circle'}
    }
    
    for sc in scatters:
        style = styles.get(sc["name"], {})
        fig.add_trace(
            go.Scatter(
                x=sc["xs"],
                y=sc["ys"],
                name=sc["name"],
                mode="markers+lines",
                marker=dict(
                    size=10,
                    color=style.get('color'),
                    symbol=style.get('symbol')
                ),
                line=dict(
                    color=style.get('color'),
                    width=1.5
                )
            )
        )

    fig.update_layout(title=title if title else "RD Curve")
    fig.update_xaxes(title_text="Bit-rate [bpp]")
    fig.update_yaxes(title_text=ylabel)
    if limits is not None:
        fig.update_xaxes(range=[limits[0], limits[1]])
        fig.update_yaxes(range=[limits[2], limits[3]])

    os.makedirs("/output", exist_ok=True)
    fig.write_image(os.path.join("/output", output_file or "plot.png"), scale=2)
    if show:
        fig.show()

def setup_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--metric",
        metavar="",
        type=str,
        default="ms-ssim",
        help="Metric (default: %(default)s)",
    )
    parser.add_argument("-t", "--title", metavar="", type=str, help="Plot title")
    parser.add_argument("-o", "--output", metavar="", type=str, help="Output file name")
    parser.add_argument(
        "--figsize",
        metavar="",
        type=float,
        nargs=2,
        default=(9, 6),
        help="Figure relative size (width, height), default: %(default)s",
    )
    parser.add_argument(
        "--axes",
        metavar="",
        type=float,
        nargs=4,
        default=None,
        help="Axes limit (xmin, xmax, ymin, ymax), default: autorange",
    )
    parser.add_argument(
        "--backend",
        type=str,
        metavar="",
        default=_backends[0],
        choices=_backends,
        help="Change plot backend (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true", help="Open plot figure")
    return parser

def main(argv):
    args = setup_args().parse_args(argv)

    scatters = []
    
    results_file = [
        '/code/reconstruction2/factorized1_results.json',
        '/code/reconstruction2/hyperprior_results.json',
        '/code/reconstruction2/joint_results.json',
        '/code/reconstruction2/factorized_mse_results.json',
        '/code/reconstruction2/hyperprior_mse_results.json',
        '/code/reconstruction2/joint_mse_results.json',
    ]
    
    for f in results_file:
        try:
            rv = parse_json_file(f, args.metric)
            scatters.append(rv)
        except Exception as e:
            print(f"Skipping invalid file {f}: {str(e)}")
            continue
    
    for sc in scatters:
        print(f"{sc['name']}: {len(sc['xs'])} points")
    
    ylabel = f"{args.metric} [dB]"
    func_map = {
        "matplotlib": matplotlib_plt,
        "plotly": plotly_plt,
    }

    func_map[args.backend](
        scatters,
        title=args.title if args.title else "RD Curve Comparison",
        ylabel=ylabel,
        output_file=args.output or "plot.png",
        limits=args.axes,
        figsize=args.figsize,
        show=args.show,
    )

if __name__ == "__main__":
    main(sys.argv[1:])