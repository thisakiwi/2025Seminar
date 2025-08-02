"""
Simple plotting utility to display Rate-Distortion curves (RD) comparison
between codecs.
"""
import argparse
import json
import sys
import os
import glob
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
        return {
            "name": data.get("name", name),
            "xs": results["bpp"],
            "ys": results[metric],
        }
    except KeyError:
        raise ValueError(f'Invalid file "{filepath}"')

#将分点文件按曲线名称分组
def group_point_files(files):
    curves = {}
    for f in files:
        #从文件名提取曲线名称（如A_results_0.0018.json -> A）
        curve_name = Path(f).name.split("_")[0]
        if curve_name not in curves:
            curves[curve_name] = []
        curves[curve_name].append(f)
    return curves

def matplotlib_plt(scatters, title, ylabel, output_file, limits=None, show=False, figsize=None):
    if figsize is None:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)
    
    styles = {
        'factorized': {'color': 'blue', 'marker': 'o'},
        'factorized1': {'color': 'orange', 'marker': 'o'},
        'hyperprior': {'color': 'red', 'marker': 's'},
        'joint': {'color': 'green', 'marker': 'o'},
        'D': {'color': 'purple', 'marker': 'D'}
    }
    
    for sc in scatters:
        style = styles.get(sc["name"][0], {})
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
        'factorized': {'color': 'blue', 'symbol': 'circle'},
        'factorized1': {'color': 'orange', 'symbol': 'circle'},
        'hyperprior': {'color': 'red', 'symbol': 'square'},
        'joint': {'color': 'green', 'symbol': 'circle'},
        'D': {'color': 'purple', 'symbol': 'diamond'}
    }
    
    for sc in scatters:
        style = styles.get(sc["name"][0], {})
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
        "--input-dir",
        metavar="",
        type=str,
        default="/code/reconstruction",
        help="Input directory containing JSON files"
    )
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

    point_files = glob.glob(os.path.join(args.input_dir, "*_*.json"))
    if not point_files:
        raise FileNotFoundError(f"No point files found in {args.input_dir}")
    
    curve_groups = group_point_files(point_files)
    scatters = []
    
    for curve_name, files in curve_groups.items():
        curve_points = []
        for f in sorted(files):  # 按文件名排序确保顺序正确
            try:
                data = parse_json_file(f, args.metric)
                curve_points.append(data)
            except Exception as e:
                print(f"Skipping invalid file {f}: {str(e)}")
                continue
        
        if curve_points:
            combined = {
                "name": curve_name,
                "xs": [x for p in curve_points for x in p["xs"]],
                "ys": [y for p in curve_points for y in p["ys"]]
            }
            scatters.append(combined)
    
    if not scatters:
        raise ValueError("No valid data found to plot")
    
    df = pd.DataFrame(scatters)  
    print(df)
    
    ylabel = f"{args.metric} [dB]"
    func_map = {
        "matplotlib": matplotlib_plt,
        "plotly": plotly_plt,
    }

    func_map[args.backend](
        scatters,
        "RD Curve Comparison",
        ylabel,
        args.output or 'plot.png',
        limits=args.axes,
        figsize=args.figsize,
        show=args.show,
    )

if __name__ == "__main__":
    main(sys.argv[1:])