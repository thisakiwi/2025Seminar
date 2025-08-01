# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time
import datetime

from collections import defaultdict
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from src.zoo import models 

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img)


def reconstruct(reconstruction, filename, recon_path):
    reconstruction = reconstruction.squeeze()
    reconstruction.clamp_(0, 1)
    reconstruction = transforms.ToPILImage()(reconstruction.cpu())
    reconstruction.save(os.path.join(recon_path, filename))


@torch.no_grad()
def inference(model, x, filename, recon_path):
    if not os.path.exists(recon_path):
        os.makedirs(recon_path)

    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 256  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    # x_padded = x

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start
    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    ps = psnr(x, out_dec["x_hat"])
    print(f"{filename}: bpp:{bpp}, psnr:{ps}")

    return {
        "psnr": ps,
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

@torch.no_grad()
def inference_entropy_estimation(model, x, filename):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    ps = psnr(x, out_net["x_hat"])
    print(f"{filename}: bpp:{bpp}, psnr:{ps}")

    return {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_checkpoint(model, checkpoint_path: str) -> nn.Module:
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return model.load_state_dict(state_dict).eval()


def eval_model(model, filepaths, entropy_estimation=False, half=False, recon_path='reconstruction'):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        _filename = f.split("\\")[-1]

        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, _filename, recon_path)
        else:
            rv = inference_entropy_estimation(model, x, _filename)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser()

    # Common options.
    parent_parser.add_argument(
        "-d", 
        "--dataset", 
        type=str,
        default='/data/lyq133/kodak/kodak',
        help="dataset path"
    )
    parent_parser.add_argument(
        "-r", 
        "--recon_path", 
        type=str, 
        default=r"./reconstruction/",
        help="where to save recon img")
    parent_parser.add_argument(
        "-a",
        "--model_name",
        type=str,
        choices=models.keys(),
        help="model name",
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        type=bool, 
        default=True, 
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="checkpoint path",
    )
    return parent_parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    runs = [args.path]
    # or self-defined
    # runs = [
    #     'model1', 
    #     'model2', 
    # ]

    opts = (args.model_name,)
    log_fmt = "\rEvaluating {run:s}"
    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        state_dict = torch.load(run, map_location=torch.device('cpu'))
        model = models[args.model_name]()
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        model = model.eval()

        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")

        model.update(force=True)

        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.recon_path)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.model_name,
        "description": f"Inference ({description})",
        "results": results,
    }

    #输出文件路径，文件名的0.0代表lambda
    output_filename = f"/output/{args.model_name}_results_0.0.json"
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)
    
    print(json.dumps(output, indent=2))
    
if __name__ == "__main__":
    main(sys.argv[1:])
