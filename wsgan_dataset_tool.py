import os
import json
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import warnings
import functools
import io
import sys
import zipfile
import PIL.Image
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from tqdm import tqdm
from torchvision import datasets
from torch_utils.misc import create_L_ind


# ----------------------------------------------------------------------------


def error(msg: str) -> None:
    print("Error: " + msg)
    sys.exit(1)


# ----------------------------------------------------------------------------


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a


# ----------------------------------------------------------------------------


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split(".")[-1]


# ----------------------------------------------------------------------------


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f".{ext}" in PIL.Image.EXTENSION  # type: ignore


# ----------------------------------------------------------------------------


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str,
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = {"box": PIL.Image.BOX, "lanczos": PIL.Image.LANCZOS}[resize_filter]

    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[
            (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
            (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
        ]
        img = PIL.Image.fromarray(img, "RGB")
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, "RGB")
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == "center-crop":
        if (output_width is None) or (output_height is None):
            error(
                "must specify --width and --height when using "
                + transform
                + "transform"
            )
        return functools.partial(center_crop, output_width, output_height)
    if transform == "center-crop-wide":
        if (output_width is None) or (output_height is None):
            error(
                "must specify --width and --height when using "
                + transform
                + " transform"
            )
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, "unknown transform"


def open_dest(
    dest: str,
) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == "zip":
        if os.path.dirname(dest) != "":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode="w", compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)

        return "", zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error("--dest folder must be empty")
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, "wb") as fout:
                if isinstance(data, str):
                    data = data.encode("utf8")
                fout.write(data)

        return dest, folder_write_bytes, lambda: None


def main(args: Namespace) -> None:
    # We use torchvision datasets for the StyleWSGAN experiments in the WSGAN paper.
    # The LFs, which needed to be stored and shared separartely, have to be matched with the correct images.
    # We ran this script with torchvision v0.12, but loading the data with v0.9 that this repository uses
    # should work as well. You can double check by computing the average LF accuracy and comparing it to the
    # number reported in the paper.

    # LSUN scene categories
    train_idxs = None
    datasetname = args.dataset
    datasetsavename = datasetname

    # ------------------------
    # pytorch vision dataset
    # ------------------------

    if args.dataset.lower() == "lsun":
        n_classes = 10
        traindataset = datasets.LSUN(args.data_path, classes="train")
    elif "cifar" in args.dataset.lower():
        n_classes = 10
        traindataset = datasets.CIFAR10(args.data_path, train=True, download=True)

    # ------------------------
    # Load fixed LFs
    # ------------------------

    # load precomputed LFs and training indices
    # Lambdas is the LF output matrix

    lfroot = args.lfroot
    if args.dataset.lower() == "cifar10-lownoise":
        with open(os.path.join(lfroot, "cifar10lownoise/LFs.csv"), "rb") as f:
            Lambdas = np.loadtxt(f, delimiter=",")
        with open(os.path.join(lfroot, "cifar10lownoise/indices.csv"), "rb") as f:
            train_idxs = np.loadtxt(f, delimiter=",").astype(int)
    elif args.dataset.lower() == "cifar10-b":
        with open(os.path.join(lfroot, "CIFAR10-B/LFs.csv"), "rb") as f:
            Lambdas = np.loadtxt(f, delimiter=",")
        with open(os.path.join(lfroot, "CIFAR10-B/indices.csv"), "rb") as f:
            train_idxs = np.loadtxt(f, delimiter=",").astype(int)
    elif args.dataset.lower() == "LSUN":
        with open(os.path.join(lfroot, "LSUN/LFs.csv"), "rb") as f:
            Lambdas = np.loadtxt(f, delimiter=",")
        with open(os.path.join(lfroot, "LSUN/indices.csv"), "rb") as f:
            train_idxs = np.loadtxt(f, delimiter=",").astype(int)

    num_LFs = Lambdas.shape[1]
    datasetsavename = datasetsavename + "_%dlfs" % num_LFs

    # subset the dataset according the training indices used in the paper
    trainset_sub = torch.utils.data.Subset(traindataset, train_idxs)

    # create a torch tensor of the LF outputs
    if Lambdas is not None:
        lambda_tensor = torch.tensor(Lambdas, requires_grad=False).float()
        # set up the indicator vector of non-abstains (i.e. at least one LF vote available for sample)
        full_filter_idx = lambda_tensor.sum(1) != 0
        print("Num samples with non-abstains:", full_filter_idx.sum())
        # create onehot representation of LFs
        lambda_oh = create_L_ind(lambda_tensor, n_classes)
    else:
        full_filter_idx = None
        lambda_oh = None

    dest_root = args.dest_root
    dest_path = os.path.join(dest_root, "style/datasets/")
    os.makedirs(dest_path, exist_ok=True)
    dest = os.path.join(dest_path, "%s.zip" % datasetsavename)

    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    lambda_oh = lambda_oh.type(torch.uint8).numpy()

    resize_filter = "lanczos"
    width = None
    height = None
    transform = None
    savelfsinmeta = False  # LFs will be saved in .txt files within the zip. Avoids issues for large datasets
    if "lsun" in datasetsavename.lower():
        width = 256
        height = 256
        transform = "center-crop"

    transform_image = make_transform(transform, width, height, resize_filter)

    dataset_attrs = None
    labels = []
    lfs = []
    lfs_oh = []
    lfs_filter = []
    full_filter_idx = full_filter_idx.type(torch.uint8).numpy()

    for i, tup in tqdm(enumerate(trainset_sub)):
        idx_str = f"{i:08d}"
        archive_fname = f"{idx_str[:5]}/img{idx_str}.png"
        archive_fname_lftxt = f"{idx_str[:5]}/img{idx_str}.txt"
        img, label = tup
        lf = Lambdas[i]
        lf_oh = lambda_oh[i]
        lffilter = full_filter_idx[i]
        img = np.array(img)

        # Apply crop and resize.
        img = transform_image(img)

        # did transform drop image?
        if img is None:
            warnings.warn("Image dropped!")
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": channels,
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs["width"]
            height = dataset_attrs["height"]
            if width != height:
                error(
                    f"Image dimensions after scale and crop are required to be square.  Got {width}x{height}"
                )
            if dataset_attrs["channels"] not in [1, 3]:
                error("Input images must be stored as RGB or grayscale")
            if width != 2 ** int(np.floor(np.log2(width))):
                error(
                    "Image width/height after scale and crop are required to be power-of-two"
                )
        elif dataset_attrs != cur_image_attrs:
            err = [
                f"  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}"
                for k in dataset_attrs.keys()
            ]
            error(
                f"Image attributes must be equal across all images of the dataset.  Got:\n"
                + "\n".join(err)
            )

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, {1: "L", 3: "RGB"}[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format="png", compress_level=0, optimize=False)
        save_bytes(
            os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer()
        )
        labels.append([archive_fname, label])
        lflist = lf.tolist()
        if not savelfsinmeta:
            save_bytes(
                os.path.join(archive_root_dir, archive_fname_lftxt), json.dumps(lflist)
            )
        else:
            lfs.append([archive_fname, lf.tolist()])
            lfs_oh.append([archive_fname, lf_oh.tolist()])
        lfs_filter.append([archive_fname, int(lffilter)])
    if not savelfsinmeta:
        metadata = {
            "labels": labels,
            "lfs_filter": lfs_filter,
        }
    else:
        metadata = {
            "labels": labels,
            "lfs": lfs,
            "lfs_oh": lfs_oh,
            "lfs_filter": lfs_filter,
        }
    save_bytes(os.path.join(archive_root_dir, "dataset.json"), json.dumps(metadata))
    close_dest()


if __name__ == "__main__":
    # e.g.
    # python wsgan_dataset_tool.py --dataset CIFAR10-B --dest_root ~/datasets/ --data_path ~/downloads/cifar/

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to load, one of: LSUN, CIFAR10-B, CIFAR10-lownoise",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory where pytorch datasets are stored or will be downloaded to.",
    )
    parser.add_argument(
        "--lfroot",
        type=str,
        default="./lfdata/",
        help="Directory that contains saved LFs and sample indices",
    )
    parser.add_argument(
        "--dest_root",
        type=str,
        required=True,
        help="Root directory where zip file will be written to.",
    )

    # Parse all arguments
    args = parser.parse_args()
    main(args)
