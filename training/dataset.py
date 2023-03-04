# This code was modified from its original source by Benedikt Boecking.
# The original source code can be found at https://github.com/NVlabs/stylegan2-ada-pytorch
# The original source code is distributed under the following license, which still applies to this modified version:

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

# ----------------------------------------------------------------------------


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,  # Name of the dataset.
        raw_shape,  # Shape of the raw image data (NCHW).
        max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
        xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed=0,  # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self._raw_lfs_oh = None
        self._raw_lfs = None
        self._raw_lfs_filter = None
        self.numlfs = None
        # LF matrix can be quite large (samples x num LFs x num classes)
        # so we load them when we need them.
        self.load_one_hot_lfs = False
        self.lazyload = True

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _do_load_raw_labels(self):
        (
            self._raw_labels,
            self._raw_lfs_oh,
            self._raw_lfs_filter,
            self._raw_lfs,
        ) = self._load_raw_labels()
        if self._raw_labels is None:
            self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
        assert isinstance(self._raw_labels, np.ndarray)
        assert self._raw_labels.shape[0] == self._raw_shape[0]
        assert self._raw_lfs_filter.shape[0] == self._raw_shape[0]
        if self._raw_lfs_oh is not None:
            assert self._raw_lfs_oh.shape[0] == self._raw_shape[0]
        if self._raw_lfs is not None:
            assert self._raw_lfs.shape[0] == self._raw_shape[0]
        assert self._raw_labels.dtype in [np.float32, np.int64]
        if self._raw_labels.dtype == np.int64:
            assert self._raw_labels.ndim == 1
            assert np.all(self._raw_labels >= 0)

    def _get_raw_lfs(self):
        if self.load_one_hot_lfs:
            if self._raw_lfs_oh is None:
                self._do_load_raw_labels()
            return self._raw_lfs_oh
        else:
            if self._raw_lfs is None:
                self._do_load_raw_labels()
            return self._raw_lfs

    def _get_raw_lfs_filter(self):
        if self._raw_lfs_filter is None:
            self._do_load_raw_labels()
        return self._raw_lfs_filter

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._do_load_raw_labels()
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def get_lfs(idx):
        return None

    def get_lfs_filter(idx):
        return None

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        lbl = self.get_label(idx)
        lfs = self.get_lfs(idx)
        lfsfilter = self.get_lfs_filter(idx)
        return image.copy(), lbl, lfs, lfsfilter

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = int(self._xflip[idx]) != 0
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        resolution=None,  # Ensure specific resolution, None = highest available.
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()
        self._image_fnames = sorted(
            fname
            for fname in self._all_fnames
            if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (
            raw_shape[2] != resolution or raw_shape[3] != resolution
        ):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == ".png":
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def get_numLFs(self):
        return self.numlfs

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None, None, None
        # lfs = None
        lfs_filter = None
        lfs_oh = None
        lfs = None
        with self._open_file(fname) as f:
            jsondata = json.load(f)
            labels = jsondata["labels"]
            if (
                "lfs" in jsondata
                and (not self.load_one_hot_lfs)
                and (not self.lazyload)
            ):
                lfs = jsondata["lfs"]
            if "lfs_oh" in jsondata and self.load_one_hot_lfs and (not self.lazyload):
                lfs_oh = jsondata["lfs_oh"]
            if "lfs_filter" in jsondata:
                lfs_filter = jsondata["lfs_filter"]

        if lfs is not None:
            lfs = dict(lfs)
            lfs = [lfs[fname.replace("\\", "/")] for fname in self._image_fnames]
            lfs = np.array(lfs).astype(np.int64)
        if lfs_filter is not None:
            lfs_filter = dict(lfs_filter)
            lfs_filter = [
                lfs_filter[fname.replace("\\", "/")] for fname in self._image_fnames
            ]
            lfs_filter = np.array(lfs_filter).astype(bool)
        if lfs_oh is not None:
            lfs_oh = dict(lfs_oh)
            lfs_oh = [lfs_oh[fname.replace("\\", "/")] for fname in self._image_fnames]
            lfs_oh = np.array(lfs_oh).astype(np.float32)
        if not self.lazyload:
            assert (lfs_oh is not None) or (
                lfs is not None
            ), "unable to load any labeling functions from dataset"
            if self.load_one_hot_lfs:
                self.numlfs = lfs_oh.shape[1]
            else:
                self.numlfs = lfs.shape[1]
        else:
            # load an LF, determine number of Lfs
            fname = "%s.json" % self._image_fnames[0][:-4]
            with self._open_file(fname) as f:
                s = f.read().decode("utf-8")
                lfs = [float(x) for x in s.strip("[]").split(",")]
                self.numlfs = len(lfs)

        if labels is None:
            return None, lfs_oh, lfs_filter, lfs
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels, lfs_oh, lfs_filter, lfs

    def get_lfs(self, idx):
        raw_idx = self._raw_idx[idx]
        if self.lazyload:
            fname = "%s.txt" % self._image_fnames[raw_idx][:-4]
            with self._open_file(fname) as f:
                s = f.read().decode("utf-8")
                lfs = np.array(
                    [float(x) for x in s.strip("[]").split(",")], dtype=np.int64
                )
            return lfs.copy()
        else:
            lfs = self._get_raw_lfs()[raw_idx]
            return lfs.copy()

    def get_lfs_filter(self, idx):
        lfs_filter = self._get_raw_lfs_filter()[self._raw_idx[idx]]
        return lfs_filter.copy()


# ----------------------------------------------------------------------------
