import os
import random
import shutil
import threading
import uuid
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset


class ERA5_Dataset(Dataset):
    def __init__(
        self,
        input_dir,
        year_range,
        window_size=16,
        sample_k=4,
        look_ahead=2,
        is_train=True,
        dt_ref=6.0,
        sampling_mode="mixed",
    ):
        self.input_root = input_dir
        self.window_size = window_size
        self.sample_k = sample_k
        self.look_ahead = look_ahead
        self.is_train = is_train
        self.dt_ref = dt_ref
        self.sampling_mode = sampling_mode

        self.shm_root = f"/dev/shm/era5_cache/{uuid.uuid4().hex}"
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.all_info = []
        for year in range(year_range[0], year_range[1] + 1):
            y_dir = os.path.join(input_dir, str(year))
            if os.path.isdir(y_dir):
                months = sorted([f for f in os.listdir(y_dir) if f.endswith(".npy")])
                for m in months:
                    self.all_info.append({
                        "nfs_path": os.path.join(y_dir, m),
                        "shm_path": os.path.join(self.shm_root, str(year), m),
                        "id": f"{year}_{m}",
                    })

        self.file_frame_offsets = [0]
        self.file_shapes = []
        self._mmap_cache = OrderedDict()
        self._mmap_lock = threading.Lock()
        self._max_cache = 4

        for info in self.all_info:
            shape = self._get_shape(info["nfs_path"])
            self.file_shapes.append(shape)
            self.file_frame_offsets.append(self.file_frame_offsets[-1] + shape[0])

        self.total_frames = self.file_frame_offsets[-1]

    def _get_shape(self, path):
        arr = np.load(path, mmap_mode="r")
        return arr.shape

    def __len__(self):
        return max(0, self.total_frames - self.window_size - self.look_ahead + 1)

    def _get_data_ptr(self, file_idx):
        info = self.all_info[file_idx]

        with self._mmap_lock:
            if file_idx in self._mmap_cache:
                self._mmap_cache.move_to_end(file_idx)
                return self._mmap_cache[file_idx]

        shm_path = info["shm_path"]
        nfs_path = info["nfs_path"]

        if os.path.exists(shm_path):
            data = np.load(shm_path, mmap_mode="r")
        else:
            data = np.load(nfs_path, mmap_mode="r")

        with self._mmap_lock:
            self._mmap_cache[file_idx] = data
            while len(self._mmap_cache) > self._max_cache:
                self._mmap_cache.popitem(last=False)

        return data

    def _get_single_frame(self, global_idx):
        f_idx = np.searchsorted(self.file_frame_offsets, global_idx, side="right") - 1
        off = global_idx - self.file_frame_offsets[f_idx]
        return self._get_data_ptr(f_idx)[off]

    def __getitem__(self, idx):
        idx = int(idx)
        use_sequential = False

        if self.is_train:
            if self.sampling_mode == "sequential":
                use_sequential = True
            elif self.sampling_mode == "mixed":
                if random.random() < 0.5:
                    use_sequential = True

            if use_sequential:
                max_start = self.window_size - self.sample_k
                start_off = random.randint(0, max(0, max_start))
                offsets = list(range(start_off, start_off + self.sample_k))
            else:
                offsets = sorted(random.sample(range(self.window_size), self.sample_k))
        else:
            use_sequential = True
            offsets = list(range(self.sample_k))

        start_global = idx + offsets[0]
        f_idx = np.searchsorted(self.file_frame_offsets, start_global, side="right") - 1
        current_file_data = self._get_data_ptr(f_idx)
        file_start_global = self.file_frame_offsets[f_idx]
        file_len = self.file_shapes[f_idx][0]
        start_local = start_global - file_start_global

        if use_sequential and (start_local + self.sample_k <= file_len):
            chunk = current_file_data[start_local : start_local + self.sample_k]
            data = torch.from_numpy(chunk.copy())
        else:
            frames = []
            for off in offsets:
                curr_global = idx + off
                local_off = curr_global - file_start_global
                if 0 <= local_off < file_len:
                    frames.append(torch.from_numpy(current_file_data[local_off].copy()))
                else:
                    frame_data = self._get_single_frame(curr_global)
                    frames.append(torch.from_numpy(frame_data.copy()))
            data = torch.stack(frames, dim=0)

        offsets_tensor = torch.tensor(offsets, dtype=torch.float32)
        dt = (offsets_tensor[1:] - offsets_tensor[:-1]) * float(self.dt_ref)
        dt_padded = torch.cat([dt, dt[-1:]], dim=0)

        return data, dt_padded

    def cleanup(self):
        if os.path.exists(self.shm_root):
            shutil.rmtree(self.shm_root, ignore_errors=True)

    def __del__(self):
        self.cleanup()

