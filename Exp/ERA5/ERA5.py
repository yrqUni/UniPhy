import os
import random
import threading
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset


class ERA5Dataset(Dataset):
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

        self.all_info = []
        for year in range(year_range[0], year_range[1] + 1):
            y_dir = os.path.join(input_dir, str(year))
            if not os.path.isdir(y_dir):
                continue
            months = sorted(
                f for f in os.listdir(y_dir) if f.endswith(".npy")
            )
            for m in months:
                self.all_info.append(os.path.join(y_dir, m))

        self.file_frame_offsets = [0]
        self.file_shapes = []
        for path in self.all_info:
            shape = np.load(path, mmap_mode="r").shape
            self.file_shapes.append(shape)
            self.file_frame_offsets.append(
                self.file_frame_offsets[-1] + shape[0]
            )

        self.total_frames = self.file_frame_offsets[-1]

        self._mmap_cache = OrderedDict()
        self._mmap_lock = threading.Lock()
        self._max_cache = 4

    def __len__(self):
        return max(
            0, self.total_frames - self.window_size - self.look_ahead + 1,
        )

    def _get_data_ptr(self, file_idx):
        with self._mmap_lock:
            if file_idx in self._mmap_cache:
                self._mmap_cache.move_to_end(file_idx)
                return self._mmap_cache[file_idx]

        data = np.load(self.all_info[file_idx], mmap_mode="r")

        with self._mmap_lock:
            if file_idx not in self._mmap_cache:
                self._mmap_cache[file_idx] = data
                while len(self._mmap_cache) > self._max_cache:
                    self._mmap_cache.popitem(last=False)
            else:
                data = self._mmap_cache[file_idx]

        return data

    def _get_single_frame(self, global_idx):
        f_idx = (
            np.searchsorted(
                self.file_frame_offsets, global_idx, side="right"
            )
            - 1
        )
        off = global_idx - self.file_frame_offsets[f_idx]
        return self._get_data_ptr(f_idx)[off]

    def _sample_offsets(self):
        if not self.is_train:
            return list(range(self.sample_k)), True

        use_sequential = False
        if self.sampling_mode == "sequential":
            use_sequential = True
        elif self.sampling_mode == "mixed":
            use_sequential = random.random() < 0.5

        if use_sequential:
            max_start = self.window_size - self.sample_k
            start_off = random.randint(0, max(0, max_start))
            offsets = list(range(start_off, start_off + self.sample_k))
        else:
            offsets = sorted(
                random.sample(range(self.window_size), self.sample_k)
            )

        return offsets, use_sequential

    def _load_frames(self, idx, offsets, use_sequential):
        start_global = idx + offsets[0]
        f_idx = (
            np.searchsorted(
                self.file_frame_offsets, start_global, side="right"
            )
            - 1
        )
        file_data = self._get_data_ptr(f_idx)
        file_start = self.file_frame_offsets[f_idx]
        file_len = self.file_shapes[f_idx][0]
        start_local = start_global - file_start

        if use_sequential and (start_local + self.sample_k <= file_len):
            chunk = file_data[start_local: start_local + self.sample_k]
            return torch.from_numpy(chunk.copy())

        frames = []
        for off in offsets:
            curr_global = idx + off
            local_off = curr_global - file_start
            if 0 <= local_off < file_len:
                frames.append(
                    torch.from_numpy(file_data[local_off].copy())
                )
            else:
                frame_data = self._get_single_frame(curr_global)
                frames.append(torch.from_numpy(frame_data.copy()))
        return torch.stack(frames, dim=0)

    def _compute_dt(self, offsets):
        offsets_f = torch.tensor(offsets, dtype=torch.float32)
        gaps = (offsets_f[1:] - offsets_f[:-1]) * self.dt_ref
        dt_first = torch.tensor([self.dt_ref], dtype=torch.float32)
        return torch.cat([dt_first, gaps], dim=0)

    def __getitem__(self, idx):
        idx = int(idx)
        offsets, use_sequential = self._sample_offsets()
        data = self._load_frames(idx, offsets, use_sequential)
        dt = self._compute_dt(offsets)
        return data, dt

    def cleanup(self):
        pass
    