import math
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
        sample_offsets_hours=None,
    ):
        self.input_root = input_dir
        self.window_size = window_size
        self.sample_k = sample_k
        self.look_ahead = look_ahead
        self.is_train = is_train
        self.dt_ref = float(dt_ref)
        self.sample_offsets_hours = None
        self.sample_positions = None

        if sample_offsets_hours is not None:
            self.sample_offsets_hours = [
                float(offset) for offset in sample_offsets_hours
            ]
            if len(self.sample_offsets_hours) != self.sample_k:
                raise ValueError(
                    "sample_offsets_hours length must match sample_k"
                )
            if any(offset < 0.0 for offset in self.sample_offsets_hours):
                raise ValueError("sample_offsets_hours must be non-negative")
            if any(
                curr <= prev
                for prev, curr in zip(
                    self.sample_offsets_hours[:-1],
                    self.sample_offsets_hours[1:],
                )
            ):
                raise ValueError(
                    "sample_offsets_hours must be strictly increasing"
                )
            self.sample_positions = [
                offset / self.dt_ref for offset in self.sample_offsets_hours
            ]

        self.all_info = []
        for year in range(year_range[0], year_range[1] + 1):
            y_dir = os.path.join(input_dir, str(year))
            if not os.path.isdir(y_dir):
                continue
            months = sorted(f for f in os.listdir(y_dir) if f.endswith(".npy"))
            for month_name in months:
                self.all_info.append(os.path.join(y_dir, month_name))

        self.file_frame_offsets = [0]
        self.file_shapes = []
        for path in self.all_info:
            shape = np.load(path, mmap_mode="r").shape
            self.file_shapes.append(shape)
            self.file_frame_offsets.append(
                self.file_frame_offsets[-1] + shape[0]
            )

        self.total_frames = self.file_frame_offsets[-1]
        if self.sample_positions is None:
            self.max_offset = float(max(0, self.window_size - 1))
        else:
            self.max_offset = float(max(self.sample_positions))

        self._mmap_cache = OrderedDict()
        self._mmap_lock = threading.Lock()
        self._max_cache = 4

    def __len__(self):
        required_span = int(math.ceil(self.max_offset))
        return max(0, self.total_frames - required_span - self.look_ahead)

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
        file_idx = (
            np.searchsorted(self.file_frame_offsets, global_idx, side="right")
            - 1
        )
        offset = global_idx - self.file_frame_offsets[file_idx]
        return self._get_data_ptr(file_idx)[offset]

    def _sample_offsets(self):
        if self.sample_positions is not None:
            return list(self.sample_positions), False

        if not self.is_train:
            return list(range(self.sample_k)), True

        if self.look_ahead == 0:
            max_start = self.window_size - self.sample_k
            start_off = random.randint(0, max(0, max_start))
            offsets = list(range(start_off, start_off + self.sample_k))
            return offsets, True

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

    def _load_frame_at_position(self, global_position):
        lower_idx = int(math.floor(global_position))
        upper_idx = int(math.ceil(global_position))
        interp_weight = float(global_position - lower_idx)

        if upper_idx == lower_idx or abs(interp_weight) < 1e-8:
            frame = self._get_single_frame(lower_idx)
            return torch.from_numpy(frame.copy())

        lower = torch.from_numpy(
            self._get_single_frame(lower_idx).astype(np.float32, copy=True)
        )
        upper = torch.from_numpy(
            self._get_single_frame(upper_idx).astype(np.float32, copy=True)
        )
        return lower * (1.0 - interp_weight) + upper * interp_weight

    def _load_frames(self, idx, offsets, use_sequential):
        offsets_are_int = all(
            abs(offset - round(offset)) < 1e-8 for offset in offsets
        )

        if use_sequential and offsets_are_int:
            start_global = idx + int(round(offsets[0]))
            file_idx = (
                np.searchsorted(
                    self.file_frame_offsets, start_global, side="right"
                )
                - 1
            )
            file_data = self._get_data_ptr(file_idx)
            file_start = self.file_frame_offsets[file_idx]
            file_len = self.file_shapes[file_idx][0]
            start_local = start_global - file_start

            if start_local + self.sample_k <= file_len:
                chunk = file_data[start_local:start_local + self.sample_k]
                return torch.from_numpy(chunk.copy())

        frames = []
        for offset in offsets:
            curr_global = idx + float(offset)
            frames.append(self._load_frame_at_position(curr_global))
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
