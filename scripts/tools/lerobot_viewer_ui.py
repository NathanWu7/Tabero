# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button


def _is_lerobot_root(p: Path) -> bool:
    """判断是否为 LeRobot 数据集根目录（兼容旧版 parquet 与 v2.1 JSONL 布局）。"""
    meta = p / "meta"
    data = p / "data"
    if not data.exists():
        return False
    # 旧版：meta/episodes/*.parquet
    if (meta / "episodes").exists():
        return True
    # 新版：meta/episodes.jsonl + meta/info.json
    if (meta / "episodes.jsonl").exists() and (meta / "info.json").exists():
        return True
    return False


def _scan_datasets(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    out: list[Path] = []
    for d in sorted([x for x in parent.iterdir() if x.is_dir()]):
        if _is_lerobot_root(d):
            out.append(d)
    return out


def _decode_image(cell: object) -> np.ndarray | None:
    """Decode a LeRobot image cell into RGB uint8 HxWx3."""
    if cell is None:
        return None
    b: bytes | None = None
    if isinstance(cell, dict):
        b = cell.get("bytes", None)
    elif isinstance(cell, (bytes, bytearray)):
        b = bytes(cell)
    if not b:
        return None
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _format_array(x: object, max_elems: int = 6) -> str:
    if isinstance(x, np.ndarray):
        flat = x.reshape(-1)
        show = flat[:max_elems]
        parts: list[str] = []
        for v in show:
            try:
                if isinstance(v, np.ndarray) and v.size == 1:
                    parts.append(f"{float(v.item()):.3f}")
                else:
                    parts.append(f"{float(v):.3f}")
            except Exception:
                parts.append(str(v))
        more = "" if flat.size <= max_elems else f" ...(+{flat.size - max_elems})"
        return f"{tuple(x.shape)} [{', '.join(parts)}]{more}"
    return str(x)


def _shape_and_mean(x: object) -> tuple[str, float | None]:
    """Best-effort summary for nested/object ndarrays: return (shape_str, mean_or_None)."""
    if isinstance(x, np.ndarray):
        shape_str = str(tuple(x.shape))
        try:
            arr = np.asarray(x, dtype=np.float32)
            return shape_str, float(arr.mean())
        except Exception:
            # Common case in parquet: object arrays of object arrays -> flatten recursively
            try:
                leaves: list[np.ndarray] = []

                def _collect(v: object) -> None:
                    if isinstance(v, np.ndarray) and v.dtype == object:
                        for item in v.reshape(-1):
                            _collect(item)
                    elif isinstance(v, np.ndarray):
                        leaves.append(v.astype(np.float32, copy=False).reshape(-1))
                    else:
                        try:
                            leaves.append(np.array([float(v)], dtype=np.float32))
                        except Exception:
                            return

                _collect(x)
                if not leaves:
                    return shape_str, None
                flat = np.concatenate(leaves, axis=0)
                return shape_str, float(flat.mean())
            except Exception:
                return shape_str, None
    return type(x).__name__, None

@dataclass
class EpisodeSlice:
    episode_index: int
    task_text: str
    data_parquet: Path
    start: int
    end: int


class LeRobotViewerUI:
    def __init__(self, datasets: list[Path], fps: float = 20.0) -> None:
        if not datasets:
            raise ValueError("No datasets provided.")
        self.datasets = datasets
        self.fps = float(fps)
        self.interval_ms = int(round(1000.0 / self.fps)) if self.fps > 0 else 50

        self.dataset_idx = 0
        self.episodes_df: pd.DataFrame | None = None
        self.episode_indices: list[int] = []
        self.episode_scroll = 0
        self.selected_episode_idx = 0  # index into episode_indices
        self.file_offsets: dict[tuple[int, int], int] = {}
        # meta_mode: 1 = 旧版 parquet meta, 2 = v2.1 JSONL meta
        self.meta_mode: int = 1
        self.info_series: pd.Series | None = None

        self.ep: EpisodeSlice | None = None
        self.df: pd.DataFrame | None = None
        self.t = 0
        self.playing = True

        # UI state
        self.fig = plt.figure(figsize=(14, 7))
        self.ax_ds = self.fig.add_axes([0.02, 0.10, 0.18, 0.85])   # dataset list
        self.ax_ep = self.fig.add_axes([0.21, 0.10, 0.12, 0.85])   # episode list
        self.ax_img1 = self.fig.add_axes([0.35, 0.35, 0.20, 0.55])
        self.ax_img2 = self.fig.add_axes([0.56, 0.35, 0.20, 0.55])
        self.ax_img3 = self.fig.add_axes([0.77, 0.35, 0.20, 0.55])
        self.ax_text = self.fig.add_axes([0.35, 0.10, 0.62, 0.20])

        self.ax_btn_play = self.fig.add_axes([0.35, 0.01, 0.10, 0.06])
        self.ax_btn_pause = self.fig.add_axes([0.46, 0.01, 0.10, 0.06])
        self.ax_btn_resume = self.fig.add_axes([0.57, 0.01, 0.10, 0.06])
        self.ax_btn_prev = self.fig.add_axes([0.68, 0.01, 0.10, 0.06])
        self.ax_btn_next = self.fig.add_axes([0.79, 0.01, 0.10, 0.06])

        # UI labels must be English (project convention).
        self.btn_play = Button(self.ax_btn_play, "Play")
        self.btn_pause = Button(self.ax_btn_pause, "Pause")
        self.btn_resume = Button(self.ax_btn_resume, "Resume")
        self.btn_prev = Button(self.ax_btn_prev, "Prev")
        self.btn_next = Button(self.ax_btn_next, "Next")

        self.btn_play.on_clicked(lambda _e: self._set_play(True))
        self.btn_pause.on_clicked(lambda _e: self._set_play(False))
        self.btn_resume.on_clicked(lambda _e: self._set_play(True))
        self.btn_prev.on_clicked(lambda _e: self._step(-1))
        self.btn_next.on_clicked(lambda _e: self._step(+1))

        for ax in [self.ax_img1, self.ax_img2, self.ax_img3, self.ax_ds, self.ax_ep, self.ax_text]:
            ax.set_xticks([])
            ax.set_yticks([])

        self.text_obj = self.ax_text.text(0.0, 1.0, "", va="top", ha="left", fontsize=10, family="monospace")
        self.img_objs = [
            self.ax_img1.imshow(np.zeros((224, 224, 3), dtype=np.uint8)),
            self.ax_img2.imshow(np.zeros((224, 224, 3), dtype=np.uint8)),
            self.ax_img3.imshow(np.zeros((224, 224, 3), dtype=np.uint8)),
        ]
        self.ax_img1.set_title("image")
        self.ax_img2.set_title("wrist_image")
        self.ax_img3.set_title("tactile_image")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._on_timer)

        self._load_dataset(0)
        self._load_episode_by_list_index(0)
        self._redraw_lists()
        self._render()
        self.timer.start()

    def _set_play(self, playing: bool) -> None:
        self.playing = bool(playing)

    def _on_timer(self) -> None:
        if self.playing:
            self._step(+1)
        self._render()

    def _step(self, delta: int) -> None:
        if self.df is None:
            return
        self.t = int(np.clip(self.t + int(delta), 0, len(self.df) - 1))

    def _load_dataset(self, idx: int) -> None:
        self.dataset_idx = int(np.clip(idx, 0, len(self.datasets) - 1))
        root = self.datasets[self.dataset_idx]
        meta_dir = root / "meta"
        # 优先尝试旧版 parquet meta
        ep_meta_dir = meta_dir / "episodes"
        ep_meta_files = sorted(ep_meta_dir.rglob("*.parquet")) if ep_meta_dir.exists() else []

        if ep_meta_files:
            # ---- 模式 1：旧版 parquet meta ----
            self.meta_mode = 1
            self.info_series = None
            episodes = pd.concat([pd.read_parquet(p) for p in ep_meta_files], ignore_index=True)
            if "episode_index" not in episodes.columns:
                raise ValueError(f"episodes meta missing column: episode_index in dataset {root}")

            self.episodes_df = episodes
            self.episode_indices = sorted([int(x) for x in episodes["episode_index"].unique().tolist()])

            # Build per-(chunk,file) base offsets so we can convert global dataset indices to file-local indices.
            # LeRobot episodes meta uses dataset_from_index/dataset_to_index in *global* dataset coordinates.
            self.file_offsets = {}
            if (
                "data/chunk_index" in episodes.columns
                and "data/file_index" in episodes.columns
                and "dataset_from_index" in episodes.columns
            ):
                grp = (
                    episodes.groupby(["data/chunk_index", "data/file_index"])["dataset_from_index"]
                    .min()
                    .reset_index()
                )
                for _, r in grp.iterrows():
                    self.file_offsets[(int(r["data/chunk_index"]), int(r["data/file_index"]))] = int(
                        r["dataset_from_index"]
                    )
        else:
            # ---- 模式 2：v2.1 JSONL meta ----
            self.meta_mode = 2
            episodes_jsonl = meta_dir / "episodes.jsonl"
            info_json = meta_dir / "info.json"
            if not episodes_jsonl.exists() or not info_json.exists():
                raise ValueError(
                    f"Dataset {root} does not contain valid LeRobot meta "
                    f"(missing {ep_meta_dir}/*.parquet and {episodes_jsonl}/{info_json})."
                )

            episodes = pd.read_json(episodes_jsonl, lines=True)
            if "episode_index" not in episodes.columns:
                raise ValueError(f"episodes.jsonl missing column: episode_index in dataset {root}")
            if "length" not in episodes.columns:
                raise ValueError(f"episodes.jsonl missing column: length in dataset {root}")

            self.episodes_df = episodes
            self.episode_indices = sorted([int(x) for x in episodes["episode_index"].tolist()])
            self.file_offsets = {}
            # 读取 info.json（包含 data_path / chunks_size 等）
            self.info_series = pd.read_json(info_json, typ="series")

        self.episode_scroll = 0
        self.selected_episode_idx = 0

    def _get_episode_slice(self, episode_index: int) -> EpisodeSlice:
        assert self.episodes_df is not None
        root = self.datasets[self.dataset_idx]
        row = self.episodes_df.loc[self.episodes_df["episode_index"] == int(episode_index)]
        if row.empty:
            raise ValueError(f"episode not found: {episode_index}")
        r = row.iloc[0]

        tasks = r.get("tasks", None)
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        task_text = str(tasks[0]) if isinstance(tasks, list) and tasks else ""

        if self.meta_mode == 1:
            # 旧版：同一大 parquet 中通过全局索引切片
            start_global = int(r["dataset_from_index"])
            end_global = int(r["dataset_to_index"])
            data_chunk = int(r.get("data/chunk_index", 0))
            data_file = int(r.get("data/file_index", 0))
            base = int(self.file_offsets.get((data_chunk, data_file), 0))
            start = max(0, start_global - base)
            end = max(start, end_global - base)
            data_parquet = root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
            if not data_parquet.exists():
                # fallback: first parquet
                candidates = sorted((root / "data").rglob("*.parquet"))
                if not candidates:
                    raise FileNotFoundError(f"no data parquet under: {root / 'data'}")
                data_parquet = candidates[0]
        else:
            # 新版：每个 episode 独立 parquet，length 即帧数
            if self.info_series is None:
                raise RuntimeError("info_series is None while meta_mode == 2")
            info = self.info_series
            data_path_tmpl = info.get("data_path", None)
            if not isinstance(data_path_tmpl, str):
                raise ValueError(f"invalid data_path in info.json for dataset {root}: {data_path_tmpl!r}")

            ep_idx = int(r["episode_index"])
            chunk_size = int(info.get("chunks_size", 1000))
            episode_chunk = ep_idx // chunk_size

            rel_path = data_path_tmpl.format(episode_chunk=episode_chunk, episode_index=ep_idx)
            data_parquet = (root / rel_path).resolve()
            if not data_parquet.exists():
                raise FileNotFoundError(f"data parquet not found for episode {ep_idx}: {data_parquet}")

            start = 0
            end = int(r["length"])

        return EpisodeSlice(int(episode_index), task_text, data_parquet, start, end)

    def _load_episode_by_list_index(self, list_idx: int) -> None:
        if not self.episode_indices:
            return
        self.selected_episode_idx = int(np.clip(list_idx, 0, len(self.episode_indices) - 1))
        ep_idx = self.episode_indices[self.selected_episode_idx]
        self.ep = self._get_episode_slice(ep_idx)
        df = pd.read_parquet(self.ep.data_parquet)
        self.df = df.iloc[self.ep.start : self.ep.end].reset_index(drop=True)
        self.t = 0

    def _redraw_lists(self) -> None:
        # Dataset list
        self.ax_ds.clear()
        self.ax_ds.set_xticks([])
        self.ax_ds.set_yticks([])
        self.ax_ds.set_title("Datasets")
        for i, p in enumerate(self.datasets):
            name = p.name
            prefix = "▶ " if i == self.dataset_idx else "  "
            self.ax_ds.text(
                0.0,
                1.0 - 0.05 * i,
                f"{prefix}{name}",
                va="top",
                ha="left",
                fontsize=10,
                transform=self.ax_ds.transAxes,
            )

        # Episode list (scrollable)
        self.ax_ep.clear()
        self.ax_ep.set_xticks([])
        self.ax_ep.set_yticks([])
        self.ax_ep.set_title("Episodes")
        per_page = 30
        self.episode_scroll = int(np.clip(self.episode_scroll, 0, max(0, len(self.episode_indices) - per_page)))
        for j in range(per_page):
            i = self.episode_scroll + j
            if i >= len(self.episode_indices):
                break
            ep = self.episode_indices[i]
            selected = (i == self.selected_episode_idx)
            prefix = "▶" if selected else " "
            self.ax_ep.text(
                0.0,
                1.0 - 0.03 * j,
                f"{prefix} {ep}",
                va="top",
                ha="left",
                fontsize=10,
                transform=self.ax_ep.transAxes,
            )

        self.fig.canvas.draw_idle()

    def _render(self) -> None:
        if self.df is None or self.ep is None:
            return
        row = self.df.iloc[self.t]

        # Image keys: prefer common three
        keys = [k for k in ("image", "wrist_image", "tactile_image") if k in self.df.columns]
        while len(keys) < 3:
            keys.append("")
        for i in range(3):
            k = keys[i]
            img = _decode_image(row[k]) if k else None
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            self.img_objs[i].set_data(img)
            if i == 0:
                self.ax_img1.set_title(k or "image")
            elif i == 1:
                self.ax_img2.set_title(k or "wrist_image")
            else:
                self.ax_img3.set_title(k or "tactile_image")

        # Text panel
        lines: list[str] = [
            f"dataset: {self.datasets[self.dataset_idx].name}",
            f"episode: {self.ep.episode_index}  frame: {self.t}/{len(self.df)-1}  playing: {self.playing}",
        ]
        if self.ep.task_text:
            lines.append(f"task: {self.ep.task_text}")
        for k in ("actions", "state"):
            if k in self.df.columns:
                lines.append(f"{k}: {_format_array(row[k])}")
        if "tactile_gripper_force" in self.df.columns:
            _shp, mean = _shape_and_mean(row["tactile_gripper_force"])
            if mean is None:
                lines.append("tactile_gripper_force: mean=N/A")
            else:
                lines.append(f"tactile_gripper_force: mean={mean:.6f}")
        if "tactile_marker_motion" in self.df.columns:
            shp, mean = _shape_and_mean(row["tactile_marker_motion"])
            if mean is None:
                lines.append(f"tactile_marker_motion: shape={shp} mean=N/A")
            else:
                lines.append(f"tactile_marker_motion: shape={shp} mean={mean:.6f}")
        self.text_obj.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()

    def _on_click(self, event: Any) -> None:
        def _index_from_click(ax, item_h: float) -> int:
            if event.y is None:
                return -1
            bbox = ax.get_window_extent()
            if bbox.height <= 0:
                return -1
            y_frac = (event.y - bbox.y0) / bbox.height  # 0(bottom)..1(top)
            if not (0.0 <= y_frac <= 1.0):
                return -1
            return int((1.0 - y_frac) / float(item_h))

        if event.inaxes == self.ax_ds:
            i = _index_from_click(self.ax_ds, 0.05)
            if 0 <= i < len(self.datasets):
                self._load_dataset(i)
                self._load_episode_by_list_index(0)
                self._redraw_lists()
                self._render()
        elif event.inaxes == self.ax_ep:
            per_page = 30
            j = _index_from_click(self.ax_ep, 0.03)
            i = self.episode_scroll + j
            if 0 <= j < per_page and 0 <= i < len(self.episode_indices):
                self._load_episode_by_list_index(i)
                self._redraw_lists()
                self._render()

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes != self.ax_ep:
            return
        step = -3 if event.button == "up" else 3
        self.episode_scroll += step
        self._redraw_lists()


def main() -> None:
    ap = argparse.ArgumentParser(description="LeRobot dataset viewer UI (matplotlib).")
    ap.add_argument(
        "--datasets_parent",
        type=str,
        default="",
        help="Parent directory containing multiple LeRobot dataset roots (each has meta/ and data/).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Single LeRobot dataset root. If provided, viewer will include only this dataset.",
    )
    ap.add_argument("--fps", type=float, default=20.0, help="Playback fps.")
    args = ap.parse_args()

    datasets: list[Path] = []
    if args.dataset:
        p = Path(args.dataset).expanduser().resolve()
        if not _is_lerobot_root(p):
            raise ValueError(f"Not a LeRobot dataset root: {p}")
        datasets = [p]
    else:
        parent = Path(args.datasets_parent).expanduser().resolve() if args.datasets_parent else Path.cwd()
        datasets = _scan_datasets(parent)
        if not datasets:
            raise ValueError(
                f"No datasets found under: {parent} "
                f"(need either meta/episodes/*.parquet or meta/episodes.jsonl+info.json, plus data/)"
            )

    _ = LeRobotViewerUI(datasets=datasets, fps=float(args.fps))
    plt.show()


if __name__ == "__main__":
    main()


