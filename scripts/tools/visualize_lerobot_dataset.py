# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _decode_image(cell: object) -> np.ndarray | None:
    """Decode a LeRobot image cell.

    Supports common formats:
    - {'bytes': <bytes>, 'path': 'frame-xxxx.png'}
    - raw bytes
    """
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


def _format_array(x: object, max_elems: int = 8) -> str:
    if isinstance(x, np.ndarray):
        flat = x.reshape(-1)
        show = flat[:max_elems]
        more = "" if flat.size <= max_elems else f" ... (+{flat.size - max_elems})"
        return f"np.ndarray{tuple(x.shape)} [{', '.join(f'{float(v):.3f}' for v in show)}]{more}"
    return repr(x)


def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, h, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    if ih == h:
        return img
    w = max(1, int(round(iw * (h / float(ih)))))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _draw_text_lines(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Overlay multiple text lines at the top-left corner (in-place)."""
    y = 22
    for line in lines:
        cv2.putText(img, line[:140], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line[:140], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple LeRobot dataset viewer (auto-play episodes).")
    ap.add_argument("--root", type=str, required=True, help="LeRobot dataset root (contains data/, meta/, images/).")
    ap.add_argument("--episode", type=int, default=0, help="Episode index to play.")
    ap.add_argument("--fps", type=float, default=20.0, help="Playback FPS.")
    ap.add_argument(
        "--list_episodes",
        action="store_true",
        help="只列出该数据集下的 episode 索引与任务文本，不进行回放。",
    )
    ap.add_argument(
        "--export_mp4",
        type=str,
        default="",
        help="If set (or if GUI is unavailable), export a preview mp4 to this path instead of interactive playback.",
    )
    ap.add_argument(
        "--image_keys",
        type=str,
        nargs="*",
        default=None,
        help="Which image keys to show (e.g. image wrist_image tactile_image). Default: auto-detect.",
    )
    ap.add_argument(
        "--value_keys",
        type=str,
        nargs="*",
        default=None,
        help="Which non-image keys to print (e.g. actions state tactile_gripper_force). Default: actions+state if present.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    # ------------------------------------------------------------------
    # Load episode meta
    # 支持两种布局：
    # 1) 旧版 LeRobot: meta/episodes/*.parquet 里包含 dataset_from_index / dataset_to_index
    # 2) 新版 v2.x:   meta/info.json + meta/episodes.jsonl, 每个 episode 一个独立 parquet:
    #       data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    # ------------------------------------------------------------------
    episodes: pd.DataFrame
    start: int
    end: int
    data_parquet: Path

    ep_meta_dir = root / "meta" / "episodes"
    ep_meta_files = sorted(ep_meta_dir.rglob("*.parquet")) if ep_meta_dir.exists() else []
    meta_mode: int
    info_json: Path | None = None

    if ep_meta_files:
        # ---- 模式 1：旧版 parquet meta ----
        meta_mode = 1
        episodes = pd.concat([pd.read_parquet(p) for p in ep_meta_files], ignore_index=True)
        if "episode_index" not in episodes.columns:
            raise ValueError("episodes meta missing column: episode_index")
    else:
        # ---- 模式 2：v2.1+ jsonl meta ----
        meta_mode = 2
        meta_dir = root / "meta"
        episodes_jsonl = meta_dir / "episodes.jsonl"
        info_json = meta_dir / "info.json"
        if not episodes_jsonl.exists() or not info_json.exists():
            raise FileNotFoundError(
                f"no episode meta parquet under: {ep_meta_dir} "
                f"and missing jsonl meta (expected {episodes_jsonl} and {info_json})"
            )

        # 读取 episodes.jsonl（每行一个 episode）
        episodes = pd.read_json(episodes_jsonl, lines=True)
        if "episode_index" not in episodes.columns:
            raise ValueError("episodes.jsonl missing column: episode_index")
        # v2.x 格式里每个 episode parquet 单独存储，length 表示帧数
        if "length" not in episodes.columns:
            raise ValueError("episodes.jsonl missing column: length (per-episode frame count)")

    # 若仅需列出 episode 目录，则在这里直接打印并退出
    if args.list_episodes:
        print(f"[viewer] dataset root: {root}")
        print(f"[viewer] total episodes: {len(episodes)}\n")
        cols = ["episode_index"]
        if "tasks" in episodes.columns:
            cols.append("tasks")
        if "length" in episodes.columns:
            cols.append("length")
        elif "dataset_to_index" in episodes.columns and "dataset_from_index" in episodes.columns:
            episodes = episodes.assign(length=episodes["dataset_to_index"] - episodes["dataset_from_index"])
            cols.append("length")

        # 只展示前若干行，避免终端太长
        to_show = episodes.sort_values("episode_index")[cols].head(50)
        print(to_show.to_string(index=False))
        if len(episodes) > len(to_show):
            print(f"\n... ({len(episodes)} episodes total, showing first {len(to_show)})")
        print("\n使用示例：")
        print(f"  python scripts/tools/visualize_lerobot_dataset.py --root {root} --episode <index>")
        return

    # ------------------------------------------------------------------
    # 选择指定 episode，并解析其数据文件路径 + 切片范围
    # ------------------------------------------------------------------
    row = episodes.loc[episodes["episode_index"] == int(args.episode)]
    if row.empty:
        available = sorted(episodes["episode_index"].tolist())[:20]
        raise ValueError(f"episode {args.episode} not found; available (first 20): {available} ...")
    row = row.iloc[0]

    # 任务文本（若存在）
    tasks = row.get("tasks", None)
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, list) and tasks:
        task_text = str(tasks[0])
    else:
        task_text = ""

    if meta_mode == 1:
        # 旧版：同一大 parquet 里按 [dataset_from_index, dataset_to_index) 切片
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        data_chunk = int(row.get("data/chunk_index", 0))
        data_file = int(row.get("data/file_index", 0))
        data_parquet = root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
        if not data_parquet.exists():
            # fallback: search all parquet and pick the first
            candidates = sorted((root / "data").rglob("*.parquet"))
            if not candidates:
                raise FileNotFoundError(f"no data parquet under: {root / 'data'}")
            data_parquet = candidates[0]
    else:
        # 新版：每个 episode 独立 parquet，length 即帧数
        assert info_json is not None
        info = pd.read_json(info_json, typ="series")
        data_path_tmpl = info.get("data_path", None)
        if not isinstance(data_path_tmpl, str):
            raise ValueError(f"invalid data_path in {info_json}: {data_path_tmpl!r}")

        episode_index = int(row["episode_index"])
        chunk_size = int(info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunk_size

        rel_path = data_path_tmpl.format(episode_chunk=episode_chunk, episode_index=episode_index)
        data_parquet = (root / rel_path).resolve()
        if not data_parquet.exists():
            raise FileNotFoundError(f"data parquet not found for episode {episode_index}: {data_parquet}")

        start = 0
        end = int(row["length"])

    # 读取该 episode 对应的数据切片
    df = pd.read_parquet(data_parquet)
    df = df.iloc[start:end].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"episode slice is empty: [{start},{end}) from {data_parquet}")

    # Auto-detect image keys if not provided
    if args.image_keys is None:
        image_keys: list[str] = []
        for c in df.columns:
            v = df[c].iloc[0]
            if isinstance(v, dict) and "bytes" in v:
                image_keys.append(c)
        if not image_keys:
            image_keys = [c for c in ("image", "wrist_image", "tactile_image") if c in df.columns]
    else:
        image_keys = list(args.image_keys)

    # Default value keys
    if args.value_keys is None:
        value_keys = [k for k in ("actions", "state") if k in df.columns]
    else:
        value_keys = list(args.value_keys)

    # Prefer matplotlib for interactive playback (your isaac5 env has it).
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    delay_ms = int(round(1000.0 / float(args.fps))) if args.fps > 0 else 0
    print(f"[viewer] root={root}")
    print(f"[viewer] episode={args.episode} frames={len(df)} data_parquet={data_parquet.name} slice=[{start},{end})")
    if task_text:
        print(f"[viewer] task: {task_text}")
    else:
        print("[viewer] task: (missing)")

    if plt is not None and not args.export_mp4:
        plt.ion()
        n_img = max(len(image_keys), 1)
        fig, axs = plt.subplots(1, n_img, figsize=(6 * n_img, 6))
        if n_img == 1:
            axs = [axs]
        for ax in axs:
            ax.axis("off")

        delay = 1.0 / float(args.fps) if args.fps > 0 else 0.0
        for t in range(len(df)):
            imgs: list[np.ndarray] = []
            for k in image_keys:
                img = _decode_image(df[k].iloc[t]) if k in df.columns else None
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                imgs.append(img)
            if not imgs:
                imgs = [np.zeros((224, 224, 3), dtype=np.uint8)]

            for i in range(len(axs)):
                im = imgs[i] if i < len(imgs) else np.zeros((224, 224, 3), dtype=np.uint8)
                axs[i].imshow(im)
                axs[i].set_title(image_keys[i] if i < len(image_keys) else "")
                axs[i].axis("off")

            overlay = [f"episode={args.episode} t={t}/{len(df)-1}"]
            if task_text:
                overlay.append(task_text)
            for k in value_keys:
                if k in df.columns:
                    overlay.append(f"{k}: {_format_array(df[k].iloc[t], max_elems=6)}")
            fig.suptitle(" | ".join(overlay)[:200])
            plt.pause(0.001)
            if delay > 0:
                time.sleep(delay)

        print("\n[viewer] done.")
        plt.ioff()
        plt.show()
        return

    # No matplotlib: require explicit mp4 export OR a GUI-capable OpenCV build.
    win = "LeRobot Viewer (q:quit, space:pause/resume)"
    use_gui = not bool(args.export_mp4)
    writer = None
    out_mp4 = Path(args.export_mp4).expanduser().resolve() if args.export_mp4 else (root / f"preview_episode{args.episode}.mp4")
    if use_gui:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            if not args.export_mp4:
                raise RuntimeError(
                    "当前环境既没有 matplotlib，也没有 OpenCV GUI（cv2 是 headless 构建）。\n"
                    "请切换到带 matplotlib 的环境（例如 conda activate isaac5）再运行。\n"
                    "或者显式传 --export_mp4 <path> 导出预览视频。"
                )
            use_gui = False
    if not use_gui:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
    paused = False
    for t in range(len(df)):
        # Compose images side-by-side
        imgs: list[np.ndarray] = []
        for k in image_keys:
            img = _decode_image(df[k].iloc[t]) if k in df.columns else None
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            imgs.append(img)
        if not imgs:
            imgs = [np.zeros((224, 224, 3), dtype=np.uint8)]
        h = max(im.shape[0] for im in imgs)
        imgs = [_resize_to_height(im, h) for im in imgs]
        canvas = cv2.hconcat(imgs)

        # Print values to terminal (avoid spamming too much)
        if t == 0 or (t % 10 == 0):
            print(f"\n[t={t:04d}/{len(df)-1:04d}]")
            if task_text:
                print("task:", task_text)
            for k in value_keys:
                if k in df.columns:
                    print(f"{k}: {_format_array(df[k].iloc[t])}")

        overlay = [f"episode={args.episode} t={t}/{len(df)-1}"] + ([f"task: {task_text}"] if task_text else [])
        for k in value_keys:
            if k in df.columns:
                overlay.append(f"{k}: {_format_array(df[k].iloc[t], max_elems=6)}")
        canvas = _draw_text_lines(canvas, overlay)

        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        if use_gui:
            cv2.imshow(win, bgr)
            key = cv2.waitKey(0 if paused else max(delay_ms, 1)) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
        else:
            if writer is None:
                h, w = bgr.shape[:2]
                w = w + (w % 2)
                h = h + (h % 2)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_mp4), fourcc, float(args.fps) if args.fps > 0 else 20.0, (w, h))
            h, w = writer.get(cv2.VIDEOWRITER_PROP_FRAME_HEIGHT), writer.get(cv2.VIDEOWRITER_PROP_FRAME_WIDTH)
            bgr2 = cv2.resize(bgr, (int(w), int(h)), interpolation=cv2.INTER_AREA)
            writer.write(bgr2)
            if (t + 1) % 50 == 0 or (t + 1) == len(df):
                print(f"[viewer] exporting mp4: {t+1}/{len(df)} frames -> {out_mp4}")

    print("\n[viewer] done.")
    if writer is not None:
        writer.release()
        print(f"[viewer] mp4 saved: {out_mp4}")
    if use_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


