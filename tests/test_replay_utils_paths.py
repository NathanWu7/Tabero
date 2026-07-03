from pathlib import Path

from scripts.tools.common.replay_utils import setup_replay_output_directories


def test_replay_output_directory_joins_root_without_trailing_separator(tmp_path: Path):
    video_root = tmp_path / "video_datasets"

    video_save_dir, tactile_outputs_save_dir = setup_replay_output_directories(
        True,
        task_suite="libero_goal",
        task_id=1,
        root_dir_prefix=str(video_root),
    )

    expected = video_root / "libero_goal_task1" / "videos"
    assert Path(video_save_dir) == expected
    assert expected.is_dir()
    assert tactile_outputs_save_dir is None
