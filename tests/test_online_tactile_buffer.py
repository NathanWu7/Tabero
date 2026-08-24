import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "benchmarks/openpi/openpi_inference_client.py"


def _load_online_tactile_buffer():
    """Load the buffer without executing the client's top-level Isaac launcher."""
    tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
    required_definitions = {
        "_to_uint8_rgb",
        "_pad_history_front",
        "_build_tactile_mosaic",
        "_OnlineTactileBuffer",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in required_definitions
    ]
    namespace = {
        "TARGET_IMAGE_HW": (224, 224),
        "cv2": cv2,
        "deque": deque,
        "np": np,
        "torch": torch,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), CLIENT_PATH, "exec"), namespace)
    return namespace["_OnlineTactileBuffer"]


def _observation(step: int) -> dict:
    marker = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
    marker[:, :, 1, :, :] = float(step)
    return {
        "policy": {
            "gripper_net_force": torch.full((1, 1, 2, 3), float(step)),
            "gripper_marker_motion": torch.from_numpy(marker),
        }
    }


def _environment(step: int) -> SimpleNamespace:
    def sensor() -> SimpleNamespace:
        frame = torch.full((1, 4, 4, 3), step, dtype=torch.uint8)
        return SimpleNamespace(data=SimpleNamespace(output={"tactile_rgb": frame}))

    scene = SimpleNamespace(sensors={"left": sensor(), "right": sensor()})
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))


def _is_tactile_buffer_call(node: ast.AST, method: str) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "tactile_buf"
        and node.value.func.attr == method
    )


def _is_env_step_assignment(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "env"
        and node.value.func.attr == "step"
    )


def _statement_lists(node: ast.AST):
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            if value and all(isinstance(item, ast.stmt) for item in value):
                yield value
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _statement_lists(item)
        elif isinstance(value, ast.AST):
            yield from _statement_lists(value)


def test_control_step_sampling_keeps_last_eight_consecutive_frames() -> None:
    online_tactile_buffer = _load_online_tactile_buffer()
    tactile_buf = online_tactile_buffer(
        tactile_sensors=("left", "right"),
        tactile_output_type="tactile_rgb",
    )

    # One reset sample followed by a ten-step action chunk at the environment control rate.
    for step in range(11):
        tactile_buf.append_control_sample(
            _observation(step),
            env=_environment(step),
            include_tactile=True,
        )

    expected_steps = np.arange(3, 11, dtype=np.float32)
    force_history = tactile_buf.get_force_history()
    marker_history = tactile_buf.get_marker_motion()

    assert force_history is not None
    assert force_history.shape == (8, 6)
    assert marker_history is not None
    assert marker_history.shape == (9, 4, 2)
    np.testing.assert_array_equal(force_history[:, 0], expected_steps)
    np.testing.assert_array_equal(marker_history[1:, 0, 0], expected_steps)
    np.testing.assert_array_equal(
        [frame[0, 0, 0] for frame in tactile_buf._left_frames],
        expected_steps.astype(np.uint8),
    )
    np.testing.assert_array_equal(
        [frame[0, 0, 0] for frame in tactile_buf._right_frames],
        expected_steps.astype(np.uint8),
    )


def test_client_samples_after_reset_and_each_environment_step() -> None:
    tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
    run_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_closed_loop_policy"
    )
    statement_lists = list(_statement_lists(run_function))

    assert any(
        _is_tactile_buffer_call(statements[index], "reset")
        and _is_tactile_buffer_call(statements[index + 1], "append_control_sample")
        for statements in statement_lists
        for index in range(len(statements) - 1)
    )
    assert any(
        _is_env_step_assignment(statements[index])
        and _is_tactile_buffer_call(statements[index + 1], "append_control_sample")
        for statements in statement_lists
        for index in range(len(statements) - 1)
    )

    direct_updates = {
        node.func.attr
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tactile_buf"
        and node.func.attr
        in {"update_force", "update_marker_motion", "update_tactile_frames"}
    }
    assert direct_updates == set()
