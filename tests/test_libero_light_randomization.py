import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def _module(path: str) -> ast.Module:
    return ast.parse(_source(path))


def _class_node(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _dataclass_has_field(path: str, class_name: str, field_name: str) -> bool:
    cls = _class_node(_module(path), class_name)
    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.target.id == field_name:
                return True
    return False


def test_libero_mdp_exports_local_events():
    source = _source("source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/__init__.py")

    assert "from .events import *" in source


def test_domelight_randomizer_has_single_definition_and_randomizes_color():
    module = _module("source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/events.py")
    funcs = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "randomize_domelight_color_intensity"
    ]

    assert len(funcs) == 1
    call_names = {
        node.func.id
        for node in ast.walk(funcs[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "sample_random_color" in call_names
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Attribute)
        and node.left.func.attr == "GetTypeName"
        and any(isinstance(comp, ast.Constant) and comp.value == "DomeLight" for comp in node.comparators)
        for node in ast.walk(funcs[0])
    )


def test_franka_libero_light_event_is_env_flag_gated():
    module = _module(
        "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py"
    )
    event_cls = _class_node(module, "EventCfgFrankaPanda")

    class_level_names = {
        target.id
        for item in event_cls.body
        if isinstance(item, ast.Assign)
        for target in item.targets
        if isinstance(target, ast.Name)
    }
    source = ast.unparse(event_cls)

    assert "randomize_light" not in class_level_names
    assert "LIBERO_RANDOMIZE_LIGHT" in source
    assert "self.randomize_light = EventTerm" in source
    assert "randomize_domelight_color_intensity" in source
    assert "NVIDIA_NUCLEUS_DIR" in _source(
        "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py"
    )


def test_tactile_cfg_does_not_duplicate_light_event():
    source = _source(
        "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_tactile_libero_env_cfg.py"
    )

    assert "randomize_light" not in source
    assert "EventTerm" not in source
    assert "JointPositionLiberoCameraEnvCfg" in source


def test_cli_entrypoints_expose_randomize_light_flag():
    argparse_paths = [
        "scripts/tools/replay_demos.py",
        "scripts/tools/replay_demos_with_camera.py",
    ]
    for path in argparse_paths:
        source = _source(path)
        assert '"--randomize_light"' in source
        assert "LIBERO_RANDOMIZE_LIGHT" in source

    assert _dataclass_has_field(
        "benchmarks/openpi/openpi_inference_client.py", "OpenpiClientArguments", "randomize_light"
    )
    assert "LIBERO_RANDOMIZE_LIGHT" in _source("benchmarks/openpi/openpi_inference_client.py")


def test_batch_scripts_pass_randomize_light_to_children():
    run_task_source = _source("scripts/tools/run_task_evaluations.py")
    run_data_source = _source("scripts/tools/run_data_evaluations.py")

    assert _dataclass_has_field("scripts/tools/run_task_evaluations.py", "EvaluationConfig", "randomize_light")
    assert 'cmd.append("--randomize_light")' in run_task_source

    assert _dataclass_has_field("scripts/tools/run_data_evaluations.py", "DataReplayEvalConfig", "randomize_light")
    assert 'cmd.append("--randomize_light")' in run_data_source
