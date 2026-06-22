import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools import run_task_evaluations as rte


def test_libero_path_under_tabero_checkout_does_not_auto_enable_tabero_subset():
    hdf5_folder = Path("/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5")

    assert not rte._should_auto_use_tabero_tasks(hdf5_folder, "")


def test_tabero_dataset_path_auto_enables_tabero_subset():
    hdf5_folder = Path("/datasets/tabero_force/replayed_demos")

    assert rte._should_auto_use_tabero_tasks(hdf5_folder, "")


def test_openpi_camera_names_cli_accepts_multiple_values():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()
    module = ast.parse(source)
    annotation = None

    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "OpenpiClientArguments":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == "camera_names":
                        annotation = ast.unparse(item.annotation)

    assert annotation == "tuple[str, ...]"


def test_openpi_osc_actions_are_sent_as_7d_actions():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()

    osc_branch_start = source.find('elif args.control_mode == "osc":')
    assert osc_branch_start != -1

    next_branch_start = source.find('elif args.control_mode == "binary":', osc_branch_start)
    assert next_branch_start != -1

    osc_branch = source[osc_branch_start:next_branch_start]
    assert "action_chunk[:, :7]" in osc_branch
    assert "axisangle2quat" not in osc_branch
