import importlib.util
from pathlib import Path

from launch_ros.actions import Node

LAUNCH_FILE = Path(__file__).resolve().parents[1] / 'launch' / 'guidance.launch.py'


def test_launch_description_starts_all_guidance_nodes() -> None:
    spec = importlib.util.spec_from_file_location('guidance_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ld = module.generate_launch_description()
    nodes = [entity for entity in ld.entities if isinstance(entity, Node)]

    assert len(nodes) == len(module.GUIDANCE_NODES)
