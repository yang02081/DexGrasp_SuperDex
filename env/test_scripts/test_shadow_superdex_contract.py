"""Static contract checks for the Shadow SuperDex cross-hand backend."""
from pathlib import Path
import ast


def _constants(path):
    tree = ast.parse(path.read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def test_shadow_crosshand_dimensions_match_registry():
    root = Path(__file__).resolve().parents[2]
    backend = root / "env/superdex/shadow_crosshand_superdex_env.py"
    values = _constants(backend)
    assert values["ACTION_DIM"] == 28
    assert values["BASE_OBSERVATION_DIM"] == 136
    assert values["OBSERVATION_DIM"] == 268
    assert values["GLOBAL_STATE_DIM"] == 179
    assert len(values["FRAME_NAMES"]) == 22
    assert len(values["CONTACT_LINK_NAMES"]) == 17


def test_official_assets_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "model/multi_hand/full_2700_r.pt").is_file()
    assert (root / "model/multi_hand/mean_r2700.csv").is_file()
    assert (root / "model/multi_hand/var_r2700.csv").is_file()
    assert (root / "rsc/shadow/shadowhand_large.urdf").is_file()





def test_object_state_round_trip():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from env.superdex.shadow_crosshand_superdex_env import (
        _raisim_object_to_superdex, _superdex_object_to_raisim,
    )

    origin = np.array([0.65, 0.0, 0.35], np.float32)
    quat_xyzw = Rotation.from_euler("XYZ", [0.2, -0.3, 0.4]).as_quat()
    source = np.array([1.0, -0.1, 0.55, *quat_xyzw[[3, 0, 1, 2]], 0.17], np.float32)
    restored = _superdex_object_to_raisim(
        _raisim_object_to_superdex(source, origin), origin
    )
    assert np.allclose(restored[:3], source[:3], atol=1e-6)
    assert np.allclose(restored[7], source[7], atol=1e-6)
    assert abs(float(np.dot(restored[3:7], source[3:7]))) > 1.0 - 1e-6


def test_relative_root_mapping_round_trip():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from env.superdex.shadow_crosshand_superdex_env import ShadowCrossHandSuperDexEnv

    env = ShadowCrossHandSuperDexEnv.__new__(ShadowCrossHandSuperDexEnv)
    env._scene_origin = np.array([0.65, 0.0, 0.35], np.float32)
    env._initial_root_position_world = np.array([1.0, -0.2, 0.55], np.float32)
    env._initial_root_rotation = Rotation.from_euler("XYZ", [0.4, -0.3, 0.2])
    target = np.linspace(-0.2, 0.2, 28, dtype=np.float32)
    mapped = env._raisim_target_to_superdex(target)
    world_translation = mapped[:3] + env._scene_origin
    restored_translation = env._initial_root_rotation.inv().apply(
        world_translation - env._initial_root_position_world
    )
    restored_rotation = (
        env._initial_root_rotation.inv() * Rotation.from_euler("XYZ", mapped[3:6])
    ).as_euler("XYZ")
    assert np.allclose(restored_translation, target[:3], atol=1e-6)
    assert np.allclose(restored_rotation, target[3:6], atol=1e-6)
    assert np.allclose(mapped[6:], target[6:], atol=1e-7)


def test_affordance_contact_takes_precedence():
    import numpy as np

    from env.superdex.shadow_crosshand_superdex_env import _raisim_contact_channels, DT

    top = np.array([2.0, 0.0, 3.0], np.float32)
    bottom = np.array([1.0, 4.0, 5.0], np.float32)
    ct, it, cb, ib = _raisim_contact_channels(top, bottom)
    assert np.array_equal(ct, [1.0, 0.0, 1.0])
    assert np.array_equal(cb, [0.0, 1.0, 0.0])
    assert np.allclose(it, top * DT)
    assert np.allclose(ib, np.array([-1.0, 4.0, 2.0], np.float32) * DT)


if __name__ == "__main__":
    test_shadow_crosshand_dimensions_match_registry()
    test_official_assets_exist()
    test_object_state_round_trip()
    test_relative_root_mapping_round_trip()
    test_affordance_contact_takes_precedence()
    print("Shadow SuperDex contract: PASS")
