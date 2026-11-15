"""Unknown config fields fail loudly in Python and YAML."""
import copy
import pickle
from dataclasses import asdict, replace

import pytest

import polarism as ps


@pytest.mark.parametrize("section,name,suggestion", [
    ("physics", "g_c", "g_C"),
    ("solver", "adpative", "adaptive"),
    ("grid", "NX", "nx"),
])
def test_unknown_section_field_raises_with_suggestion(section, name, suggestion):
    cfg = ps.Config()
    with pytest.raises(AttributeError, match=rf"Unknown config key '{section}\.{name}'; did you mean '{suggestion}'"):
        setattr(getattr(cfg, section), name, 1.0)


def test_unknown_root_field_raises_with_suggestion():
    cfg = ps.Config()
    with pytest.raises(AttributeError, match=r"Unknown config key 'phyiscs'; did you mean 'physics'"):
        cfg.phyiscs = None


def test_unknown_private_field_raises():
    cfg = ps.Config()
    with pytest.raises(AttributeError, match="Unknown config key '_anything'"):
        cfg._anything = 1


def test_known_fields_copy_and_serialize():
    cfg = ps.Config()
    cfg.physics.g_C = 0.5
    cfg.solver = replace(cfg.solver, dt=0.2)
    assert copy.deepcopy(cfg).physics.g_C == 0.5
    assert pickle.loads(pickle.dumps(cfg)).solver.dt == 0.2
    assert asdict(cfg)["physics"]["g_C"] == 0.5


def test_yaml_unknown_key_suggests(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("physics:\n  g_c: 0.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unknown config key 'physics\.g_c'; did you mean 'g_C'"):
        ps.Config.from_yaml(path)


def test_entry_overrides_yaml_round_trip(tmp_path):
    cfg = ps.Config()
    cfg.laser.mode = "multiple"
    cfg.laser.entry_overrides = {1: {"delay": 0.2}}
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    restored = ps.Config.from_yaml(path)
    assert restored.laser.entry_overrides == {1: {"delay": 0.2}}
