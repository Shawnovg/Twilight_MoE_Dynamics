"""Keep even temporary test datasets on the configured data drive."""
import os
import uuid
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        root = Path(os.environ.get('TWILIGHT_DATA_DIR', 'E:/HarunoKen/Twilight_MoE_Dynamics'))
        base = root / 'test_runs' / uuid.uuid4().hex
        base.parent.mkdir(parents=True, exist_ok=True)
        # A unique, previously nonexistent directory avoids touching previous data.
        assert not base.exists()
        config.option.basetemp = str(base)
