"""
Baseline smoke tests for project configuration.
"""

from src.utils import config


def test_config_paths():
    """Verify project paths and configuration are properly loaded."""
    assert config.PROJECT_ROOT.exists()
    assert config.DATA_DIR.exists()
    assert config.RAW_DATA_PATH.exists() or config.RAW_DATA_PATH.parent.exists()
    assert isinstance(config.LLM_MODEL_NAME, str)
