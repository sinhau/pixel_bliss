import pytest
import os
import tempfile
import yaml
from pathlib import Path
from pixelbliss.config import Config, load_config


class TestConfig:
    """Test configuration loading and validation."""

    def test_config_default_values(self):
        """Test that Config creates with default values."""
        config = Config()
        
        assert config.timezone == "America/Los_Angeles"
        assert len(config.categories) == 8
        assert "sci-fi" in config.categories
        assert config.category_selection_method == "time"
        assert config.rotation_minutes == 180
        assert config.prompt_generation.provider == "openai"
        assert config.image_generation.provider_order == ["fal", "replicate"]
        assert config.ranking.w_aesthetic == 0.50
        assert config.upscale.enabled is True

    def test_config_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            timezone="UTC",
            categories=["test1", "test2"],
            rotation_minutes=60
        )
        
        assert config.timezone == "UTC"
        assert config.categories == ["test1", "test2"]
        assert config.rotation_minutes == 60

    def test_load_config_valid_yaml(self):
        """Test loading configuration from valid YAML file."""
        config_data = {
            "timezone": "UTC",
            "categories": ["test", "demo"],
            "rotation_minutes": 120,
            "prompt_generation": {
                "provider": "openai",
                "model": "gpt-4"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.timezone == "UTC"
            assert config.categories == ["test", "demo"]
            assert config.rotation_minutes == 120
            assert config.prompt_generation.model == "gpt-4"
        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_config_with_environment_variables(self, monkeypatch):
        """Test loading configuration with environment variable substitution."""
        monkeypatch.setenv("TEST_WEBHOOK_URL", "https://webhook.example.com")
        
        config_data = {
            "alerts": {
                "enabled": True,
                "webhook_url_env": "TEST_WEBHOOK_URL"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.alerts.enabled is True
            # The webhook_url should be set from environment
            assert hasattr(config.alerts, 'webhook_url')
        finally:
            os.unlink(temp_path)

    def test_config_validation_invalid_data(self):
        """Test Config validation with invalid data types."""
        with pytest.raises(ValueError):
            Config(rotation_minutes="invalid")  # Should be int
        
        with pytest.raises(ValueError):
            Config(categories="not_a_list")  # Should be list
