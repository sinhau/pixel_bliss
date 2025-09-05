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
        assert config.prompt_generation.provider == "openai"
        assert config.prompt_generation.use_knobs is True
        assert config.prompt_generation.variant_strategy == "single"
        assert config.image_generation.provider_order == ["fal", "replicate"]
        assert config.ranking.w_aesthetic == 0.50
        assert config.upscale.enabled is True

    def test_config_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            timezone="UTC",
            prompt_generation={
                "provider": "dummy",
                "use_knobs": False,
                "variant_strategy": "multiple"
            }
        )
        
        assert config.timezone == "UTC"
        assert config.prompt_generation.provider == "dummy"
        assert config.prompt_generation.use_knobs is False
        assert config.prompt_generation.variant_strategy == "multiple"

    def test_load_config_valid_yaml(self):
        """Test loading configuration from valid YAML file."""
        config_data = {
            "timezone": "UTC",
            "prompt_generation": {
                "provider": "openai",
                "model": "gpt-4",
                "use_knobs": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.timezone == "UTC"
            assert config.prompt_generation.model == "gpt-4"
            assert config.prompt_generation.use_knobs is True
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
            # The webhook_url_env should be set
            assert config.alerts.webhook_url_env == "TEST_WEBHOOK_URL"
        finally:
            os.unlink(temp_path)

    def test_config_validation_invalid_data(self):
        """Test Config validation with invalid data types."""
        with pytest.raises(ValueError):
            Config(timezone=123)  # Should be string
        
        with pytest.raises(ValueError):
            Config(wallpaper_variants="not_a_list")  # Should be list
