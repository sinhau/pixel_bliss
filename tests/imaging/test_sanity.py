import pytest
from pixelbliss.imaging.sanity import passes_floors


class TestSanity:
    """Test image sanity checking functions."""

    def test_passes_floors_all_criteria_met(self, sample_config):
        """Test passes_floors when all criteria are met."""
        # Config has: entropy_min=3.5, brightness_min=10, brightness_max=245
        brightness = 100.0
        entropy = 5.0
        
        result = passes_floors(brightness, entropy, sample_config)
        
        assert result is True

    def test_passes_floors_entropy_too_low(self, sample_config):
        """Test passes_floors when entropy is below minimum."""
        brightness = 100.0
        entropy = 2.0  # Below entropy_min of 3.5
        
        result = passes_floors(brightness, entropy, sample_config)
        
        assert result is False

    def test_passes_floors_brightness_too_low(self, sample_config):
        """Test passes_floors when brightness is below minimum."""
        brightness = 5.0  # Below brightness_min of 10
        entropy = 5.0
        
        result = passes_floors(brightness, entropy, sample_config)
        
        assert result is False

    def test_passes_floors_brightness_too_high(self, sample_config):
        """Test passes_floors when brightness is above maximum."""
        brightness = 250.0  # Above brightness_max of 245
        entropy = 5.0
        
        result = passes_floors(brightness, entropy, sample_config)
        
        assert result is False

    def test_passes_floors_boundary_values(self, sample_config):
        """Test passes_floors with exact boundary values."""
        # Test minimum entropy boundary
        result = passes_floors(100.0, 3.5, sample_config)  # entropy_min = 3.5
        assert result is True
        
        # Test minimum brightness boundary
        result = passes_floors(10.0, 5.0, sample_config)  # brightness_min = 10
        assert result is True
        
        # Test maximum brightness boundary
        result = passes_floors(245.0, 5.0, sample_config)  # brightness_max = 245
        assert result is True

    def test_passes_floors_multiple_failures(self, sample_config):
        """Test passes_floors when multiple criteria fail."""
        brightness = 5.0   # Too low
        entropy = 2.0      # Too low
        
        result = passes_floors(brightness, entropy, sample_config)
        
        assert result is False
