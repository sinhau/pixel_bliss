"""Tests for the knobs functionality in the prompts module."""

import pytest
from unittest.mock import Mock, patch
from pixelbliss.prompts import make_base_with_knobs, make_variants_with_knobs
from pixelbliss.config import Config, PromptGeneration
from pixelbliss.prompt_engine.dummy_local import DummyLocalProvider


class TestPromptsKnobs:
    """Test knobs functionality in the prompts module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(
            prompt_generation=PromptGeneration(
                provider="dummy"
            )
        )
    
    def test_make_base_with_knobs_enabled(self):
        """Test base prompt generation with knobs enabled."""
        with patch('pixelbliss.prompts.KnobSelector.select_base_knobs') as mock_select, \
             patch('pixelbliss.prompts.KnobSelector.get_avoid_list') as mock_avoid:
            
            mock_select.return_value = {
                "vibe": "serene",
                "palette": "monochrome powder blue",
                "light": "high-key airy diffused",
                "texture": "silky",
                "composition": "negative space emphasis",
                "style": "watercolor wash"
            }
            mock_avoid.return_value = ["harsh clipping", "noise"]
            
            result = make_base_with_knobs("nature", self.config)
            
            assert isinstance(result, str)
            assert len(result) > 0
            mock_select.assert_called_once_with("nature")
            mock_avoid.assert_called_once()
    
    def test_make_base_with_knobs_without_progress_logger(self):
        """Test base prompt generation with knobs without progress logger."""
        with patch('pixelbliss.prompts.KnobSelector.select_base_knobs') as mock_select, \
             patch('pixelbliss.prompts.KnobSelector.get_avoid_list') as mock_avoid, \
             patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            
            mock_select.return_value = {
                "vibe": "serene",
                "palette": "monochrome powder blue",
                "light": "high-key airy diffused",
                "texture": "silky",
                "composition": "negative space emphasis",
                "style": "watercolor wash"
            }
            mock_avoid.return_value = ["harsh clipping", "noise"]
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = make_base_with_knobs("nature", self.config)
            
            assert isinstance(result, str)
            assert len(result) > 0
            mock_select.assert_called_once_with("nature")
            mock_avoid.assert_called_once()
            mock_logger.info.assert_called()
    
    def test_make_variants_with_knobs_multiple_strategy_default(self):
        """Test variant generation with multiple knobs strategy (now default)."""
        base_prompt = "A beautiful landscape"
        k = 3
        
        with patch('pixelbliss.prompts.KnobSelector.select_variant_knobs') as mock_select, \
             patch('pixelbliss.prompts.KnobSelector.get_avoid_list') as mock_avoid:
            
            mock_select.side_effect = [
                {"tone_curve": "high-key airy matte", "color_grade": "neutral balanced", "surface_fx": "crystal clean"},
                {"tone_curve": "mid-key balanced soft S-curve", "color_grade": "warm gold + muted", "surface_fx": "crystal clean"},
                {"tone_curve": "mid-key balanced soft S-curve", "color_grade": "neutral balanced", "surface_fx": "pearl glow"}
            ]
            mock_avoid.return_value = ["watermarks"]
            
            result = make_variants_with_knobs(base_prompt, k, self.config)
            
            assert isinstance(result, list)
            assert len(result) == k
            assert mock_select.call_count == k
            mock_avoid.assert_called_once()
    
    def test_make_variants_with_knobs_multiple_strategy(self):
        """Test variant generation with multiple knobs strategy."""
        base_prompt = "A beautiful landscape"
        k = 2
        
        with patch('pixelbliss.prompts.KnobSelector.select_variant_knobs') as mock_select, \
             patch('pixelbliss.prompts.KnobSelector.get_avoid_list') as mock_avoid:
            
            mock_select.side_effect = [
                {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
                {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
            ]
            mock_avoid.return_value = ["noise"]
            
            result = make_variants_with_knobs(base_prompt, k, self.config)
            
            assert isinstance(result, list)
            assert len(result) == k
            assert mock_select.call_count == k
            mock_avoid.assert_called_once()
    
    def test_make_variants_with_knobs_without_progress_logger(self):
        """Test variant generation with knobs without progress logger."""
        base_prompt = "A beautiful landscape"
        k = 2
        
        with patch('pixelbliss.prompts.KnobSelector.select_variant_knobs') as mock_select, \
             patch('pixelbliss.prompts.KnobSelector.get_avoid_list') as mock_avoid, \
             patch('pixelbliss.prompts.get_logger') as mock_get_logger:
            
            mock_select.side_effect = [
                {"tone_curve": "high-key airy matte", "color_grade": "neutral balanced", "surface_fx": "crystal clean"},
                {"tone_curve": "mid-key balanced soft S-curve", "color_grade": "warm gold + muted", "surface_fx": "crystal clean"}
            ]
            mock_avoid.return_value = ["watermarks"]
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = make_variants_with_knobs(base_prompt, k, self.config)
            
            assert isinstance(result, list)
            assert len(result) == k
            assert mock_select.call_count == k
            mock_avoid.assert_called_once()
            mock_logger.info.assert_called()
    
    def test_logging_with_knobs(self):
        """Test that logging works correctly with knobs system."""
        mock_logger = Mock()
        
        with patch('pixelbliss.prompts.get_logger') as mock_get_logger, \
             patch('pixelbliss.prompts.KnobSelector.select_base_knobs') as mock_select, \
             patch('pixelbliss.prompts.KnobSelector.get_avoid_list') as mock_avoid:
            
            mock_get_logger.return_value = mock_logger
            mock_select.return_value = {"vibe": "serene"}
            mock_avoid.return_value = ["noise"]
            
            result = make_base_with_knobs("nature", self.config)
            
            # Should log successful generation
            mock_logger.info.assert_called()
            mock_logger.debug.assert_called()
            
            # Check that knobs are logged
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("knobs" in call.lower() for call in debug_calls)


class TestPromptsKnobsIntegration:
    """Integration tests for knobs functionality."""
    
    def test_end_to_end_knobs_workflow(self):
        """Test complete workflow with knobs system."""
        config = Config(
            prompt_generation=PromptGeneration(
                provider="dummy"
            )
        )
        
        # Generate base prompt with knobs
        base_prompt = make_base_with_knobs("nature", config)
        assert isinstance(base_prompt, str)
        assert len(base_prompt) > 0
        
        # Generate variants with knobs
        variants = make_variants_with_knobs(base_prompt, 3, config)
        assert isinstance(variants, list)
        assert len(variants) == 3
        assert all(isinstance(v, str) for v in variants)
        assert all(len(v) > 0 for v in variants)
    
    def test_knobs_produce_different_results(self):
        """Test that knobs system produces varied results."""
        config = Config(
            prompt_generation=PromptGeneration(
                provider="dummy"
            )
        )
        
        # Generate multiple base prompts
        base_prompts = [make_base_with_knobs("nature", config) for _ in range(5)]
        
        # Should have some variation (dummy provider includes knob values in output)
        unique_prompts = set(base_prompts)
        assert len(unique_prompts) > 1  # Should have some diversity
        
        # Generate multiple variant sets
        base_prompt = "A beautiful landscape"
        variant_sets = [make_variants_with_knobs(base_prompt, 2, config) for _ in range(3)]
        
        # Should have some variation
        flattened_variants = [v for variant_set in variant_sets for v in variant_set]
        unique_variants = set(flattened_variants)
        assert len(unique_variants) > 2  # Should have some diversity
