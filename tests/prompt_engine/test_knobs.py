"""Tests for the knobs system."""

import pytest
from pixelbliss.prompt_engine.knobs import (
    KnobSelector, BASE_KNOBS, VARIANT_KNOBS, AVOID
)


class TestKnobSelector:
    """Test the KnobSelector utility class."""
    
    def test_select_base_knobs(self):
        """Test base knobs selection."""
        knobs = KnobSelector.select_base_knobs()
        
        # Should return all 6 base knob categories
        assert len(knobs) == 6
        expected_keys = {"vibe", "palette", "light", "texture", "composition", "style"}
        assert set(knobs.keys()) == expected_keys
        
        # Each value should be from the corresponding knob list
        for knob_name, knob_value in knobs.items():
            assert knob_value in BASE_KNOBS[knob_name]
    
    def test_select_base_knobs_with_category(self):
        """Test base knobs selection with category hint."""
        knobs = KnobSelector.select_base_knobs("nature")
        
        # Should still return all 6 base knob categories
        assert len(knobs) == 6
        expected_keys = {"vibe", "palette", "light", "texture", "composition", "style"}
        assert set(knobs.keys()) == expected_keys
    
    def test_select_variant_knobs(self):
        """Test variant knobs selection."""
        knobs = KnobSelector.select_variant_knobs()
        
        # Should return all 3 variant knob categories
        assert len(knobs) == 3
        expected_keys = {"tone_curve", "color_grade", "surface_fx"}
        assert set(knobs.keys()) == expected_keys
        
        # Each value should be from the corresponding knob list
        for knob_name, knob_value in knobs.items():
            assert knob_value in VARIANT_KNOBS[knob_name]
    
    
    def test_get_avoid_list(self):
        """Test avoid list retrieval."""
        avoid_list = KnobSelector.get_avoid_list()
        
        # Should return a copy of the AVOID list
        assert avoid_list == AVOID
        assert avoid_list is not AVOID  # Should be a copy
    
    def test_multiple_selections_are_different(self):
        """Test that multiple selections produce different results."""
        # Generate multiple base knob selections
        selections = [KnobSelector.select_base_knobs() for _ in range(10)]
        
        # At least some should be different (very high probability)
        unique_selections = set(tuple(sorted(s.items())) for s in selections)
        assert len(unique_selections) > 1
        
        # Generate multiple variant knob selections
        variant_selections = [KnobSelector.select_variant_knobs() for _ in range(10)]
        unique_variant_selections = set(tuple(sorted(s.items())) for s in variant_selections)
        assert len(unique_variant_selections) > 1


class TestKnobData:
    """Test the knob data structures."""
    
    def test_base_knobs_structure(self):
        """Test base knobs data structure."""
        assert len(BASE_KNOBS) == 6
        expected_keys = {"vibe", "palette", "light", "texture", "composition", "style"}
        assert set(BASE_KNOBS.keys()) == expected_keys
        
        # Each knob should have multiple options
        for knob_name, knob_values in BASE_KNOBS.items():
            assert isinstance(knob_values, list)
            assert len(knob_values) >= 10  # Should have good diversity
            # All values should be strings
            assert all(isinstance(v, str) for v in knob_values)
    
    def test_variant_knobs_structure(self):
        """Test variant knobs data structure."""
        assert len(VARIANT_KNOBS) == 3
        expected_keys = {"tone_curve", "color_grade", "surface_fx"}
        assert set(VARIANT_KNOBS.keys()) == expected_keys
        
        # Each knob should have multiple options
        for knob_name, knob_values in VARIANT_KNOBS.items():
            assert isinstance(knob_values, list)
            assert len(knob_values) >= 10  # Should have good diversity
            # All values should be strings
            assert all(isinstance(v, str) for v in knob_values)
    
    def test_avoid_list_structure(self):
        """Test avoid list structure."""
        assert isinstance(AVOID, list)
        assert len(AVOID) > 0
        # All values should be strings
        assert all(isinstance(v, str) for v in AVOID)
    
    def test_specific_knob_counts(self):
        """Test that knobs have the expected counts from the specification."""
        # From the specification
        assert len(BASE_KNOBS["vibe"]) == 18
        assert len(BASE_KNOBS["palette"]) == 16
        assert len(BASE_KNOBS["light"]) == 16
        assert len(BASE_KNOBS["texture"]) == 16
        assert len(BASE_KNOBS["composition"]) == 16
        assert len(BASE_KNOBS["style"]) == 16
        
        assert len(VARIANT_KNOBS["tone_curve"]) == 12
        assert len(VARIANT_KNOBS["color_grade"]) == 16
        assert len(VARIANT_KNOBS["surface_fx"]) == 16
        
        assert len(AVOID) == 11




class TestKnobValues:
    """Test specific knob values for quality."""
    
    def test_vibe_values_quality(self):
        """Test that vibe values are appropriate."""
        vibes = BASE_KNOBS["vibe"]
        
        # Should contain expected aesthetic terms
        expected_terms = ["serene", "gentle", "dreamlike", "zen", "tranquil"]
        for term in expected_terms:
            assert any(term in vibe for vibe in vibes)
    
    def test_palette_values_quality(self):
        """Test that palette values are appropriate."""
        palettes = BASE_KNOBS["palette"]
        
        # Should contain color-related terms
        color_terms = ["blue", "ivory", "slate", "mint", "lilac", "gold"]
        for term in color_terms:
            assert any(term in palette for palette in palettes)
    
    def test_avoid_list_quality(self):
        """Test that avoid list contains appropriate terms."""
        avoid_terms = AVOID
        
        # Should contain quality-related negative terms
        expected_terms = ["harsh", "noise", "watermarks", "clutter", "contrast"]
        for term in expected_terms:
            assert any(term in avoid_term for avoid_term in avoid_terms)
