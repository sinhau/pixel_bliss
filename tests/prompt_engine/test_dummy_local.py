import pytest
from pixelbliss.prompt_engine.dummy_local import DummyLocalProvider


class TestDummyLocalProvider:
    def test_make_base(self):
        provider = DummyLocalProvider()
        result = provider.make_base("sunset")
        expected = "A beautiful sunset landscape with vibrant colors, high detail, sharp focus, no blur, no text."
        assert result == expected

    def test_make_variants_from_base_without_styles(self):
        provider = DummyLocalProvider()
        base_prompt = "A beautiful landscape"
        variants = provider.make_variants_from_base(base_prompt, 3)
        assert len(variants) == 3
        assert all("Variation" in v for v in variants)
        assert all("with added artistic elements, enhanced lighting" in v for v in variants)

    def test_make_variants_from_base_with_styles(self):
        provider = DummyLocalProvider()
        base_prompt = "A beautiful landscape"
        art_styles = ["impressionist", "abstract"]
        variants = provider.make_variants_from_base(base_prompt, 2, art_styles)
        assert len(variants) == 2
        assert "impressionist style" in variants[0]
        assert "abstract style" in variants[1]

    def test_make_alt_text(self):
        provider = DummyLocalProvider()
        result = provider.make_alt_text("base", "variant")
        expected = "A stunning aesthetic image featuring vibrant colors and intricate details."
        assert result == expected

    def test_make_base_with_knobs(self):
        provider = DummyLocalProvider()
        base_knobs = {
            "vibe": "serene",
            "palette": "monochrome powder blue",
            "light": "high-key airy diffused",
            "texture": "silky",
            "composition": "negative space emphasis",
            "style": "watercolor wash"
        }
        avoid_list = ["harsh clipping", "noise"]
        
        result = provider.make_base_with_knobs(base_knobs, avoid_list)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "vibe: serene" in result
        assert "avoiding harsh clipping, noise" in result
        assert "no blur, no text" in result

    def test_make_variants_with_knobs(self):
        provider = DummyLocalProvider()
        base_prompt = "A beautiful landscape"
        variant_knobs_list = [
            {"tone_curve": "high-key airy matte", "color_grade": "warm gold + muted", "surface_fx": "soft matte"},
            {"tone_curve": "low-key velvet deep", "color_grade": "cool silver + desaturated", "surface_fx": "pearl glow"}
        ]
        avoid_list = ["watermarks"]
        
        result = provider.make_variants_with_knobs(base_prompt, 2, variant_knobs_list, avoid_list)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        assert "Variation 1" in result[0]
        assert "tone_curve: high-key airy matte" in result[0]
        assert "avoiding watermarks" in result[0]
        
        assert "Variation 2" in result[1]
        assert "tone_curve: low-key velvet deep" in result[1]
        assert "avoiding watermarks" in result[1]
