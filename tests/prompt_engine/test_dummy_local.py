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
