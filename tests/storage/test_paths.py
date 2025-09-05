import pytest
from pixelbliss.storage.paths import make_slug, output_dir


class TestMakeSlug:
    """Test cases for make_slug function."""

    def test_make_slug_basic(self):
        """Test basic slug creation."""
        result = make_slug("nature", "beautiful sunset")
        
        assert result == "nature_beautiful_sunset"

    def test_make_slug_with_special_characters(self):
        """Test slug creation with special characters."""
        result = make_slug("art & design", "modern art!")
        
        assert result == "art_design_modern_art_"

    def test_make_slug_with_numbers(self):
        """Test slug creation with numbers."""
        result = make_slug("tech2024", "ai prompt 123")
        
        assert result == "tech2024_ai_prompt_123"

    def test_make_slug_with_hyphens_and_underscores(self):
        """Test slug creation preserving hyphens and underscores."""
        result = make_slug("sci-fi", "space_travel")
        
        assert result == "sci-fi_space_travel"

    def test_make_slug_collapse_multiple_underscores(self):
        """Test that multiple consecutive underscores are collapsed."""
        result = make_slug("test   category", "prompt   with   spaces")
        
        assert result == "test_category_prompt_with_spaces"

    def test_make_slug_empty_strings(self):
        """Test slug creation with empty strings."""
        result = make_slug("", "")
        
        assert result == "_"

    def test_make_slug_only_special_characters(self):
        """Test slug creation with only special characters."""
        result = make_slug("@#$%", "!@#$%^&*()")
        
        assert result == "_"

    def test_make_slug_length_limit_under_50(self):
        """Test slug creation with content under 50 characters."""
        category = "short"
        prompt = "brief"
        result = make_slug(category, prompt)
        
        assert result == "short_brief"
        assert len(result) <= 50

    def test_make_slug_length_limit_exactly_50(self):
        """Test slug creation with content exactly 50 characters."""
        # Create content that results in exactly 50 characters
        category = "category"  # 8 chars
        prompt = "a" * 41  # 41 chars, total with underscore = 50
        result = make_slug(category, prompt)
        
        assert len(result) == 50
        assert result == f"category_{'a' * 41}"

    def test_make_slug_length_limit_over_50(self):
        """Test slug creation with content over 50 characters."""
        category = "very_long_category_name"
        prompt = "this_is_a_very_long_prompt_that_exceeds_fifty_characters_total"
        result = make_slug(category, prompt)
        
        assert len(result) == 50
        assert result.startswith("very_long_category_name_this_is_a_very_long")

    def test_make_slug_unicode_characters(self):
        """Test slug creation with unicode characters."""
        result = make_slug("café", "naïve résumé")
        
        assert result == "café_naïve_résumé"

    def test_make_slug_mixed_case(self):
        """Test slug creation preserves case."""
        result = make_slug("Nature", "Beautiful SUNSET")
        
        assert result == "Nature_Beautiful_SUNSET"

    def test_make_slug_with_dots_and_commas(self):
        """Test slug creation with dots and commas."""
        result = make_slug("art.design", "modern, contemporary")
        
        assert result == "art_design_modern_contemporary"

    def test_make_slug_with_parentheses_and_brackets(self):
        """Test slug creation with parentheses and brackets."""
        result = make_slug("tech(2024)", "ai[prompt]")
        
        assert result == "tech_2024_ai_prompt_"

    def test_make_slug_leading_trailing_spaces(self):
        """Test slug creation with leading and trailing spaces."""
        result = make_slug("  category  ", "  prompt  ")
        
        assert result == "_category_prompt_"

    def test_make_slug_newlines_and_tabs(self):
        """Test slug creation with newlines and tabs."""
        result = make_slug("cat\negory", "pro\tmpt")
        
        assert result == "cat_egory_pro_mpt"


class TestOutputDir:
    """Test cases for output_dir function."""

    def test_output_dir_basic(self):
        """Test basic output directory generation."""
        result = output_dir("2024-01-15", "nature_sunset")
        
        assert result == "outputs/2024-01-15/nature_sunset"

    def test_output_dir_different_date_formats(self):
        """Test output directory with different date strings."""
        result1 = output_dir("2024-12-31", "test_slug")
        result2 = output_dir("2023-01-01", "another_slug")
        
        assert result1 == "outputs/2024-12-31/test_slug"
        assert result2 == "outputs/2023-01-01/another_slug"

    def test_output_dir_empty_slug(self):
        """Test output directory with empty slug."""
        result = output_dir("2024-01-15", "")
        
        assert result == "outputs/2024-01-15/"

    def test_output_dir_empty_date(self):
        """Test output directory with empty date."""
        result = output_dir("", "test_slug")
        
        assert result == "outputs//test_slug"

    def test_output_dir_both_empty(self):
        """Test output directory with both empty parameters."""
        result = output_dir("", "")
        
        assert result == "outputs//"

    def test_output_dir_special_characters_in_slug(self):
        """Test output directory with special characters in slug."""
        result = output_dir("2024-01-15", "test_slug_with_underscores")
        
        assert result == "outputs/2024-01-15/test_slug_with_underscores"

    def test_output_dir_long_slug(self):
        """Test output directory with long slug."""
        long_slug = "a" * 100
        result = output_dir("2024-01-15", long_slug)
        
        assert result == f"outputs/2024-01-15/{long_slug}"

    def test_output_dir_numeric_slug(self):
        """Test output directory with numeric slug."""
        result = output_dir("2024-01-15", "12345")
        
        assert result == "outputs/2024-01-15/12345"

    def test_output_dir_mixed_case(self):
        """Test output directory preserves case."""
        result = output_dir("2024-01-15", "Test_Slug_Mixed_Case")
        
        assert result == "outputs/2024-01-15/Test_Slug_Mixed_Case"

    def test_output_dir_with_hyphens(self):
        """Test output directory with hyphens in slug."""
        result = output_dir("2024-01-15", "test-slug-with-hyphens")
        
        assert result == "outputs/2024-01-15/test-slug-with-hyphens"


class TestIntegration:
    """Integration tests combining make_slug and output_dir."""

    def test_make_slug_to_output_dir_workflow(self):
        """Test the typical workflow from make_slug to output_dir."""
        category = "nature photography"
        prompt = "beautiful mountain landscape at sunset"
        date = "2024-01-15"
        
        slug = make_slug(category, prompt)
        output_path = output_dir(date, slug)
        
        expected_slug = "nature_photography_beautiful_mountain_landscape_at"
        expected_path = f"outputs/2024-01-15/{expected_slug}"
        
        assert slug == expected_slug
        assert output_path == expected_path

    def test_long_content_workflow(self):
        """Test workflow with content that exceeds slug length limit."""
        category = "very long category name that exceeds normal limits"
        prompt = "extremely long prompt text that definitely exceeds the fifty character limit for slugs"
        date = "2024-12-31"
        
        slug = make_slug(category, prompt)
        output_path = output_dir(date, slug)
        
        assert len(slug) == 50
        assert output_path.startswith("outputs/2024-12-31/")
        assert len(output_path.split("/")[-1]) == 50

    def test_special_characters_workflow(self):
        """Test workflow with special characters throughout."""
        category = "art & design!"
        prompt = "modern @rt with #hashtags"
        date = "2024-06-15"
        
        slug = make_slug(category, prompt)
        output_path = output_dir(date, slug)
        
        expected_slug = "art_design_modern_rt_with_hashtags"
        expected_path = f"outputs/2024-06-15/{expected_slug}"
        
        assert slug == expected_slug
        assert output_path == expected_path
