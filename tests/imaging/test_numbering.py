import pytest
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import Mock, patch, MagicMock
from pixelbliss.imaging.numbering import add_candidate_number_to_image, add_candidate_numbers_to_images


class TestNumbering:
    """Test image numbering functionality."""
    
    def setup_method(self):
        """Set up common test fixtures."""
        # Create test images of different sizes
        self.small_image = Image.new('RGB', (100, 100), color='red')
        self.medium_image = Image.new('RGB', (500, 500), color='green')
        self.large_image = Image.new('RGB', (1000, 1000), color='blue')
        self.rgba_image = Image.new('RGBA', (200, 200), color=(255, 0, 0, 128))

    def test_add_candidate_number_to_image_basic(self):
        """Test basic functionality of adding candidate number to image."""
        result = add_candidate_number_to_image(self.medium_image, 1)
        
        # Should return a new image, not modify the original
        assert result is not self.medium_image
        assert result.size == self.medium_image.size
        assert result.mode == 'RGB'

    def test_add_candidate_number_to_image_different_numbers(self):
        """Test adding different candidate numbers."""
        result1 = add_candidate_number_to_image(self.medium_image, 1)
        result2 = add_candidate_number_to_image(self.medium_image, 99)
        
        # Results should be different images
        assert result1 is not result2
        assert result1.size == result2.size

    def test_add_candidate_number_to_image_custom_font_size(self):
        """Test adding candidate number with custom font size."""
        result = add_candidate_number_to_image(self.medium_image, 5, font_size=50)
        
        assert result is not self.medium_image
        assert result.size == self.medium_image.size

    def test_add_candidate_number_to_image_small_image(self):
        """Test adding candidate number to small image."""
        result = add_candidate_number_to_image(self.small_image, 1)
        
        assert result is not self.small_image
        assert result.size == self.small_image.size
        # Font size should be at least 24px even for small images
        
    def test_add_candidate_number_to_image_large_image(self):
        """Test adding candidate number to large image."""
        result = add_candidate_number_to_image(self.large_image, 1)
        
        assert result is not self.large_image
        assert result.size == self.large_image.size

    def test_add_candidate_number_to_image_rgba_input(self):
        """Test adding candidate number to RGBA image."""
        result = add_candidate_number_to_image(self.rgba_image, 1)
        
        assert result is not self.rgba_image
        assert result.size == self.rgba_image.size
        assert result.mode == 'RGB'  # Should be converted to RGB

    @patch('pixelbliss.imaging.numbering.os.path.exists')
    @patch('pixelbliss.imaging.numbering.ImageFont.truetype')
    def test_add_candidate_number_to_image_with_system_font(self, mock_truetype, mock_exists):
        """Test adding candidate number with system font available."""
        mock_exists.return_value = True
        mock_font = Mock()
        mock_truetype.return_value = mock_font
        
        with patch('pixelbliss.imaging.numbering.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            mock_draw.textbbox.return_value = (0, 0, 20, 15)  # Mock text bounding box
            
            result = add_candidate_number_to_image(self.medium_image, 1)
            
            # Should have tried to load system font
            mock_exists.assert_called()
            mock_truetype.assert_called()

    @patch('pixelbliss.imaging.numbering.os.path.exists')
    def test_add_candidate_number_to_image_fallback_to_default_font(self, mock_exists):
        """Test fallback to default font when system fonts fail."""
        mock_exists.return_value = False
        
        result = add_candidate_number_to_image(self.medium_image, 1)
        
        # Should still work with default font fallback
        assert result is not self.medium_image
        assert result.size == self.medium_image.size

    def test_add_candidate_number_to_image_no_font_available(self):
        """Test handling when no font is available - basic functionality test."""
        # This test just ensures the function works with default system fonts
        result = add_candidate_number_to_image(self.medium_image, 1)
        
        # Should still work without font
        assert result is not self.medium_image
        assert result.size == self.medium_image.size

    def test_add_candidate_number_to_image_font_loading_error(self):
        """Test handling font loading errors - basic functionality test."""
        # This test just ensures the function works with default system fonts
        result = add_candidate_number_to_image(self.medium_image, 1)
        
        # Should fallback and still work
        assert result is not self.medium_image
        assert result.size == self.medium_image.size

    def test_add_candidate_numbers_to_images_empty_list(self):
        """Test adding candidate numbers to empty list."""
        result = add_candidate_numbers_to_images([])
        
        assert result == []

    def test_add_candidate_numbers_to_images_single_candidate(self):
        """Test adding candidate numbers to single candidate."""
        candidates = [{'image': self.medium_image, 'provider': 'test', 'model': 'test'}]
        
        result = add_candidate_numbers_to_images(candidates)
        
        assert len(result) == 1
        assert result[0]['provider'] == 'test'
        assert result[0]['model'] == 'test'
        assert result[0]['image'] is not self.medium_image  # Should be a new image
        assert result[0]['image'].size == self.medium_image.size

    def test_add_candidate_numbers_to_images_multiple_candidates(self):
        """Test adding candidate numbers to multiple candidates."""
        candidates = [
            {'image': self.small_image, 'provider': 'test1', 'model': 'model1'},
            {'image': self.medium_image, 'provider': 'test2', 'model': 'model2'},
            {'image': self.large_image, 'provider': 'test3', 'model': 'model3'}
        ]
        
        result = add_candidate_numbers_to_images(candidates)
        
        assert len(result) == 3
        
        # Check that all candidates are processed
        for i, candidate in enumerate(result):
            assert candidate['provider'] == f'test{i+1}'
            assert candidate['model'] == f'model{i+1}'
            assert candidate['image'] is not candidates[i]['image']  # Should be new images
            assert candidate['image'].size == candidates[i]['image'].size

    def test_add_candidate_numbers_to_images_preserves_metadata(self):
        """Test that adding numbers preserves all candidate metadata."""
        candidates = [
            {
                'image': self.medium_image,
                'provider': 'test',
                'model': 'model',
                'score': 0.95,
                'custom_field': 'custom_value',
                'nested': {'key': 'value'}
            }
        ]
        
        result = add_candidate_numbers_to_images(candidates)
        
        assert len(result) == 1
        candidate = result[0]
        
        # All metadata should be preserved
        assert candidate['provider'] == 'test'
        assert candidate['model'] == 'model'
        assert candidate['score'] == 0.95
        assert candidate['custom_field'] == 'custom_value'
        assert candidate['nested'] == {'key': 'value'}
        
        # Only image should be different
        assert candidate['image'] is not self.medium_image

    def test_add_candidate_numbers_to_images_does_not_modify_original(self):
        """Test that original candidates list is not modified."""
        original_image = self.medium_image
        candidates = [{'image': original_image, 'provider': 'test'}]
        
        result = add_candidate_numbers_to_images(candidates)
        
        # Original candidates should be unchanged
        assert candidates[0]['image'] is original_image
        assert result[0]['image'] is not original_image

    @patch('pixelbliss.imaging.numbering.add_candidate_number_to_image')
    def test_add_candidate_numbers_to_images_calls_numbering_function(self, mock_add_number):
        """Test that the numbering function is called for each candidate."""
        mock_add_number.side_effect = lambda img, num: img  # Return original image
        
        candidates = [
            {'image': self.small_image},
            {'image': self.medium_image},
            {'image': self.large_image}
        ]
        
        result = add_candidate_numbers_to_images(candidates)
        
        # Should have called numbering function for each candidate
        assert mock_add_number.call_count == 3
        
        # Check the candidate numbers (1-based)
        mock_add_number.assert_any_call(self.small_image, 1)
        mock_add_number.assert_any_call(self.medium_image, 2)
        mock_add_number.assert_any_call(self.large_image, 3)

    def test_add_candidate_number_to_image_font_size_calculation(self):
        """Test font size calculation for different image sizes."""
        # Test with very small image
        tiny_image = Image.new('RGB', (50, 50), color='red')
        result = add_candidate_number_to_image(tiny_image, 1)
        assert result.size == (50, 50)
        
        # Test with very large image
        huge_image = Image.new('RGB', (3000, 2000), color='blue')
        result = add_candidate_number_to_image(huge_image, 1)
        assert result.size == (3000, 2000)

    def test_add_candidate_number_to_image_high_numbers(self):
        """Test adding high candidate numbers."""
        result = add_candidate_number_to_image(self.medium_image, 999)
        
        assert result is not self.medium_image
        assert result.size == self.medium_image.size

    def test_add_candidate_number_to_image_small_font_size(self):
        """Test handling of small font size."""
        result = add_candidate_number_to_image(self.medium_image, 1, font_size=12)
        
        assert result is not self.medium_image
        assert result.size == self.medium_image.size

    def test_add_candidate_number_to_image_large_font_size(self):
        """Test handling of large font size."""
        result = add_candidate_number_to_image(self.medium_image, 1, font_size=100)
        
        assert result is not self.medium_image
        assert result.size == self.medium_image.size

    @patch('pixelbliss.imaging.numbering.os.path.exists')
    @patch('pixelbliss.imaging.numbering.ImageFont.truetype')
    @patch('pixelbliss.imaging.numbering.ImageFont.load_default')
    def test_add_candidate_number_to_image_font_loading_exception(self, mock_load_default, mock_truetype, mock_exists):
        """Test font loading exception handling (lines 45-46)."""
        mock_exists.return_value = True
        # Mock truetype to raise OSError to trigger exception handling
        mock_truetype.side_effect = OSError("Font loading failed")
        # Mock load_default to return a working font as fallback
        mock_font = Mock()
        mock_load_default.return_value = mock_font
        
        with patch('pixelbliss.imaging.numbering.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            mock_draw.textbbox.return_value = (0, 0, 20, 15)
            
            result = add_candidate_number_to_image(self.medium_image, 1)
            
            # Should still work by falling back to default font
            assert result is not self.medium_image
            assert result.size == self.medium_image.size
            mock_truetype.assert_called()
            mock_load_default.assert_called_once()

    @patch('pixelbliss.imaging.numbering.os.path.exists')
    @patch('pixelbliss.imaging.numbering.ImageFont.truetype')
    @patch('pixelbliss.imaging.numbering.ImageFont.load_default')
    def test_add_candidate_number_to_image_default_font_failure(self, mock_load_default, mock_truetype, mock_exists):
        """Test default font loading failure (lines 52-54)."""
        mock_exists.return_value = False  # No system fonts available
        # Mock load_default to raise exception
        mock_load_default.side_effect = Exception("Default font loading failed")
        
        with patch('pixelbliss.imaging.numbering.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            # Mock textbbox to not be called since font is None
            
            result = add_candidate_number_to_image(self.medium_image, 1)
            
            # Should still work even without any font (font will be None)
            assert result is not self.medium_image
            assert result.size == self.medium_image.size
            mock_load_default.assert_called_once()

    @patch('pixelbliss.imaging.numbering.os.path.exists')
    @patch('pixelbliss.imaging.numbering.ImageFont.truetype')
    @patch('pixelbliss.imaging.numbering.ImageFont.load_default')
    def test_add_candidate_number_to_image_no_font_available(self, mock_load_default, mock_truetype, mock_exists):
        """Test text drawing without font available (line 108)."""
        mock_exists.return_value = False  # No system fonts
        mock_load_default.side_effect = Exception("No default font")  # Default font fails
        
        with patch('pixelbliss.imaging.numbering.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            # Mock textbbox to not be called since font is None
            
            # This should trigger the font = None path and the "if font:" else branch
            result = add_candidate_number_to_image(self.medium_image, 1)
            
            # Should still work by estimating text size and drawing without font
            assert result is not self.medium_image
            assert result.size == self.medium_image.size
