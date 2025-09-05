"""
Tests for image compression utilities.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from pixelbliss.imaging.compression import (
    get_file_size_mb,
    get_image_size_bytes,
    resize_image_proportionally,
    compress_image_smart,
    compress_image_file,
    prepare_for_twitter_upload,
    TWITTER_MAX_SIZE_BYTES,
    TWITTER_MAX_SIZE_MB
)


class TestGetFileSizeMb:
    """Test file size calculation."""
    
    def test_existing_file(self):
        """Test getting size of existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write 1KB of data
            tmp.write(b'x' * 1024)
            tmp.flush()
            
            size_mb = get_file_size_mb(tmp.name)
            assert abs(size_mb - 0.001) < 0.0001  # ~1KB = ~0.001MB
            
            os.unlink(tmp.name)
    
    def test_nonexistent_file(self):
        """Test getting size of non-existent file."""
        size_mb = get_file_size_mb("/nonexistent/file.jpg")
        assert size_mb == 0.0


class TestGetImageSizeBytes:
    """Test image size calculation in memory."""
    
    def test_png_format(self):
        """Test PNG format size calculation."""
        # Create a small test image
        image = Image.new('RGB', (100, 100), color='red')
        size_bytes = get_image_size_bytes(image, 'PNG')
        
        assert size_bytes > 0
        assert isinstance(size_bytes, int)
    
    def test_jpeg_format(self):
        """Test JPEG format size calculation."""
        image = Image.new('RGB', (100, 100), color='red')
        size_bytes = get_image_size_bytes(image, 'JPEG', quality=90)
        
        assert size_bytes > 0
        assert isinstance(size_bytes, int)
    
    def test_jpeg_quality_affects_size(self):
        """Test that JPEG quality affects file size."""
        image = Image.new('RGB', (200, 200), color='red')
        
        high_quality_size = get_image_size_bytes(image, 'JPEG', quality=95)
        low_quality_size = get_image_size_bytes(image, 'JPEG', quality=50)
        
        # Higher quality should generally result in larger file size
        assert high_quality_size >= low_quality_size


class TestResizeImageProportionally:
    """Test proportional image resizing."""
    
    def test_no_resize_needed(self):
        """Test image that doesn't need resizing."""
        image = Image.new('RGB', (500, 300), color='blue')
        resized = resize_image_proportionally(image, 1000)
        
        assert resized.size == (500, 300)
    
    def test_resize_width_larger(self):
        """Test resizing when width is larger dimension."""
        image = Image.new('RGB', (1000, 600), color='blue')
        resized = resize_image_proportionally(image, 800)
        
        assert resized.size == (800, 480)  # Maintains aspect ratio
    
    def test_resize_height_larger(self):
        """Test resizing when height is larger dimension."""
        image = Image.new('RGB', (600, 1000), color='blue')
        resized = resize_image_proportionally(image, 800)
        
        assert resized.size == (480, 800)  # Maintains aspect ratio
    
    def test_square_image(self):
        """Test resizing square image."""
        image = Image.new('RGB', (1000, 1000), color='blue')
        resized = resize_image_proportionally(image, 500)
        
        assert resized.size == (500, 500)


class TestCompressImageSmart:
    """Test smart image compression."""
    
    def test_small_image_png_success(self):
        """Test that small images stay as PNG."""
        # Create a very small image that should fit in PNG
        image = Image.new('RGB', (50, 50), color='white')
        
        compressed, format_used, quality = compress_image_smart(image, TWITTER_MAX_SIZE_BYTES)
        
        assert format_used == 'PNG'
        assert quality == 95
        assert compressed.size == (50, 50)
    
    def test_jpeg_compression_path(self):
        """Test that JPEG compression path is used when PNG is too large."""
        # Create image and mock PNG size to be too large
        image = Image.new('RGB', (100, 100), color='red')
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock PNG to be too large, JPEG to be acceptable
            mock_size.side_effect = lambda img, fmt, quality=95: (
                10000000 if fmt == 'PNG' else 1000  # 10MB for PNG, 1KB for JPEG
            )
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            assert format_used == 'JPEG'
            assert quality <= 95  # Should use some JPEG quality
    
    def test_mode_conversion(self):
        """Test various color mode conversions."""
        # Test P mode with transparency
        image = Image.new('P', (100, 100))
        image.info['transparency'] = 0
        
        compressed, format_used, quality = compress_image_smart(image, TWITTER_MAX_SIZE_BYTES)
        
        assert compressed.mode in ('RGB', 'RGBA')
    
    @patch('pixelbliss.imaging.compression.ImageOps.exif_transpose')
    def test_exif_orientation(self, mock_transpose):
        """Test EXIF orientation handling."""
        image = Image.new('RGB', (100, 100), color='red')
        mock_transpose.return_value = image
        
        compress_image_smart(image, TWITTER_MAX_SIZE_BYTES)
        
        mock_transpose.assert_called_once()


class TestCompressImageFile:
    """Test file-based image compression."""
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            compress_image_file("/nonexistent/file.jpg")
    
    def test_no_compression_needed(self):
        """Test file that doesn't need compression."""
        # Create a small image file
        image = Image.new('RGB', (50, 50), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            output_path, final_size, format_used, quality = compress_image_file(
                tmp.name, target_size_mb=10.0
            )
            
            assert output_path == tmp.name
            assert format_used == 'ORIGINAL'
            assert quality == 95
            assert final_size < 10.0
            
            os.unlink(tmp.name)
    
    def test_compression_with_output_path(self):
        """Test compression with different output path."""
        image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as input_tmp:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as output_tmp:
                image.save(input_tmp.name, 'PNG')
                
                output_path, final_size, format_used, quality = compress_image_file(
                    input_tmp.name, 
                    output_path=output_tmp.name,
                    target_size_mb=0.001  # Very small to force compression
                )
                
                assert output_path == output_tmp.name
                assert os.path.exists(output_tmp.name)
                
                os.unlink(input_tmp.name)
                os.unlink(output_tmp.name)
    
    def test_copy_when_no_compression_needed(self):
        """Test file copying when no compression needed but different output path."""
        image = Image.new('RGB', (50, 50), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as input_tmp:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_tmp:
                image.save(input_tmp.name, 'PNG')
                
                output_path, final_size, format_used, quality = compress_image_file(
                    input_tmp.name, 
                    output_path=output_tmp.name,
                    target_size_mb=10.0  # Large enough to not need compression
                )
                
                assert output_path == output_tmp.name
                assert format_used == 'ORIGINAL'
                assert os.path.exists(output_tmp.name)
                
                os.unlink(input_tmp.name)
                os.unlink(output_tmp.name)


class TestPrepareForTwitterUpload:
    """Test Twitter upload preparation."""
    
    def test_no_config(self):
        """Test without configuration object."""
        # Create test files
        image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            processed_paths = prepare_for_twitter_upload([tmp.name])
            
            assert len(processed_paths) == 1
            assert processed_paths[0] == tmp.name
            
            os.unlink(tmp.name)
    
    def test_compression_disabled_in_config(self):
        """Test when compression is disabled in config."""
        # Mock config with compression disabled
        mock_config = Mock()
        mock_config.twitter_compression = Mock()
        mock_config.twitter_compression.enabled = False
        
        test_paths = ['/path/to/image1.jpg', '/path/to/image2.png']
        
        processed_paths = prepare_for_twitter_upload(test_paths, config=mock_config)
        
        assert processed_paths == test_paths
    
    def test_custom_max_size_from_config(self):
        """Test using custom max size from config."""
        # Mock config with custom max size
        mock_config = Mock()
        mock_config.twitter_compression = Mock()
        mock_config.twitter_compression.enabled = True
        mock_config.twitter_compression.max_size_mb = 3.0
        
        image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            with patch('pixelbliss.imaging.compression.compress_image_file') as mock_compress:
                mock_compress.return_value = (tmp.name, 2.5, 'ORIGINAL', 95)
                
                processed_paths = prepare_for_twitter_upload([tmp.name], config=mock_config)
                
                # Verify compress_image_file was called with custom target size
                mock_compress.assert_called_once_with(tmp.name, target_size_mb=3.0)
                assert len(processed_paths) == 1
            
            os.unlink(tmp.name)
    
    def test_multiple_files(self):
        """Test processing multiple files."""
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (100, 100), color='blue')
        ]
        
        temp_files = []
        try:
            for i, image in enumerate(images):
                tmp = tempfile.NamedTemporaryFile(suffix=f'_{i}.png', delete=False)
                image.save(tmp.name, 'PNG')
                temp_files.append(tmp.name)
            
            processed_paths = prepare_for_twitter_upload(temp_files)
            
            assert len(processed_paths) == 2
            assert all(os.path.exists(path) for path in processed_paths)
            
        finally:
            for tmp_file in temp_files:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
    
    def test_error_handling(self):
        """Test error handling for failed compression."""
        # Include one valid file and one invalid path
        image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            test_paths = [tmp.name, '/nonexistent/file.jpg']
            
            with patch('pixelbliss.imaging.compression.logger') as mock_logger:
                processed_paths = prepare_for_twitter_upload(test_paths)
                
                # Should include both paths (original file even if compression failed)
                assert len(processed_paths) == 2
                assert processed_paths[0] == tmp.name
                assert processed_paths[1] == '/nonexistent/file.jpg'
                
                # Should have logged an error
                mock_logger.error.assert_called_once()
            
            os.unlink(tmp.name)
    
    def test_config_without_twitter_compression(self):
        """Test config object without twitter_compression attribute."""
        mock_config = Mock()
        # Don't add twitter_compression attribute
        
        image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            processed_paths = prepare_for_twitter_upload([tmp.name], config=mock_config)
            
            assert len(processed_paths) == 1
            assert processed_paths[0] == tmp.name
            
            os.unlink(tmp.name)


class TestConstants:
    """Test module constants."""
    
    def test_twitter_constants(self):
        """Test Twitter-related constants."""
        assert TWITTER_MAX_SIZE_MB == 5
        assert TWITTER_MAX_SIZE_BYTES == 5 * 1024 * 1024
    
    def test_quality_levels(self):
        """Test quality level constants."""
        from pixelbliss.imaging.compression import QUALITY_LEVELS, MIN_QUALITY
        
        assert isinstance(QUALITY_LEVELS, list)
        assert all(isinstance(q, int) for q in QUALITY_LEVELS)
        assert all(50 <= q <= 95 for q in QUALITY_LEVELS)
        assert QUALITY_LEVELS == sorted(QUALITY_LEVELS, reverse=True)  # Should be descending
        assert MIN_QUALITY == 50


class TestIntegration:
    """Integration tests for the compression workflow."""
    
    def test_end_to_end_compression_workflow(self):
        """Test complete compression workflow."""
        # Create a larger image that will need compression
        image = Image.new('RGB', (2000, 2000), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG', optimize=False)  # Save unoptimized to ensure larger size
            
            # Check original size
            original_size = get_file_size_mb(tmp.name)
            
            # Process for Twitter upload
            processed_paths = prepare_for_twitter_upload([tmp.name])
            
            assert len(processed_paths) == 1
            processed_path = processed_paths[0]
            
            # Check final size
            final_size = get_file_size_mb(processed_path)
            
            # Should be under Twitter limit
            assert final_size <= TWITTER_MAX_SIZE_MB
            
            # Verify image is still valid
            with Image.open(processed_path) as final_image:
                assert final_image.size[0] > 0
                assert final_image.size[1] > 0
            
            os.unlink(tmp.name)
