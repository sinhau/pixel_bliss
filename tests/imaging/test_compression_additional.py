"""
Additional tests for image compression utilities to improve coverage.
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
    TWITTER_MAX_SIZE_MB,
    QUALITY_LEVELS,
    MIN_QUALITY
)


class TestCompressImageSmartAdditional:
    """Additional tests for compress_image_smart to improve coverage."""
    
    def test_rgba_to_rgb_conversion_for_jpeg(self):
        """Test RGBA to RGB conversion when using JPEG format."""
        # Create RGBA image
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock PNG to be too large, force JPEG path
            mock_size.side_effect = lambda img, fmt, quality=95: (
                10000000 if fmt == 'PNG' else 1000  # 10MB for PNG, 1KB for JPEG
            )
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            assert format_used == 'JPEG'
            # Should have converted RGBA to RGB
            assert compressed.mode == 'RGB'
    
    def test_dimension_reduction_path(self):
        """Test dimension reduction when quality reduction is insufficient."""
        # Create large image
        image = Image.new('RGB', (4000, 4000), color='red')
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock all sizes to be too large except for very small dimensions
            def mock_size_func(img, fmt, quality=95):
                if max(img.size) <= 1024:
                    return 1000  # Small enough
                return 10000000  # Too large
            
            mock_size.side_effect = mock_size_func
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            # Should have reduced dimensions
            assert max(compressed.size) <= 1024
    
    def test_fallback_compression(self):
        """Test fallback compression when all else fails."""
        # Create image
        image = Image.new('RGB', (2000, 2000), color='red')
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock all sizes to be too large to force fallback
            mock_size.return_value = 10000000
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            assert format_used == 'JPEG'
            assert quality == MIN_QUALITY
            assert max(compressed.size) <= 1024  # Should use 1024px fallback
    
    def test_rgba_fallback_conversion(self):
        """Test RGBA to RGB conversion in fallback path."""
        # Create RGBA image
        image = Image.new('RGBA', (2000, 2000), color=(255, 0, 0, 128))
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock all sizes to be too large to force fallback
            mock_size.return_value = 10000000
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            assert format_used == 'JPEG'
            assert quality == MIN_QUALITY
            assert compressed.mode == 'RGB'  # Should have converted RGBA to RGB
    
    def test_p_mode_without_transparency(self):
        """Test P mode image without transparency."""
        # Create P mode image without transparency
        image = Image.new('P', (100, 100))
        # Don't set transparency info
        
        compressed, format_used, quality = compress_image_smart(image, TWITTER_MAX_SIZE_BYTES)
        
        # Should convert to RGB
        assert compressed.mode == 'RGB'
    
    def test_dimension_steps_iteration(self):
        """Test iteration through different dimension steps."""
        # Create large image
        image = Image.new('RGB', (5000, 5000), color='red')
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock sizes to be acceptable only at 2048px
            def mock_size_func(img, fmt, quality=95):
                if max(img.size) <= 2048:
                    return 1000  # Small enough
                return 10000000  # Too large
            
            mock_size.side_effect = mock_size_func
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            # Should have reduced to 2048px or smaller
            assert max(compressed.size) <= 2048
    
    def test_resized_png_success(self):
        """Test successful PNG compression after resizing."""
        # Create large image
        image = Image.new('RGB', (4000, 4000), color='red')
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock original PNG to be too large, but resized PNG to be acceptable
            def mock_size_func(img, fmt, quality=95):
                if fmt == 'PNG' and max(img.size) <= 3840:
                    return 1000  # Small enough for resized PNG
                elif fmt == 'PNG':
                    return 10000000  # Too large for original PNG
                else:
                    return 10000000  # JPEG also too large to force PNG path
            
            mock_size.side_effect = mock_size_func
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            assert format_used == 'PNG'
            assert quality == 95
            assert max(compressed.size) <= 3840
    
    def test_skip_dimension_if_already_smaller(self):
        """Test skipping dimension reduction if image is already smaller."""
        # Create medium-sized image
        image = Image.new('RGB', (2000, 2000), color='red')
        
        with patch('pixelbliss.imaging.compression.get_image_size_bytes') as mock_size:
            # Mock to force dimension reduction path but skip larger dimensions
            def mock_size_func(img, fmt, quality=95):
                if max(img.size) <= 1920:
                    return 1000  # Small enough
                return 10000000  # Too large
            
            mock_size.side_effect = mock_size_func
            
            compressed, format_used, quality = compress_image_smart(image, 5000)
            
            # Should have reduced to 1920px or smaller
            assert max(compressed.size) <= 1920


class TestCompressImageFileAdditional:
    """Additional tests for compress_image_file to improve coverage."""
    
    def test_shutil_import_and_copy(self):
        """Test shutil import and file copying when no compression needed."""
        # Create small image
        image = Image.new('RGB', (50, 50), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as input_tmp:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_tmp:
                image.save(input_tmp.name, 'PNG')
                
                # Test the shutil.copy2 path
                output_path, final_size, format_used, quality = compress_image_file(
                    input_tmp.name, 
                    output_path=output_tmp.name,
                    target_size_mb=10.0  # Large enough to not need compression
                )
                
                assert output_path == output_tmp.name
                assert format_used == 'ORIGINAL'
                assert os.path.exists(output_tmp.name)
                
                # Verify file was actually copied
                with Image.open(output_tmp.name) as copied_image:
                    assert copied_image.size == (50, 50)
                
                os.unlink(input_tmp.name)
                os.unlink(output_tmp.name)
    
    def test_compression_needed_path(self):
        """Test the compression path when file is too large."""
        # Create a very large image that will definitely need compression
        image = Image.new('RGB', (3000, 3000), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as input_tmp:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as output_tmp:
                # Save as PNG to make it large (uncompressed)
                image.save(input_tmp.name, 'PNG', compress_level=0)
                
                # Verify the file is actually large
                input_size = os.path.getsize(input_tmp.name) / (1024 * 1024)
                assert input_size > 5.0, f"Input file should be > 5MB, got {input_size:.2f}MB"
                
                # Mock compress_image_smart to return a specific result
                with patch('pixelbliss.imaging.compression.compress_image_smart') as mock_compress:
                    # Create a smaller image for the mock return
                    compressed_image = Image.new('RGB', (1000, 1000), color='red')
                    mock_compress.return_value = (compressed_image, 'JPEG', 80)
                    
                    # Test compression with default target size
                    output_path, final_size, format_used, quality = compress_image_file(
                        input_tmp.name,
                        output_path=output_tmp.name,
                        target_size_mb=5.0  # Twitter default
                    )
                    
                    assert output_path == output_tmp.name
                    assert format_used == 'JPEG'
                    assert quality == 80
                    assert os.path.exists(output_tmp.name)
                    
                    # Verify compress_image_smart was called with correct parameters
                    mock_compress.assert_called_once()
                    call_args = mock_compress.call_args
                    # Check the second positional argument (target_size_bytes)
                    assert call_args[0][1] == int(5.0 * 1024 * 1024)  # Target size in bytes
                
                os.unlink(input_tmp.name)
                os.unlink(output_tmp.name)


class TestPrepareForTwitterUploadAdditional:
    """Additional tests for prepare_for_twitter_upload to improve coverage."""
    
    def test_compression_success_logging(self):
        """Test logging when compression is successful."""
        # Create image that will need compression
        image = Image.new('RGB', (200, 200), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            with patch('pixelbliss.imaging.compression.compress_image_file') as mock_compress:
                # Mock successful compression
                mock_compress.return_value = (tmp.name, 2.5, 'JPEG', 80)
                
                with patch('pixelbliss.imaging.compression.logger') as mock_logger:
                    processed_paths = prepare_for_twitter_upload([tmp.name])
                    
                    # Should log compression success
                    mock_logger.info.assert_any_call(f"Compressed {os.path.basename(tmp.name)}: 2.50MB (JPEG, Q80)")
                    assert len(processed_paths) == 1
            
            os.unlink(tmp.name)
    
    def test_no_compression_needed_logging(self):
        """Test logging when no compression is needed."""
        # Create small image
        image = Image.new('RGB', (50, 50), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            with patch('pixelbliss.imaging.compression.compress_image_file') as mock_compress:
                # Mock no compression needed
                mock_compress.return_value = (tmp.name, 0.5, 'ORIGINAL', 95)
                
                with patch('pixelbliss.imaging.compression.logger') as mock_logger:
                    processed_paths = prepare_for_twitter_upload([tmp.name])
                    
                    # Should log no compression needed
                    mock_logger.info.assert_any_call(f"No compression needed for {os.path.basename(tmp.name)}: 0.50MB")
                    assert len(processed_paths) == 1
            
            os.unlink(tmp.name)
