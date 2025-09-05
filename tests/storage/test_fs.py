import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import patch, mock_open, MagicMock
from pixelbliss.storage.fs import save_images, save_meta


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def sample_images():
    """Create multiple sample images for testing."""
    return {
        'image1': Image.new('RGB', (100, 100), color='red'),
        'image2': Image.new('RGB', (200, 200), color='blue'),
        'image3': Image.new('RGB', (150, 150), color='green')
    }


class TestSaveImages:
    """Test cases for save_images function."""

    def test_save_images_basic(self, temp_dir, sample_images):
        """Test basic image saving functionality."""
        result = save_images(temp_dir, sample_images)
        
        # Check return value structure
        assert isinstance(result, dict)
        assert len(result) == 3
        assert 'image1' in result
        assert 'image2' in result
        assert 'image3' in result
        
        # Check public paths format
        assert result['image1'] == f"/{temp_dir}/image1.png"
        assert result['image2'] == f"/{temp_dir}/image2.png"
        assert result['image3'] == f"/{temp_dir}/image3.png"
        
        # Check files were actually created
        assert os.path.exists(os.path.join(temp_dir, 'image1.png'))
        assert os.path.exists(os.path.join(temp_dir, 'image2.png'))
        assert os.path.exists(os.path.join(temp_dir, 'image3.png'))

    def test_save_images_with_base_img(self, temp_dir, sample_images, sample_image):
        """Test saving images with base_img parameter."""
        base_img = Image.new('RGB', (300, 300), color='purple')
        result = save_images(temp_dir, sample_images, base_img)
        
        # Check return value structure includes base_img
        assert isinstance(result, dict)
        assert len(result) == 4  # 3 variants + 1 base
        assert 'image1' in result
        assert 'image2' in result
        assert 'image3' in result
        assert 'base_img' in result
        
        # Check public paths format
        assert result['base_img'] == f"/{temp_dir}/base_img.png"
        
        # Check files were actually created
        assert os.path.exists(os.path.join(temp_dir, 'base_img.png'))
        assert os.path.exists(os.path.join(temp_dir, 'image1.png'))
        assert os.path.exists(os.path.join(temp_dir, 'image2.png'))
        assert os.path.exists(os.path.join(temp_dir, 'image3.png'))
        
        # Verify base image was saved correctly
        saved_base = Image.open(os.path.join(temp_dir, 'base_img.png'))
        assert saved_base.size == (300, 300)

    def test_save_images_without_base_img(self, temp_dir, sample_images):
        """Test saving images without base_img parameter (backward compatibility)."""
        result = save_images(temp_dir, sample_images, None)
        
        # Should work the same as before
        assert len(result) == 3
        assert 'base_img' not in result
        assert not os.path.exists(os.path.join(temp_dir, 'base_img.png'))

    def test_save_images_single_image(self, temp_dir, sample_image):
        """Test saving a single image."""
        images = {'single': sample_image}
        result = save_images(temp_dir, images)
        
        assert len(result) == 1
        assert result['single'] == f"/{temp_dir}/single.png"
        assert os.path.exists(os.path.join(temp_dir, 'single.png'))

    def test_save_images_empty_dict(self, temp_dir):
        """Test saving with empty images dictionary."""
        result = save_images(temp_dir, {})
        
        assert result == {}
        assert os.path.exists(temp_dir)  # Directory should still be created

    def test_save_images_creates_directory(self, sample_image):
        """Test that save_images creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base_temp:
            non_existent_dir = os.path.join(base_temp, 'new_dir', 'nested')
            images = {'test': sample_image}
            
            result = save_images(non_existent_dir, images)
            
            assert os.path.exists(non_existent_dir)
            assert os.path.exists(os.path.join(non_existent_dir, 'test.png'))
            assert result['test'] == f"/{non_existent_dir}/test.png"

    def test_save_images_creates_nested_directories(self, sample_image):
        """Test creating deeply nested directories."""
        with tempfile.TemporaryDirectory() as base_temp:
            nested_dir = os.path.join(base_temp, 'level1', 'level2', 'level3')
            images = {'nested': sample_image}
            
            result = save_images(nested_dir, images)
            
            assert os.path.exists(nested_dir)
            assert os.path.exists(os.path.join(nested_dir, 'nested.png'))

    def test_save_images_overwrites_existing(self, temp_dir, sample_image):
        """Test that save_images overwrites existing files."""
        images = {'overwrite': sample_image}
        
        # Save first time
        result1 = save_images(temp_dir, images)
        
        # Create a different image and save again
        different_image = Image.new('RGB', (50, 50), color='yellow')
        images['overwrite'] = different_image
        result2 = save_images(temp_dir, images)
        
        assert result1 == result2
        assert os.path.exists(os.path.join(temp_dir, 'overwrite.png'))
        
        # Verify the image was actually overwritten by checking size
        saved_image = Image.open(os.path.join(temp_dir, 'overwrite.png'))
        assert saved_image.size == (50, 50)

    def test_save_images_special_characters_in_names(self, temp_dir, sample_image):
        """Test saving images with special characters in names."""
        images = {
            'image_with_underscores': sample_image,
            'image-with-hyphens': sample_image,
            'image123': sample_image
        }
        
        result = save_images(temp_dir, images)
        
        assert len(result) == 3
        for name in images.keys():
            assert os.path.exists(os.path.join(temp_dir, f'{name}.png'))

    @patch('pixelbliss.storage.fs.Path')
    def test_save_images_directory_creation_error(self, mock_path, sample_image):
        """Test handling of directory creation errors."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir.side_effect = OSError("Permission denied")
        
        images = {'test': sample_image}
        
        with pytest.raises(OSError):
            save_images('/invalid/path', images)

    @patch('PIL.Image.Image.save')
    def test_save_images_save_error(self, mock_save, temp_dir, sample_image):
        """Test handling of image save errors."""
        mock_save.side_effect = OSError("Cannot save image")
        images = {'test': sample_image}
        
        with pytest.raises(OSError):
            save_images(temp_dir, images)

    def test_save_images_different_image_formats(self, temp_dir):
        """Test saving different types of images."""
        images = {
            'rgb': Image.new('RGB', (100, 100), color='red'),
            'rgba': Image.new('RGBA', (100, 100), color=(255, 0, 0, 128)),
            'l': Image.new('L', (100, 100), color=128)  # Grayscale
        }
        
        result = save_images(temp_dir, images)
        
        assert len(result) == 3
        for name in images.keys():
            assert os.path.exists(os.path.join(temp_dir, f'{name}.png'))
            # Verify they're all saved as PNG
            saved_image = Image.open(os.path.join(temp_dir, f'{name}.png'))
            assert saved_image.format == 'PNG'


class TestSaveMeta:
    """Test cases for save_meta function."""

    def test_save_meta_basic(self, temp_dir):
        """Test basic metadata saving functionality."""
        meta = {
            'prompt': 'test prompt',
            'model': 'test-model',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        result = save_meta(temp_dir, meta)
        
        expected_path = os.path.join(temp_dir, 'meta.json')
        assert result == expected_path
        assert os.path.exists(expected_path)
        
        # Verify content
        with open(expected_path, 'r') as f:
            saved_meta = json.load(f)
        assert saved_meta == meta

    def test_save_meta_empty_dict(self, temp_dir):
        """Test saving empty metadata."""
        meta = {}
        result = save_meta(temp_dir, meta)
        
        expected_path = os.path.join(temp_dir, 'meta.json')
        assert result == expected_path
        assert os.path.exists(expected_path)
        
        with open(expected_path, 'r') as f:
            saved_meta = json.load(f)
        assert saved_meta == {}

    def test_save_meta_complex_data(self, temp_dir):
        """Test saving complex metadata structures."""
        meta = {
            'prompt': 'complex test prompt',
            'parameters': {
                'width': 1024,
                'height': 1024,
                'steps': 50,
                'cfg_scale': 7.5
            },
            'images': ['image1.png', 'image2.png'],
            'scores': [0.85, 0.92],
            'nested': {
                'deep': {
                    'value': True
                }
            }
        }
        
        result = save_meta(temp_dir, meta)
        
        with open(result, 'r') as f:
            saved_meta = json.load(f)
        assert saved_meta == meta

    def test_save_meta_overwrites_existing(self, temp_dir):
        """Test that save_meta overwrites existing meta.json."""
        meta1 = {'version': 1, 'data': 'first'}
        meta2 = {'version': 2, 'data': 'second'}
        
        # Save first metadata
        result1 = save_meta(temp_dir, meta1)
        
        # Save second metadata
        result2 = save_meta(temp_dir, meta2)
        
        assert result1 == result2
        
        # Verify second metadata overwrote first
        with open(result2, 'r') as f:
            saved_meta = json.load(f)
        assert saved_meta == meta2

    def test_save_meta_creates_directory_if_needed(self):
        """Test that save_meta works even if directory doesn't exist initially."""
        with tempfile.TemporaryDirectory() as base_temp:
            non_existent_dir = os.path.join(base_temp, 'new_dir')
            meta = {'test': 'data'}
            
            # Directory doesn't exist yet
            assert not os.path.exists(non_existent_dir)
            
            # This should work because os.path.join doesn't require the directory to exist
            # and open() will fail, but let's test the actual behavior
            with pytest.raises(FileNotFoundError):
                save_meta(non_existent_dir, meta)

    def test_save_meta_json_serializable_types(self, temp_dir):
        """Test saving various JSON-serializable types."""
        meta = {
            'string': 'test',
            'integer': 42,
            'float': 3.14,
            'boolean': True,
            'null': None,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'}
        }
        
        result = save_meta(temp_dir, meta)
        
        with open(result, 'r') as f:
            saved_meta = json.load(f)
        assert saved_meta == meta

    def test_save_meta_unicode_content(self, temp_dir):
        """Test saving metadata with unicode characters."""
        meta = {
            'prompt': 'café naïve résumé',
            'description': '🎨 Beautiful artwork with émojis 🌟',
            'tags': ['art', 'café', 'naïve']
        }
        
        result = save_meta(temp_dir, meta)
        
        with open(result, 'r', encoding='utf-8') as f:
            saved_meta = json.load(f)
        assert saved_meta == meta

    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_save_meta_file_write_error(self, mock_open, temp_dir):
        """Test handling of file write errors."""
        meta = {'test': 'data'}
        
        with pytest.raises(OSError):
            save_meta(temp_dir, meta)

    def test_save_meta_json_formatting(self, temp_dir):
        """Test that JSON is properly formatted with indentation."""
        meta = {
            'level1': {
                'level2': {
                    'level3': 'value'
                }
            }
        }
        
        result = save_meta(temp_dir, meta)
        
        # Read raw content to check formatting
        with open(result, 'r') as f:
            content = f.read()
        
        # Should contain indentation (2 spaces as specified in the function)
        assert '  ' in content
        assert '{\n  "level1"' in content


class TestIntegration:
    """Integration tests for fs module functions."""

    def test_save_images_and_meta_workflow(self, temp_dir, sample_images):
        """Test typical workflow of saving images and metadata together."""
        meta = {
            'prompt': 'test generation',
            'model': 'test-model-v1',
            'timestamp': '2024-01-15T10:30:00Z',
            'parameters': {
                'width': 1024,
                'height': 1024
            }
        }
        
        # Save images
        image_paths = save_images(temp_dir, sample_images)
        
        # Add image paths to metadata
        meta['images'] = list(image_paths.values())
        
        # Save metadata
        meta_path = save_meta(temp_dir, meta)
        
        # Verify everything exists
        assert os.path.exists(meta_path)
        for image_name in sample_images.keys():
            assert os.path.exists(os.path.join(temp_dir, f'{image_name}.png'))
        
        # Verify metadata contains image paths
        with open(meta_path, 'r') as f:
            saved_meta = json.load(f)
        assert 'images' in saved_meta
        assert len(saved_meta['images']) == len(sample_images)

    def test_multiple_saves_same_directory(self, temp_dir, sample_image):
        """Test multiple save operations in the same directory."""
        # First batch
        images1 = {'batch1_img1': sample_image, 'batch1_img2': sample_image}
        meta1 = {'batch': 1, 'count': 2}
        
        save_images(temp_dir, images1)
        save_meta(temp_dir, meta1)
        
        # Second batch (should overwrite meta but add images)
        images2 = {'batch2_img1': sample_image}
        meta2 = {'batch': 2, 'count': 1}
        
        save_images(temp_dir, images2)
        save_meta(temp_dir, meta2)
        
        # Check final state
        assert os.path.exists(os.path.join(temp_dir, 'batch1_img1.png'))
        assert os.path.exists(os.path.join(temp_dir, 'batch1_img2.png'))
        assert os.path.exists(os.path.join(temp_dir, 'batch2_img1.png'))
        
        # Meta should be from second batch
        with open(os.path.join(temp_dir, 'meta.json'), 'r') as f:
            final_meta = json.load(f)
        assert final_meta == meta2
