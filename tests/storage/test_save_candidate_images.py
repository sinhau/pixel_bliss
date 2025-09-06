import pytest
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from pixelbliss.storage.fs import save_candidate_images


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_candidates():
    """Create sample candidate dictionaries with images."""
    return [
        {
            'image': Image.new('RGB', (100, 100), color='red'),
            'prompt': 'red image prompt',
            'provider': 'test_provider',
            'model': 'test_model'
        },
        {
            'image': Image.new('RGB', (150, 150), color='blue'),
            'prompt': 'blue image prompt',
            'provider': 'test_provider',
            'model': 'test_model'
        },
        {
            'image': Image.new('RGB', (200, 200), color='green'),
            'prompt': 'green image prompt',
            'provider': 'test_provider',
            'model': 'test_model'
        }
    ]


class TestSaveCandidateImages:
    """Test cases for save_candidate_images function."""

    def test_save_candidate_images_basic(self, temp_dir, sample_candidates):
        """Test basic candidate image saving functionality."""
        result = save_candidate_images(temp_dir, sample_candidates)
        
        # Check return value structure
        assert isinstance(result, dict)
        assert len(result) == 3
        assert 'candidate_001' in result
        assert 'candidate_002' in result
        assert 'candidate_003' in result
        
        # Check public paths format
        candidates_dir = os.path.join(temp_dir, 'candidates')
        assert result['candidate_001'] == f"/{candidates_dir}/candidate_001.png"
        assert result['candidate_002'] == f"/{candidates_dir}/candidate_002.png"
        assert result['candidate_003'] == f"/{candidates_dir}/candidate_003.png"
        
        # Check files were actually created
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_001.png'))
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_002.png'))
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_003.png'))

    def test_save_candidate_images_creates_candidates_directory(self, temp_dir, sample_candidates):
        """Test that save_candidate_images creates the candidates subdirectory."""
        candidates_dir = os.path.join(temp_dir, 'candidates')
        
        # Directory shouldn't exist initially
        assert not os.path.exists(candidates_dir)
        
        result = save_candidate_images(temp_dir, sample_candidates)
        
        # Directory should be created
        assert os.path.exists(candidates_dir)
        assert os.path.isdir(candidates_dir)

    def test_save_candidate_images_empty_list(self, temp_dir):
        """Test saving with empty candidates list."""
        result = save_candidate_images(temp_dir, [])
        
        assert result == {}
        # Candidates directory should still be created
        candidates_dir = os.path.join(temp_dir, 'candidates')
        assert os.path.exists(candidates_dir)

    def test_save_candidate_images_single_candidate(self, temp_dir):
        """Test saving a single candidate image."""
        candidates = [{
            'image': Image.new('RGB', (100, 100), color='purple'),
            'prompt': 'single candidate',
            'provider': 'test'
        }]
        
        result = save_candidate_images(temp_dir, candidates)
        
        assert len(result) == 1
        assert 'candidate_001' in result
        
        candidates_dir = os.path.join(temp_dir, 'candidates')
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_001.png'))

    def test_save_candidate_images_many_candidates(self, temp_dir):
        """Test saving many candidate images with proper numbering."""
        # Create 15 candidates to test 3-digit numbering
        candidates = []
        for i in range(15):
            candidates.append({
                'image': Image.new('RGB', (50, 50), color=(i*10, i*10, i*10)),
                'prompt': f'candidate {i+1}',
                'provider': 'test'
            })
        
        result = save_candidate_images(temp_dir, candidates)
        
        assert len(result) == 15
        assert 'candidate_001' in result
        assert 'candidate_015' in result
        
        candidates_dir = os.path.join(temp_dir, 'candidates')
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_001.png'))
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_015.png'))

    def test_save_candidate_images_candidates_without_image_key(self, temp_dir):
        """Test handling candidates that don't have 'image' key."""
        candidates = [
            {
                'image': Image.new('RGB', (100, 100), color='red'),
                'prompt': 'valid candidate'
            },
            {
                'prompt': 'invalid candidate - no image',
                'provider': 'test'
            },
            {
                'image': Image.new('RGB', (100, 100), color='blue'),
                'prompt': 'another valid candidate'
            }
        ]
        
        result = save_candidate_images(temp_dir, candidates)
        
        # Should only save candidates with 'image' key
        assert len(result) == 2
        assert 'candidate_001' in result
        assert 'candidate_003' in result  # Note: numbering continues despite skipped candidate
        
        candidates_dir = os.path.join(temp_dir, 'candidates')
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_001.png'))
        assert not os.path.exists(os.path.join(candidates_dir, 'candidate_002.png'))
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_003.png'))

    def test_save_candidate_images_creates_nested_directories(self):
        """Test creating deeply nested directories."""
        with tempfile.TemporaryDirectory() as base_temp:
            nested_dir = os.path.join(base_temp, 'level1', 'level2', 'level3')
            candidates = [{
                'image': Image.new('RGB', (100, 100), color='yellow'),
                'prompt': 'nested test'
            }]
            
            result = save_candidate_images(nested_dir, candidates)
            
            candidates_dir = os.path.join(nested_dir, 'candidates')
            assert os.path.exists(candidates_dir)
            assert os.path.exists(os.path.join(candidates_dir, 'candidate_001.png'))

    def test_save_candidate_images_overwrites_existing(self, temp_dir):
        """Test that save_candidate_images overwrites existing files."""
        candidates1 = [{
            'image': Image.new('RGB', (100, 100), color='red'),
            'prompt': 'first save'
        }]
        
        # Save first time
        result1 = save_candidate_images(temp_dir, candidates1)
        
        # Save different image with same structure
        candidates2 = [{
            'image': Image.new('RGB', (50, 50), color='yellow'),
            'prompt': 'second save'
        }]
        result2 = save_candidate_images(temp_dir, candidates2)
        
        assert result1 == result2
        
        candidates_dir = os.path.join(temp_dir, 'candidates')
        assert os.path.exists(os.path.join(candidates_dir, 'candidate_001.png'))
        
        # Verify the image was actually overwritten by checking size
        saved_image = Image.open(os.path.join(candidates_dir, 'candidate_001.png'))
        assert saved_image.size == (50, 50)

    def test_save_candidate_images_different_image_formats(self, temp_dir):
        """Test saving different types of images."""
        candidates = [
            {
                'image': Image.new('RGB', (100, 100), color='red'),
                'prompt': 'rgb image'
            },
            {
                'image': Image.new('RGBA', (100, 100), color=(255, 0, 0, 128)),
                'prompt': 'rgba image'
            },
            {
                'image': Image.new('L', (100, 100), color=128),
                'prompt': 'grayscale image'
            }
        ]
        
        result = save_candidate_images(temp_dir, candidates)
        
        assert len(result) == 3
        candidates_dir = os.path.join(temp_dir, 'candidates')
        
        for i in range(1, 4):
            filename = f'candidate_{i:03d}.png'
            assert os.path.exists(os.path.join(candidates_dir, filename))
            # Verify they're all saved as PNG
            saved_image = Image.open(os.path.join(candidates_dir, filename))
            assert saved_image.format == 'PNG'

    def test_save_candidate_images_preserves_candidate_data(self, temp_dir, sample_candidates):
        """Test that the function doesn't modify the original candidate data."""
        original_candidates = [dict(c) for c in sample_candidates]  # Deep copy
        
        result = save_candidate_images(temp_dir, sample_candidates)
        
        # Original candidates should be unchanged
        assert len(sample_candidates) == len(original_candidates)
        for i, (original, current) in enumerate(zip(original_candidates, sample_candidates)):
            assert original.keys() == current.keys()
            assert original['prompt'] == current['prompt']
            assert original['provider'] == current['provider']
            assert original['model'] == current['model']
            # Images should be the same object
            assert original['image'] is current['image']

    def test_save_candidate_images_with_mixed_candidate_types(self, temp_dir):
        """Test saving candidates with various metadata structures."""
        candidates = [
            {
                'image': Image.new('RGB', (100, 100), color='red'),
                'prompt': 'minimal candidate'
            },
            {
                'image': Image.new('RGB', (100, 100), color='blue'),
                'prompt': 'full candidate',
                'provider': 'test_provider',
                'model': 'test_model',
                'seed': 12345,
                'aesthetic': 0.85,
                'brightness': 0.6,
                'entropy': 0.7
            },
            {
                'image': Image.new('RGB', (100, 100), color='green'),
                'prompt': 'another candidate',
                'extra_field': 'should be ignored'
            }
        ]
        
        result = save_candidate_images(temp_dir, candidates)
        
        assert len(result) == 3
        candidates_dir = os.path.join(temp_dir, 'candidates')
        
        # All should be saved regardless of metadata differences
        for i in range(1, 4):
            filename = f'candidate_{i:03d}.png'
            assert os.path.exists(os.path.join(candidates_dir, filename))
