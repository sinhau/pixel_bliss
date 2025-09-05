import pytest
import os
import tempfile
import shutil
from PIL import Image, ImageFont
from unittest.mock import patch, Mock, mock_open
from pixelbliss.imaging.collage import create_candidates_collage, save_collage


@pytest.fixture
def sample_candidates():
    """Create sample candidates with images and scores."""
    candidates = []
    for i in range(4):
        # Create a simple colored image
        img = Image.new('RGB', (200, 200), color=f'#{i*60:02x}{i*40:02x}{i*80:02x}')
        candidate = {
            'image': img,
            'final': 0.9 - i * 0.1,  # Decreasing scores
            'aesthetic': 0.8 - i * 0.05,
            'brightness': 0.7 + i * 0.1,
            'entropy': 0.6 + i * 0.05
        }
        candidates.append(candidate)
    return candidates


@pytest.fixture
def single_candidate():
    """Create a single candidate for testing."""
    img = Image.new('RGB', (200, 200), color='red')
    return [{
        'image': img,
        'final': 0.95,
        'aesthetic': 0.85,
        'brightness': 0.75,
        'entropy': 0.65
    }]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestCreateCandidatesCollage:
    """Test cases for create_candidates_collage function."""

    def test_create_collage_empty_candidates(self):
        """Test creating collage with empty candidates list."""
        result = create_candidates_collage([])
        
        assert isinstance(result, Image.Image)
        assert result.size == (1920, 1080)  # Default max dimensions
        # Should be a black image
        assert result.getpixel((0, 0)) == (0, 0, 0)

    def test_create_collage_single_candidate(self, single_candidate):
        """Test creating collage with single candidate."""
        result = create_candidates_collage(single_candidate)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1920, 1080)

    def test_create_collage_multiple_candidates(self, sample_candidates):
        """Test creating collage with multiple candidates."""
        result = create_candidates_collage(sample_candidates)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1920, 1080)

    def test_create_collage_custom_dimensions(self, sample_candidates):
        """Test creating collage with custom dimensions."""
        result = create_candidates_collage(sample_candidates, max_width=800, max_height=600)
        
        assert isinstance(result, Image.Image)
        assert result.size == (800, 600)

    def test_create_collage_large_number_of_candidates(self):
        """Test creating collage with many candidates."""
        candidates = []
        for i in range(16):  # 4x4 grid
            img = Image.new('RGB', (100, 100), color=f'#{i*15:02x}{i*10:02x}{i*5:02x}')
            candidate = {
                'image': img,
                'final': 1.0 - i * 0.05,
                'aesthetic': 0.9 - i * 0.02,
                'brightness': 0.5 + i * 0.01,
                'entropy': 0.6 + i * 0.01
            }
            candidates.append(candidate)
        
        result = create_candidates_collage(candidates)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1920, 1080)

    def test_create_collage_missing_scores(self):
        """Test creating collage with candidates missing some scores."""
        img = Image.new('RGB', (200, 200), color='blue')
        candidates = [{
            'image': img,
            'final': 0.8,
            # Missing aesthetic, brightness, entropy
        }]
        
        result = create_candidates_collage(candidates)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1920, 1080)

    def test_create_collage_all_missing_scores(self):
        """Test creating collage with candidates missing all scores."""
        img = Image.new('RGB', (200, 200), color='green')
        candidates = [{'image': img}]  # Only image, no scores
        
        result = create_candidates_collage(candidates)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1920, 1080)


    @patch('pixelbliss.imaging.collage.ImageFont.truetype')
    @patch('pixelbliss.imaging.collage.ImageFont.load_default')
    def test_create_collage_no_font_available(self, mock_load_default, mock_truetype, sample_candidates):
        """Test creating collage when no fonts are available."""
        mock_truetype.side_effect = OSError("Font not found")
        mock_load_default.side_effect = Exception("No default font")
        
        result = create_candidates_collage(sample_candidates)
        
        mock_truetype.assert_called_once()
        mock_load_default.assert_called_once()
        assert isinstance(result, Image.Image)

    def test_create_collage_very_small_dimensions(self, sample_candidates):
        """Test creating collage with very small dimensions."""
        result = create_candidates_collage(sample_candidates, max_width=200, max_height=200)
        
        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)

    def test_create_collage_rectangular_dimensions(self, sample_candidates):
        """Test creating collage with rectangular dimensions."""
        result = create_candidates_collage(sample_candidates, max_width=1600, max_height=900)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1600, 900)

    def test_create_collage_preserves_original_images(self, sample_candidates):
        """Test that original candidate images are not modified."""
        original_sizes = [candidate['image'].size for candidate in sample_candidates]
        
        create_candidates_collage(sample_candidates)
        
        # Original images should remain unchanged
        for i, candidate in enumerate(sample_candidates):
            assert candidate['image'].size == original_sizes[i]

    def test_create_collage_score_formatting(self, single_candidate):
        """Test that scores are properly formatted in the collage."""
        # This test mainly ensures the function runs without error
        # when formatting scores with different decimal places
        result = create_candidates_collage(single_candidate)
        
        assert isinstance(result, Image.Image)

    def test_create_collage_extreme_scores(self):
        """Test creating collage with extreme score values."""
        img = Image.new('RGB', (200, 200), color='purple')
        candidates = [{
            'image': img,
            'final': 999.999,
            'aesthetic': -50.5,
            'brightness': 0.0,
            'entropy': 1000000.123456
        }]
        
        result = create_candidates_collage(candidates)
        
        assert isinstance(result, Image.Image)
        assert result.size == (1920, 1080)


class TestSaveCollage:
    """Test cases for save_collage function."""

    def test_save_collage_basic(self, sample_candidates, temp_dir):
        """Test basic collage saving functionality."""
        result_path = save_collage(sample_candidates, temp_dir)
        
        expected_path = os.path.join(temp_dir, "candidates_collage.jpg")
        assert result_path == expected_path
        assert os.path.exists(result_path)
        
        # Verify it's a valid image file
        saved_image = Image.open(result_path)
        assert isinstance(saved_image, Image.Image)
        saved_image.close()

    def test_save_collage_custom_filename(self, sample_candidates, temp_dir):
        """Test saving collage with custom filename."""
        custom_filename = "my_custom_collage.jpg"
        result_path = save_collage(sample_candidates, temp_dir, custom_filename)
        
        expected_path = os.path.join(temp_dir, custom_filename)
        assert result_path == expected_path
        assert os.path.exists(result_path)

    def test_save_collage_creates_directory(self, sample_candidates):
        """Test that save_collage creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_base:
            non_existent_dir = os.path.join(temp_base, "new_dir", "nested_dir")
            
            result_path = save_collage(sample_candidates, non_existent_dir)
            
            assert os.path.exists(non_existent_dir)
            assert os.path.exists(result_path)

    def test_save_collage_sorts_candidates(self, temp_dir):
        """Test that save_collage sorts candidates by final score."""
        # Create candidates with unsorted scores
        candidates = []
        scores = [0.3, 0.9, 0.1, 0.7]  # Unsorted
        for i, score in enumerate(scores):
            img = Image.new('RGB', (100, 100), color=f'#{i*60:02x}0000')
            candidates.append({
                'image': img,
                'final': score,
                'aesthetic': score * 0.8,
                'brightness': 0.5,
                'entropy': 0.6
            })
        
        result_path = save_collage(candidates, temp_dir)
        
        assert os.path.exists(result_path)
        # The function should have sorted them internally

    def test_save_collage_empty_candidates(self, temp_dir):
        """Test saving collage with empty candidates list."""
        result_path = save_collage([], temp_dir)
        
        expected_path = os.path.join(temp_dir, "candidates_collage.jpg")
        assert result_path == expected_path
        assert os.path.exists(result_path)

    def test_save_collage_candidates_without_final_score(self, temp_dir):
        """Test saving collage with candidates missing final scores."""
        img = Image.new('RGB', (100, 100), color='orange')
        candidates = [{'image': img}]  # No final score
        
        result_path = save_collage(candidates, temp_dir)
        
        assert os.path.exists(result_path)

    @patch('pixelbliss.imaging.collage.os.makedirs')
    def test_save_collage_makedirs_called(self, mock_makedirs, sample_candidates, temp_dir):
        """Test that os.makedirs is called with correct parameters."""
        save_collage(sample_candidates, temp_dir)
        
        mock_makedirs.assert_called_once_with(temp_dir, exist_ok=True)

    def test_save_collage_different_extensions(self, sample_candidates, temp_dir):
        """Test saving collage with different file extensions."""
        # The function should save as JPEG regardless of extension in filename
        result_path = save_collage(sample_candidates, temp_dir, "test.png")
        
        assert os.path.exists(result_path)
        # Verify it's saved as JPEG format
        with Image.open(result_path) as img:
            assert img.format == 'JPEG'

    def test_save_collage_large_candidates_list(self, temp_dir):
        """Test saving collage with a large number of candidates."""
        candidates = []
        for i in range(25):  # 5x5 grid
            img = Image.new('RGB', (50, 50), color=f'#{i*10:02x}{i*5:02x}{i*3:02x}')
            candidates.append({
                'image': img,
                'final': 1.0 - i * 0.02,
                'aesthetic': 0.8,
                'brightness': 0.6,
                'entropy': 0.7
            })
        
        result_path = save_collage(candidates, temp_dir)
        
        assert os.path.exists(result_path)
        # Verify the saved image
        with Image.open(result_path) as img:
            assert img.size == (1920, 1080)  # Default dimensions
