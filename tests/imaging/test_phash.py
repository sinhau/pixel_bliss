import pytest
from PIL import Image
from unittest.mock import patch, Mock, MagicMock
from pixelbliss.imaging.phash import phash_hex, is_duplicate


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def different_image():
    """Create a different image for testing."""
    return Image.new('RGB', (100, 100), color='blue')


@pytest.fixture
def similar_image():
    """Create a similar image for testing."""
    # Create an image that's very similar to the sample image
    img = Image.new('RGB', (100, 100), color='red')
    # Add a small variation
    pixels = img.load()
    pixels[50, 50] = (255, 0, 0)  # Single red pixel
    return img


class TestPhashHex:
    """Test cases for phash_hex function."""

    @patch('pixelbliss.imaging.phash.imagehash.phash')
    def test_phash_hex_basic(self, mock_phash, sample_image):
        """Test basic phash_hex functionality."""
        # Mock the imagehash.phash return value
        mock_hash = Mock()
        mock_hash.__str__ = Mock(return_value="abcd1234efgh5678")
        mock_phash.return_value = mock_hash
        
        result = phash_hex(sample_image)
        
        mock_phash.assert_called_once_with(sample_image)
        assert result == "abcd1234efgh5678"

    @patch('pixelbliss.imaging.phash.imagehash.phash')
    def test_phash_hex_different_images(self, mock_phash, sample_image, different_image):
        """Test that different images produce different hashes."""
        # Mock different hash values for different images
        def mock_phash_side_effect(image):
            mock_hash = Mock()
            if image == sample_image:
                mock_hash.__str__ = Mock(return_value="hash1234567890ab")
            else:
                mock_hash.__str__ = Mock(return_value="hash0987654321ba")
            return mock_hash
        
        mock_phash.side_effect = mock_phash_side_effect
        
        result1 = phash_hex(sample_image)
        result2 = phash_hex(different_image)
        
        assert result1 == "hash1234567890ab"
        assert result2 == "hash0987654321ba"
        assert result1 != result2

    @patch('pixelbliss.imaging.phash.imagehash.phash')
    def test_phash_hex_same_image_consistent(self, mock_phash, sample_image):
        """Test that the same image produces consistent hashes."""
        mock_hash = Mock()
        mock_hash.__str__ = Mock(return_value="consistent_hash_123")
        mock_phash.return_value = mock_hash
        
        result1 = phash_hex(sample_image)
        result2 = phash_hex(sample_image)
        
        assert result1 == result2
        assert result1 == "consistent_hash_123"

    @patch('pixelbliss.imaging.phash.imagehash.phash')
    def test_phash_hex_empty_hash(self, mock_phash, sample_image):
        """Test handling of empty hash string."""
        mock_hash = Mock()
        mock_hash.__str__ = Mock(return_value="")
        mock_phash.return_value = mock_hash
        
        result = phash_hex(sample_image)
        
        assert result == ""

    @patch('pixelbliss.imaging.phash.imagehash.phash')
    def test_phash_hex_long_hash(self, mock_phash, sample_image):
        """Test handling of long hash string."""
        long_hash = "a" * 64  # Very long hash
        mock_hash = Mock()
        mock_hash.__str__ = Mock(return_value=long_hash)
        mock_phash.return_value = mock_hash
        
        result = phash_hex(sample_image)
        
        assert result == long_hash


class TestIsDuplicate:
    """Test cases for is_duplicate function."""

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_no_recent_hashes(self, mock_hex_to_hash):
        """Test duplicate check with empty recent hashes list."""
        # Mock the hex_to_hash to return a mock hash object
        mock_hash = MagicMock()
        mock_hex_to_hash.return_value = mock_hash
        
        result = is_duplicate("abcd1234", [], 5)
        
        assert result is False
        # Should call hex_to_hash once for the input hash
        mock_hex_to_hash.assert_called_once_with("abcd1234")

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_identical_hash(self, mock_hex_to_hash):
        """Test duplicate detection with identical hash."""
        # Mock hash objects
        mock_current_hash = Mock()
        mock_recent_hash = Mock()
        
        # Mock the subtraction to return 0 (identical)
        mock_current_hash.__sub__ = Mock(return_value=0)
        
        mock_hex_to_hash.side_effect = [mock_current_hash, mock_recent_hash]
        
        result = is_duplicate("abcd1234", ["abcd1234"], 5)
        
        assert result is True
        assert mock_hex_to_hash.call_count == 2
        mock_current_hash.__sub__.assert_called_once_with(mock_recent_hash)

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_similar_hash_below_threshold(self, mock_hex_to_hash):
        """Test duplicate detection with similar hash below distance threshold."""
        mock_current_hash = Mock()
        mock_recent_hash = Mock()
        
        # Mock the subtraction to return 3 (below threshold of 5)
        mock_current_hash.__sub__ = Mock(return_value=3)
        
        mock_hex_to_hash.side_effect = [mock_current_hash, mock_recent_hash]
        
        result = is_duplicate("abcd1234", ["efgh5678"], 5)
        
        assert result is True

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_different_hash_above_threshold(self, mock_hex_to_hash):
        """Test duplicate detection with different hash above distance threshold."""
        mock_current_hash = Mock()
        mock_recent_hash = Mock()
        
        # Mock the subtraction to return 10 (above threshold of 5)
        mock_current_hash.__sub__ = Mock(return_value=10)
        
        mock_hex_to_hash.side_effect = [mock_current_hash, mock_recent_hash]
        
        result = is_duplicate("abcd1234", ["efgh5678"], 5)
        
        assert result is False

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_multiple_recent_hashes_no_match(self, mock_hex_to_hash):
        """Test duplicate detection with multiple recent hashes, no matches."""
        mock_current_hash = Mock()
        mock_recent_hash1 = Mock()
        mock_recent_hash2 = Mock()
        mock_recent_hash3 = Mock()
        
        # Mock all distances to be above threshold
        mock_current_hash.__sub__ = Mock(side_effect=[8, 12, 15])
        
        mock_hex_to_hash.side_effect = [
            mock_current_hash, mock_recent_hash1,
            mock_recent_hash2, mock_recent_hash3
        ]
        
        result = is_duplicate("abcd1234", ["hash1", "hash2", "hash3"], 5)
        
        assert result is False
        assert mock_hex_to_hash.call_count == 4  # 1 current + 3 recent
        assert mock_current_hash.__sub__.call_count == 3

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_multiple_recent_hashes_with_match(self, mock_hex_to_hash):
        """Test duplicate detection with multiple recent hashes, one match."""
        mock_current_hash = Mock()
        mock_recent_hash1 = Mock()
        mock_recent_hash2 = Mock()
        
        # First comparison above threshold, second below threshold
        mock_current_hash.__sub__ = Mock(side_effect=[8, 2])
        
        mock_hex_to_hash.side_effect = [
            mock_current_hash, mock_recent_hash1, mock_recent_hash2
        ]
        
        result = is_duplicate("abcd1234", ["hash1", "hash2"], 5)
        
        assert result is True
        # Should stop after finding first match
        assert mock_current_hash.__sub__.call_count == 2

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_zero_distance_threshold(self, mock_hex_to_hash):
        """Test duplicate detection with zero distance threshold."""
        mock_current_hash = Mock()
        mock_recent_hash = Mock()
        
        # Mock distance of 0 (identical)
        mock_current_hash.__sub__ = Mock(return_value=0)
        
        mock_hex_to_hash.side_effect = [mock_current_hash, mock_recent_hash]
        
        result = is_duplicate("abcd1234", ["abcd1234"], 0)
        
        assert result is False  # 0 is not < 0

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_high_distance_threshold(self, mock_hex_to_hash):
        """Test duplicate detection with high distance threshold."""
        mock_current_hash = Mock()
        mock_recent_hash = Mock()
        
        # Mock distance of 20
        mock_current_hash.__sub__ = Mock(return_value=20)
        
        mock_hex_to_hash.side_effect = [mock_current_hash, mock_recent_hash]
        
        result = is_duplicate("abcd1234", ["efgh5678"], 25)
        
        assert result is True  # 20 < 25

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_edge_case_distance_equals_threshold(self, mock_hex_to_hash):
        """Test duplicate detection when distance equals threshold."""
        mock_current_hash = Mock()
        mock_recent_hash = Mock()
        
        # Mock distance exactly equal to threshold
        mock_current_hash.__sub__ = Mock(return_value=5)
        
        mock_hex_to_hash.side_effect = [mock_current_hash, mock_recent_hash]
        
        result = is_duplicate("abcd1234", ["efgh5678"], 5)
        
        assert result is False  # 5 is not < 5

    def test_is_duplicate_empty_hash_string(self):
        """Test duplicate detection with empty hash string."""
        # Empty hash string will cause ValueError in imagehash.hex_to_hash
        with pytest.raises(ValueError):
            is_duplicate("", [], 5)

    @patch('pixelbliss.imaging.phash.imagehash.hex_to_hash')
    def test_is_duplicate_single_recent_hash(self, mock_hex_to_hash):
        """Test duplicate detection with single recent hash."""
        mock_current_hash = Mock()
        mock_recent_hash = Mock()
        
        mock_current_hash.__sub__ = Mock(return_value=3)
        mock_hex_to_hash.side_effect = [mock_current_hash, mock_recent_hash]
        
        result = is_duplicate("abcd1234", ["efgh5678"], 5)
        
        assert result is True
        assert mock_hex_to_hash.call_count == 2
