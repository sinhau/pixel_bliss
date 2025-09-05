import pytest
import datetime
import pytz
from unittest.mock import Mock, patch
from pixelbliss.run_once import (
    category_by_time, category_by_random, select_category,
    normalize_and_rescore, today_local, now_iso, tweet_url, fs_abs
)


class TestCategorySelection:
    """Test category selection functions."""

    def test_category_by_time_basic(self, fixed_datetime):
        """Test time-based category selection."""
        categories = ["sci-fi", "tech", "nature"]
        rotation_minutes = 180  # 3 hours per category
        
        # At 14:30, we're in slot 4 (14*60+30 = 870 minutes, 870//180 = 4)
        # 4 % 3 = 1, so should select categories[1] = "tech"
        result = category_by_time(categories, rotation_minutes, fixed_datetime)
        assert result == "tech"

    def test_category_by_time_different_rotation(self, fixed_datetime):
        """Test time-based selection with different rotation period."""
        categories = ["a", "b", "c", "d"]
        rotation_minutes = 60  # 1 hour per category
        
        # At 14:30, we're in slot 14 (14*60+30 = 870 minutes, 870//60 = 14)
        # 14 % 4 = 2, so should select categories[2] = "c"
        result = category_by_time(categories, rotation_minutes, fixed_datetime)
        assert result == "c"

    def test_category_by_time_single_category(self, fixed_datetime):
        """Test time-based selection with single category."""
        categories = ["only"]
        rotation_minutes = 180
        
        result = category_by_time(categories, rotation_minutes, fixed_datetime)
        assert result == "only"

    def test_category_by_random(self):
        """Test random category selection."""
        categories = ["sci-fi", "tech", "nature"]
        
        # Run multiple times to ensure it returns valid categories
        for _ in range(10):
            result = category_by_random(categories)
            assert result in categories

    def test_select_category_random_method(self, sample_config):
        """Test select_category with random method."""
        sample_config.category_selection_method = "random"
        
        result = select_category(sample_config)
        assert result in sample_config.categories

    def test_select_category_time_method(self, sample_config, fixed_datetime):
        """Test select_category with time method."""
        sample_config.category_selection_method = "time"
        
        with patch('pixelbliss.run_once.datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = fixed_datetime
            result = select_category(sample_config)
            assert result in sample_config.categories


class TestNormalizeAndRescore:
    """Test scoring normalization and rescoring."""

    def test_normalize_and_rescore_basic(self, sample_config):
        """Test basic normalization and rescoring."""
        items = [
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.8},
            {"brightness": 200, "entropy": 5.0, "aesthetic": 0.6},
            {"brightness": 150, "entropy": 4.5, "aesthetic": 0.7}
        ]
        
        result = normalize_and_rescore(items, sample_config)
        
        # Check that all items have final scores
        for item in result:
            assert "final" in item
            assert isinstance(item["final"], float)
        
        # Verify normalization ranges (0-1 for brightness and entropy)
        # Item with highest brightness (200) should have normalized brightness = 1.0
        # Item with lowest brightness (100) should have normalized brightness = 0.0

    def test_normalize_and_rescore_empty_list(self, sample_config):
        """Test normalization with empty list."""
        items = []
        result = normalize_and_rescore(items, sample_config)
        assert result == []

    def test_normalize_and_rescore_single_item(self, sample_config):
        """Test normalization with single item."""
        items = [{"brightness": 100, "entropy": 4.0, "aesthetic": 0.8}]
        
        result = normalize_and_rescore(items, sample_config)
        
        assert len(result) == 1
        assert "final" in result[0]
        # With single item, normalized brightness and entropy should be 0.5
        expected_final = (
            sample_config.ranking.w_brightness * 0.5 +
            sample_config.ranking.w_entropy * 0.5 +
            sample_config.ranking.w_aesthetic * 0.8
        )
        assert abs(result[0]["final"] - expected_final) < 0.001

    def test_normalize_and_rescore_identical_values(self, sample_config):
        """Test normalization when all values are identical."""
        items = [
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.8},
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.6},
            {"brightness": 100, "entropy": 4.0, "aesthetic": 0.7}
        ]
        
        result = normalize_and_rescore(items, sample_config)
        
        # When min == max, normalized values should be 0.5
        for item in result:
            assert "final" in item
            expected_final = (
                sample_config.ranking.w_brightness * 0.5 +
                sample_config.ranking.w_entropy * 0.5 +
                sample_config.ranking.w_aesthetic * item["aesthetic"]
            )
            assert abs(item["final"] - expected_final) < 0.001


class TestUtilityFunctions:
    """Test utility functions."""

    def test_today_local(self):
        """Test today_local returns correct format."""
        with patch('pixelbliss.run_once.datetime.datetime') as mock_datetime:
            tz = pytz.timezone("America/Los_Angeles")
            mock_now = datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=tz)
            mock_datetime.now.return_value = mock_now
            
            result = today_local()
            assert result == "2024-01-15"

    def test_now_iso(self):
        """Test now_iso returns ISO format."""
        with patch('pixelbliss.run_once.datetime.datetime') as mock_datetime:
            tz = pytz.timezone("America/Los_Angeles")
            mock_now = datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=tz)
            mock_datetime.now.return_value = mock_now
            
            result = now_iso()
            assert result == mock_now.isoformat()

    def test_tweet_url(self):
        """Test tweet URL generation."""
        tweet_id = "1234567890"
        result = tweet_url(tweet_id)
        assert result == "https://x.com/user/status/1234567890"

    def test_fs_abs(self):
        """Test filesystem absolute path conversion."""
        path = "relative/path/to/file.jpg"
        result = fs_abs(path)
        assert result == path  # Currently just returns input
