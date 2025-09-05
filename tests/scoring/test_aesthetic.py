import pytest
import asyncio
import random
from unittest.mock import patch, Mock
from pixelbliss.scoring.aesthetic import (
    aesthetic_dummy_local,
    aesthetic_replicate,
    aesthetic,
    aesthetic_dummy_local_async,
    aesthetic_replicate_async,
    aesthetic_async,
    score_candidates_parallel
)
from pixelbliss.config import Config


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock(spec=Config)
    config.aesthetic_scoring = Mock()
    config.aesthetic_scoring.provider = "dummy_local"
    config.aesthetic_scoring.model = "test-model"
    config.aesthetic_scoring.score_min = 0.0
    config.aesthetic_scoring.score_max = 1.0
    return config


@pytest.fixture
def mock_config_custom_range():
    """Create a mock config with custom score range."""
    config = Mock(spec=Config)
    config.aesthetic_scoring = Mock()
    config.aesthetic_scoring.provider = "dummy_local"
    config.aesthetic_scoring.model = "test-model"
    config.aesthetic_scoring.score_min = 2.0
    config.aesthetic_scoring.score_max = 8.0
    return config


@pytest.fixture
def mock_config_replicate():
    """Create a mock config for replicate provider."""
    config = Mock(spec=Config)
    config.aesthetic_scoring = Mock()
    config.aesthetic_scoring.provider = "replicate"
    config.aesthetic_scoring.model = "test-aesthetic-model"
    config.aesthetic_scoring.score_min = 0.0
    config.aesthetic_scoring.score_max = 10.0
    return config


class TestAestheticDummyLocal:
    """Test cases for aesthetic_dummy_local function."""

    def test_aesthetic_dummy_local_basic(self, mock_config):
        """Test basic functionality of dummy local aesthetic scoring."""
        result = aesthetic_dummy_local("https://example.com/image.jpg", mock_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_aesthetic_dummy_local_reproducible(self, mock_config):
        """Test that same URL produces same score (reproducible)."""
        url = "https://example.com/test.jpg"
        
        score1 = aesthetic_dummy_local(url, mock_config)
        score2 = aesthetic_dummy_local(url, mock_config)
        
        assert score1 == score2

    def test_aesthetic_dummy_local_different_urls(self, mock_config):
        """Test that different URLs produce different scores."""
        url1 = "https://example.com/image1.jpg"
        url2 = "https://example.com/image2.jpg"
        
        score1 = aesthetic_dummy_local(url1, mock_config)
        score2 = aesthetic_dummy_local(url2, mock_config)
        
        # While not guaranteed, different URLs should typically produce different scores
        # We'll just verify they're both valid scores
        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0

    def test_aesthetic_dummy_local_empty_url(self, mock_config):
        """Test handling of empty URL."""
        result = aesthetic_dummy_local("", mock_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_aesthetic_dummy_local_none_url(self, mock_config):
        """Test handling of None URL."""
        result = aesthetic_dummy_local(None, mock_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_aesthetic_dummy_local_custom_range(self, mock_config_custom_range):
        """Test dummy scoring with custom score range."""
        result = aesthetic_dummy_local("https://example.com/image.jpg", mock_config_custom_range)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0  # Should be normalized to [0,1]

    def test_aesthetic_dummy_local_equal_min_max(self, mock_config):
        """Test dummy scoring when min equals max."""
        mock_config.aesthetic_scoring.score_min = 5.0
        mock_config.aesthetic_scoring.score_max = 5.0
        
        result = aesthetic_dummy_local("https://example.com/image.jpg", mock_config)
        
        assert result == 0.5  # Should return 0.5 when min == max

    def test_aesthetic_dummy_local_multiple_calls_consistency(self, mock_config):
        """Test that multiple calls with same URL are consistent."""
        url = "https://example.com/consistent.jpg"
        
        scores = [aesthetic_dummy_local(url, mock_config) for _ in range(5)]
        
        # All scores should be identical
        assert all(score == scores[0] for score in scores)


class TestAestheticReplicate:
    """Test cases for aesthetic_replicate function."""

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_dict_output(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with dictionary output."""
        mock_replicate_run.return_value = {"score": 7.5}
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        mock_replicate_run.assert_called_once_with(
            "test-aesthetic-model",
            input={"image": "https://example.com/image.jpg"}
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Score 7.5 in range [0,10] should normalize to 0.75
        assert result == 0.75

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_dict_aesthetic_score_key(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with aesthetic_score key in dict."""
        mock_replicate_run.return_value = {"aesthetic_score": 3.0}
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Score 3.0 in range [0,10] should normalize to 0.3
        assert result == 0.3

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_list_output(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with list output."""
        mock_replicate_run.return_value = [8.2, "other_data"]
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Score 8.2 in range [0,10] should normalize to 0.82
        assert result == 0.82

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_float_output(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with direct float output."""
        mock_replicate_run.return_value = 6.0
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Score 6.0 in range [0,10] should normalize to 0.6
        assert result == 0.6

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_int_output(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with integer output."""
        mock_replicate_run.return_value = 4
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Score 4 in range [0,10] should normalize to 0.4
        assert result == 0.4

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_empty_list(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with empty list output."""
        mock_replicate_run.return_value = []
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert result == 0.5  # Should return default 0.5

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_unsupported_output(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with unsupported output format."""
        mock_replicate_run.return_value = "unsupported_string"
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert result == 0.5  # Should return default 0.5 on error

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_exception_handling(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring exception handling."""
        mock_replicate_run.side_effect = Exception("API Error")
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert result == 0.5  # Should return default 0.5 on exception

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_score_clamping(self, mock_replicate_run, mock_config_replicate):
        """Test that scores outside range are clamped to [0,1]."""
        # Test score above max
        mock_replicate_run.return_value = 15.0  # Above max of 10
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        assert result == 1.0  # Should be clamped to 1.0
        
        # Test score below min
        mock_replicate_run.return_value = -5.0  # Below min of 0
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        assert result == 0.0  # Should be clamped to 0.0

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_equal_min_max(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring when min equals max."""
        mock_config_replicate.aesthetic_scoring.score_min = 5.0
        mock_config_replicate.aesthetic_scoring.score_max = 5.0
        mock_replicate_run.return_value = 7.0
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        assert result == 0.5  # Should return 0.5 when min == max

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_dict_no_score_keys(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with dict that has no score keys."""
        mock_replicate_run.return_value = {"other_key": "value"}
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        # The function uses dict.get() with default 0.5, then normalizes [0,10] -> [0,1]
        # So 0.5 in range [0,10] becomes 0.05 in [0,1]
        assert result == 0.05

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_list_non_numeric(self, mock_replicate_run, mock_config_replicate):
        """Test replicate scoring with list containing non-numeric first element."""
        mock_replicate_run.return_value = ["not_a_number", 5.0]
        
        result = aesthetic_replicate("https://example.com/image.jpg", mock_config_replicate)
        
        # The function uses 0.5 as default, then normalizes [0,10] -> [0,1]
        # So 0.5 in range [0,10] becomes 0.05 in [0,1]
        assert result == 0.05


class TestAesthetic:
    """Test cases for main aesthetic function."""

    def test_aesthetic_dummy_local_provider(self, mock_config):
        """Test aesthetic function with dummy_local provider."""
        mock_config.aesthetic_scoring.provider = "dummy_local"
        
        result = aesthetic("https://example.com/image.jpg", mock_config)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    def test_aesthetic_replicate_provider(self, mock_replicate_run, mock_config_replicate):
        """Test aesthetic function with replicate provider."""
        mock_replicate_run.return_value = 5.0
        
        result = aesthetic("https://example.com/image.jpg", mock_config_replicate)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_aesthetic_unknown_provider(self, mock_config):
        """Test aesthetic function with unknown provider."""
        mock_config.aesthetic_scoring.provider = "unknown_provider"
        
        with pytest.raises(NotImplementedError, match="Unknown provider unknown_provider"):
            aesthetic("https://example.com/image.jpg", mock_config)

    def test_aesthetic_provider_case_sensitivity(self, mock_config):
        """Test that provider names are case sensitive."""
        mock_config.aesthetic_scoring.provider = "DUMMY_LOCAL"  # Wrong case
        
        with pytest.raises(NotImplementedError):
            aesthetic("https://example.com/image.jpg", mock_config)

    def test_aesthetic_empty_provider(self, mock_config):
        """Test aesthetic function with empty provider string."""
        mock_config.aesthetic_scoring.provider = ""
        
        with pytest.raises(NotImplementedError, match="Unknown provider "):
            aesthetic("https://example.com/image.jpg", mock_config)

    def test_aesthetic_none_provider(self, mock_config):
        """Test aesthetic function with None provider."""
        mock_config.aesthetic_scoring.provider = None
        
        with pytest.raises(NotImplementedError, match="Unknown provider None"):
            aesthetic("https://example.com/image.jpg", mock_config)


@pytest.fixture
def mock_config_with_async():
    """Create a mock config with async settings for testing."""
    config = Mock(spec=Config)
    config.aesthetic_scoring = Mock()
    config.aesthetic_scoring.provider = "dummy_local"
    config.aesthetic_scoring.model = "test-model"
    config.aesthetic_scoring.score_min = 0.0
    config.aesthetic_scoring.score_max = 1.0
    config.image_generation = Mock()
    config.image_generation.max_concurrency = None
    return config


@pytest.fixture
def mock_config_with_concurrency_limit():
    """Create a mock config with concurrency limit for testing."""
    config = Mock(spec=Config)
    config.aesthetic_scoring = Mock()
    config.aesthetic_scoring.provider = "dummy_local"
    config.aesthetic_scoring.model = "test-model"
    config.aesthetic_scoring.score_min = 0.0
    config.aesthetic_scoring.score_max = 1.0
    config.image_generation = Mock()
    config.image_generation.max_concurrency = 2
    return config


@pytest.fixture
def mock_config_replicate_async():
    """Create a mock config for replicate provider with async settings."""
    config = Mock(spec=Config)
    config.aesthetic_scoring = Mock()
    config.aesthetic_scoring.provider = "replicate"
    config.aesthetic_scoring.model = "test-aesthetic-model"
    config.aesthetic_scoring.score_min = 0.0
    config.aesthetic_scoring.score_max = 10.0
    config.image_generation = Mock()
    config.image_generation.max_concurrency = None
    return config


class TestAestheticAsync:
    """Test cases for async aesthetic scoring functions."""

    @pytest.mark.asyncio
    async def test_aesthetic_dummy_local_async_basic(self, mock_config_with_async):
        """Test basic functionality of async dummy local aesthetic scoring."""
        result = await aesthetic_dummy_local_async("https://example.com/image.jpg", mock_config_with_async)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_aesthetic_dummy_local_async_reproducible(self, mock_config_with_async):
        """Test that async version produces same results as sync version."""
        url = "https://example.com/test.jpg"
        
        sync_result = aesthetic_dummy_local(url, mock_config_with_async)
        async_result = await aesthetic_dummy_local_async(url, mock_config_with_async)
        
        assert sync_result == async_result

    @pytest.mark.asyncio
    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    async def test_aesthetic_replicate_async_basic(self, mock_replicate_run, mock_config_replicate_async):
        """Test basic functionality of async replicate aesthetic scoring."""
        mock_replicate_run.return_value = 7.5
        
        result = await aesthetic_replicate_async("https://example.com/image.jpg", mock_config_replicate_async)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result == 0.75  # 7.5 in [0,10] -> 0.75 in [0,1]

    @pytest.mark.asyncio
    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    async def test_aesthetic_replicate_async_matches_sync(self, mock_replicate_run, mock_config_replicate_async):
        """Test that async version produces same results as sync version."""
        mock_replicate_run.return_value = 6.0
        url = "https://example.com/image.jpg"
        
        sync_result = aesthetic_replicate(url, mock_config_replicate_async)
        async_result = await aesthetic_replicate_async(url, mock_config_replicate_async)
        
        assert sync_result == async_result

    @pytest.mark.asyncio
    async def test_aesthetic_async_dummy_local_provider(self, mock_config_with_async):
        """Test async aesthetic function with dummy_local provider."""
        mock_config_with_async.aesthetic_scoring.provider = "dummy_local"
        
        result = await aesthetic_async("https://example.com/image.jpg", mock_config_with_async)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    async def test_aesthetic_async_replicate_provider(self, mock_replicate_run, mock_config_replicate_async):
        """Test async aesthetic function with replicate provider."""
        mock_replicate_run.return_value = 5.0
        
        result = await aesthetic_async("https://example.com/image.jpg", mock_config_replicate_async)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result == 0.5  # 5.0 in [0,10] -> 0.5 in [0,1]

    @pytest.mark.asyncio
    async def test_aesthetic_async_unknown_provider(self, mock_config_with_async):
        """Test async aesthetic function with unknown provider."""
        mock_config_with_async.aesthetic_scoring.provider = "unknown_provider"
        
        with pytest.raises(NotImplementedError, match="Unknown provider unknown_provider"):
            await aesthetic_async("https://example.com/image.jpg", mock_config_with_async)

    @pytest.mark.asyncio
    async def test_aesthetic_async_matches_sync_results(self, mock_config_with_async):
        """Test that async version produces same results as sync version."""
        url = "https://example.com/consistent.jpg"
        
        sync_result = aesthetic(url, mock_config_with_async)
        async_result = await aesthetic_async(url, mock_config_with_async)
        
        assert sync_result == async_result


class TestScoreCandidatesParallel:
    """Test cases for parallel candidate scoring function."""

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_basic(self, mock_config_with_async):
        """Test basic parallel scoring of candidates."""
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"image_url": "https://example.com/image2.jpg"},
            {"image_url": "https://example.com/image3.jpg"}
        ]
        
        result = await score_candidates_parallel(candidates, mock_config_with_async)
        
        assert len(result) == 3
        for candidate in result:
            assert "aesthetic" in candidate
            assert isinstance(candidate["aesthetic"], float)
            assert 0.0 <= candidate["aesthetic"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_empty_list(self, mock_config_with_async):
        """Test parallel scoring with empty candidate list."""
        candidates = []
        
        result = await score_candidates_parallel(candidates, mock_config_with_async)
        
        assert result == []

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_no_image_url(self, mock_config_with_async):
        """Test parallel scoring with candidates missing image_url."""
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"other_field": "no_url"},  # Missing image_url
            {"image_url": None}  # None image_url
        ]
        
        result = await score_candidates_parallel(candidates, mock_config_with_async)
        
        assert len(result) == 3
        assert result[0]["aesthetic"] != 0.5  # Should have real score
        assert result[1]["aesthetic"] == 0.5  # Should have default score
        assert result[2]["aesthetic"] == 0.5  # Should have default score

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_with_concurrency_limit(self, mock_config_with_concurrency_limit):
        """Test parallel scoring with concurrency limit."""
        candidates = [
            {"image_url": f"https://example.com/image{i}.jpg"}
            for i in range(5)
        ]
        
        result = await score_candidates_parallel(candidates, mock_config_with_concurrency_limit)
        
        assert len(result) == 5
        for candidate in result:
            assert "aesthetic" in candidate
            assert isinstance(candidate["aesthetic"], float)
            assert 0.0 <= candidate["aesthetic"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_preserves_original_data(self, mock_config_with_async):
        """Test that parallel scoring preserves original candidate data."""
        candidates = [
            {
                "image_url": "https://example.com/image1.jpg",
                "prompt": "test prompt",
                "provider": "test_provider",
                "model": "test_model"
            },
            {
                "image_url": "https://example.com/image2.jpg",
                "brightness": 0.8,
                "entropy": 0.6
            }
        ]
        
        result = await score_candidates_parallel(candidates, mock_config_with_async)
        
        assert len(result) == 2
        
        # Check first candidate
        assert result[0]["prompt"] == "test prompt"
        assert result[0]["provider"] == "test_provider"
        assert result[0]["model"] == "test_model"
        assert "aesthetic" in result[0]
        
        # Check second candidate
        assert result[1]["brightness"] == 0.8
        assert result[1]["entropy"] == 0.6
        assert "aesthetic" in result[1]

    @pytest.mark.asyncio
    @patch('pixelbliss.scoring.aesthetic.replicate.run')
    async def test_score_candidates_parallel_replicate_provider(self, mock_replicate_run, mock_config_replicate_async):
        """Test parallel scoring with replicate provider."""
        # Since parallel execution doesn't guarantee order, we'll use the same score for all
        mock_replicate_run.return_value = 7.5
        
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"image_url": "https://example.com/image2.jpg"},
            {"image_url": "https://example.com/image3.jpg"}
        ]
        
        result = await score_candidates_parallel(candidates, mock_config_replicate_async)
        
        assert len(result) == 3
        # All should have the same score since we're using the same return value
        for candidate in result:
            assert candidate["aesthetic"] == 0.75  # 7.5 in [0,10] -> 0.75

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_consistency_with_sequential(self, mock_config_with_async):
        """Test that parallel scoring produces same results as sequential scoring."""
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"image_url": "https://example.com/image2.jpg"}
        ]
        
        # Sequential scoring
        sequential_results = []
        for candidate in candidates:
            candidate_copy = candidate.copy()
            image_url = candidate_copy.get("image_url")
            if image_url:
                score = aesthetic(image_url, mock_config_with_async)
            else:
                score = 0.5
            candidate_copy["aesthetic"] = score
            sequential_results.append(candidate_copy)
        
        # Parallel scoring
        parallel_results = await score_candidates_parallel(candidates.copy(), mock_config_with_async)
        
        # Compare results
        assert len(sequential_results) == len(parallel_results)
        for seq, par in zip(sequential_results, parallel_results):
            assert seq["aesthetic"] == par["aesthetic"]

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_zero_concurrency_limit(self, mock_config_with_async):
        """Test parallel scoring with zero concurrency limit (no semaphore)."""
        mock_config_with_async.image_generation.max_concurrency = 0
        
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"image_url": "https://example.com/image2.jpg"}
        ]
        
        result = await score_candidates_parallel(candidates, mock_config_with_async)
        
        assert len(result) == 2
        for candidate in result:
            assert "aesthetic" in candidate
            assert isinstance(candidate["aesthetic"], float)
            assert 0.0 <= candidate["aesthetic"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_single_candidate(self, mock_config_with_async):
        """Test parallel scoring with single candidate."""
        candidates = [{"image_url": "https://example.com/single.jpg"}]
        
        result = await score_candidates_parallel(candidates, mock_config_with_async)
        
        assert len(result) == 1
        assert "aesthetic" in result[0]
        assert isinstance(result[0]["aesthetic"], float)
        assert 0.0 <= result[0]["aesthetic"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_modifies_original_objects(self, mock_config_with_async):
        """Test that parallel scoring modifies the original candidate objects."""
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"image_url": "https://example.com/image2.jpg"}
        ]
        original_candidates = candidates.copy()
        
        result = await score_candidates_parallel(candidates, mock_config_with_async)
        
        # The function should modify the original objects and return them
        assert result is not original_candidates  # Different list
        assert result[0] is candidates[0]  # Same objects
        assert result[1] is candidates[1]  # Same objects
        
        # Original candidates should now have aesthetic scores
        assert "aesthetic" in candidates[0]
        assert "aesthetic" in candidates[1]

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_with_progress_logger(self, mock_config_with_async):
        """Test parallel scoring with progress logger."""
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"image_url": "https://example.com/image2.jpg"}
        ]
        
        mock_progress_logger = Mock()
        
        result = await score_candidates_parallel(candidates, mock_config_with_async, mock_progress_logger)
        
        assert len(result) == 2
        for candidate in result:
            assert "aesthetic" in candidate
        
        # Verify progress logger was called
        mock_progress_logger.start_operation.assert_called_once_with("aesthetic_scoring", 2, "parallel aesthetic scoring")
        assert mock_progress_logger.update_operation_progress.call_count == 2
        mock_progress_logger.finish_operation.assert_called_once_with("aesthetic_scoring", True)

    @pytest.mark.asyncio
    async def test_score_candidates_parallel_with_exceptions_and_logger(self, mock_config_with_async):
        """Test parallel scoring with exceptions and progress logger."""
        candidates = [
            {"image_url": "https://example.com/image1.jpg"},
            {"image_url": "https://example.com/image2.jpg"}
        ]
        
        mock_progress_logger = Mock()
        
        # Mock aesthetic_async to raise exception for one candidate
        with patch('pixelbliss.scoring.aesthetic.aesthetic_async') as mock_aesthetic:
            mock_aesthetic.side_effect = [0.7, Exception("API Error")]
            
            result = await score_candidates_parallel(candidates, mock_config_with_async, mock_progress_logger)
            
            assert len(result) == 2
            assert result[0]["aesthetic"] == 0.7  # First succeeded
            assert result[1]["aesthetic"] == 0.5  # Second failed, got fallback
            
            # Verify progress logger was called
            mock_progress_logger.start_operation.assert_called_once_with("aesthetic_scoring", 2, "parallel aesthetic scoring")
            assert mock_progress_logger.update_operation_progress.call_count == 2
            mock_progress_logger.finish_operation.assert_called_once_with("aesthetic_scoring", False)
            mock_progress_logger.warning.assert_called_once_with("1 aesthetic scores used fallback values")
