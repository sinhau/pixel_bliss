import os
import pytest
from unittest.mock import patch, Mock
from pixelbliss.alerts.webhook import send_success, send_failure
from pixelbliss.config import Config


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock(spec=Config)
    config.alerts = Mock()
    config.alerts.enabled = True
    config.alerts.webhook_url_env = "TEST_WEBHOOK_URL"
    return config


@pytest.fixture
def disabled_config():
    """Create a mock config with alerts disabled."""
    config = Mock(spec=Config)
    config.alerts = Mock()
    config.alerts.enabled = False
    config.alerts.webhook_url_env = "TEST_WEBHOOK_URL"
    return config


class TestSendSuccess:
    """Test cases for send_success function."""

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {'TEST_WEBHOOK_URL': 'https://hooks.slack.com/test'})
    def test_send_success_with_enabled_alerts(self, mock_post, mock_config):
        """Test sending success notification when alerts are enabled."""
        send_success(
            category="nature",
            model="flux-dev",
            tweet_url="https://twitter.com/test/123",
            image_url="https://example.com/image.jpg",
            cfg=mock_config
        )
        
        mock_post.assert_called_once_with(
            'https://hooks.slack.com/test',
            json={
                "content": "[PixelBliss] Posted nature via flux-dev → https://twitter.com/test/123\nImage: https://example.com/image.jpg"
            }
        )

    @patch('pixelbliss.alerts.webhook.requests.post')
    def test_send_success_with_disabled_alerts(self, mock_post, disabled_config):
        """Test that no webhook is sent when alerts are disabled."""
        send_success(
            category="nature",
            model="flux-dev",
            tweet_url="https://twitter.com/test/123",
            image_url="https://example.com/image.jpg",
            cfg=disabled_config
        )
        
        mock_post.assert_not_called()

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {}, clear=True)
    def test_send_success_without_webhook_url(self, mock_post, mock_config):
        """Test that no webhook is sent when webhook URL is not set."""
        send_success(
            category="nature",
            model="flux-dev",
            tweet_url="https://twitter.com/test/123",
            image_url="https://example.com/image.jpg",
            cfg=mock_config
        )
        
        mock_post.assert_not_called()

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {'TEST_WEBHOOK_URL': ''})
    def test_send_success_with_empty_webhook_url(self, mock_post, mock_config):
        """Test that no webhook is sent when webhook URL is empty."""
        send_success(
            category="nature",
            model="flux-dev",
            tweet_url="https://twitter.com/test/123",
            image_url="https://example.com/image.jpg",
            cfg=mock_config
        )
        
        mock_post.assert_not_called()


class TestSendFailure:
    """Test cases for send_failure function."""

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {'TEST_WEBHOOK_URL': 'https://hooks.slack.com/test'})
    def test_send_failure_with_enabled_alerts(self, mock_post, mock_config):
        """Test sending failure notification when alerts are enabled."""
        send_failure(
            reason="API rate limit exceeded",
            cfg=mock_config
        )
        
        mock_post.assert_called_once_with(
            'https://hooks.slack.com/test',
            json={
                "content": "[PixelBliss] FAIL: API rate limit exceeded"
            }
        )

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {'TEST_WEBHOOK_URL': 'https://hooks.slack.com/test'})
    def test_send_failure_with_details(self, mock_post, mock_config):
        """Test sending failure notification with additional details."""
        send_failure(
            reason="API rate limit exceeded",
            cfg=mock_config,
            details="Rate limit: 100 requests per hour. Try again in 30 minutes."
        )
        
        mock_post.assert_called_once_with(
            'https://hooks.slack.com/test',
            json={
                "content": "[PixelBliss] FAIL: API rate limit exceeded\nRate limit: 100 requests per hour. Try again in 30 minutes."
            }
        )

    @patch('pixelbliss.alerts.webhook.requests.post')
    def test_send_failure_with_disabled_alerts(self, mock_post, disabled_config):
        """Test that no webhook is sent when alerts are disabled."""
        send_failure(
            reason="API rate limit exceeded",
            cfg=disabled_config
        )
        
        mock_post.assert_not_called()

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {}, clear=True)
    def test_send_failure_without_webhook_url(self, mock_post, mock_config):
        """Test that no webhook is sent when webhook URL is not set."""
        send_failure(
            reason="API rate limit exceeded",
            cfg=mock_config
        )
        
        mock_post.assert_not_called()

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {'TEST_WEBHOOK_URL': ''})
    def test_send_failure_with_empty_webhook_url(self, mock_post, mock_config):
        """Test that no webhook is sent when webhook URL is empty."""
        send_failure(
            reason="API rate limit exceeded",
            cfg=mock_config
        )
        
        mock_post.assert_not_called()

    @patch('pixelbliss.alerts.webhook.requests.post')
    @patch.dict(os.environ, {'TEST_WEBHOOK_URL': 'https://hooks.slack.com/test'})
    def test_send_failure_with_empty_details(self, mock_post, mock_config):
        """Test sending failure notification with empty details string."""
        send_failure(
            reason="API rate limit exceeded",
            cfg=mock_config,
            details=""
        )
        
        mock_post.assert_called_once_with(
            'https://hooks.slack.com/test',
            json={
                "content": "[PixelBliss] FAIL: API rate limit exceeded"
            }
        )
