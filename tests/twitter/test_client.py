import os
import pytest
from unittest.mock import patch, Mock, MagicMock
from pixelbliss.twitter.client import (
    get_client,
    upload_media,
    set_alt_text,
    create_tweet
)


class TestGetClient:
    """Test cases for get_client function."""

    @patch.dict(os.environ, {
        'X_API_KEY': 'test_api_key',
        'X_API_SECRET': 'test_api_secret',
        'X_ACCESS_TOKEN': 'test_access_token',
        'X_ACCESS_TOKEN_SECRET': 'test_access_token_secret'
    })
    @patch('pixelbliss.twitter.client.tweepy.Client')
    def test_get_client_with_credentials(self, mock_client):
        """Test creating client with all credentials set."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        result = get_client()
        
        mock_client.assert_called_once_with(
            consumer_key='test_api_key',
            consumer_secret='test_api_secret',
            access_token='test_access_token',
            access_token_secret='test_access_token_secret'
        )
        assert result == mock_client_instance

    @patch.dict(os.environ, {}, clear=True)
    @patch('pixelbliss.twitter.client.tweepy.Client')
    def test_get_client_without_credentials(self, mock_client):
        """Test creating client without credentials (should pass None values)."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        result = get_client()
        
        mock_client.assert_called_once_with(
            consumer_key=None,
            consumer_secret=None,
            access_token=None,
            access_token_secret=None
        )
        assert result == mock_client_instance

    @patch.dict(os.environ, {
        'X_API_KEY': 'partial_key',
        'X_API_SECRET': '',  # Empty string
        'X_ACCESS_TOKEN': 'partial_token'
        # Missing X_ACCESS_TOKEN_SECRET
    })
    @patch('pixelbliss.twitter.client.tweepy.Client')
    def test_get_client_with_partial_credentials(self, mock_client):
        """Test creating client with partial credentials."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        result = get_client()
        
        mock_client.assert_called_once_with(
            consumer_key='partial_key',
            consumer_secret='',
            access_token='partial_token',
            access_token_secret=None
        )
        assert result == mock_client_instance


class TestUploadMedia:
    """Test cases for upload_media function."""

    @patch.dict(os.environ, {
        'X_API_KEY': 'test_api_key',
        'X_API_SECRET': 'test_api_secret',
        'X_ACCESS_TOKEN': 'test_access_token',
        'X_ACCESS_TOKEN_SECRET': 'test_access_token_secret'
    })
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_single_file(self, mock_api_class, mock_oauth):
        """Test uploading a single media file."""
        # Mock OAuth handler
        mock_auth = Mock()
        mock_oauth.return_value = mock_auth
        
        # Mock API and media upload
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        mock_media = Mock()
        mock_media.media_id_string = "123456789"
        mock_api.media_upload.return_value = mock_media
        
        result = upload_media(["/path/to/image.jpg"])
        
        # Verify OAuth setup
        mock_oauth.assert_called_once_with(
            consumer_key='test_api_key',
            consumer_secret='test_api_secret',
            access_token='test_access_token',
            access_token_secret='test_access_token_secret'
        )
        
        # Verify API setup and upload
        mock_api_class.assert_called_once_with(mock_auth)
        mock_api.media_upload.assert_called_once_with("/path/to/image.jpg")
        
        assert result == ["123456789"]

    @patch.dict(os.environ, {
        'X_API_KEY': 'test_api_key',
        'X_API_SECRET': 'test_api_secret',
        'X_ACCESS_TOKEN': 'test_access_token',
        'X_ACCESS_TOKEN_SECRET': 'test_access_token_secret'
    })
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_multiple_files(self, mock_api_class, mock_oauth):
        """Test uploading multiple media files."""
        # Mock OAuth handler
        mock_auth = Mock()
        mock_oauth.return_value = mock_auth
        
        # Mock API
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        # Mock multiple media uploads
        mock_media1 = Mock()
        mock_media1.media_id_string = "123456789"
        mock_media2 = Mock()
        mock_media2.media_id_string = "987654321"
        mock_media3 = Mock()
        mock_media3.media_id_string = "555666777"
        
        mock_api.media_upload.side_effect = [mock_media1, mock_media2, mock_media3]
        
        paths = ["/path/to/image1.jpg", "/path/to/image2.png", "/path/to/image3.gif"]
        result = upload_media(paths)
        
        # Verify all files were uploaded
        assert mock_api.media_upload.call_count == 3
        mock_api.media_upload.assert_any_call("/path/to/image1.jpg")
        mock_api.media_upload.assert_any_call("/path/to/image2.png")
        mock_api.media_upload.assert_any_call("/path/to/image3.gif")
        
        assert result == ["123456789", "987654321", "555666777"]

    @patch.dict(os.environ, {
        'X_API_KEY': 'test_api_key',
        'X_API_SECRET': 'test_api_secret',
        'X_ACCESS_TOKEN': 'test_access_token',
        'X_ACCESS_TOKEN_SECRET': 'test_access_token_secret'
    })
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_empty_list(self, mock_api_class, mock_oauth):
        """Test uploading with empty file list."""
        # Mock OAuth handler
        mock_auth = Mock()
        mock_oauth.return_value = mock_auth
        
        # Mock API
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        result = upload_media([])
        
        # Should not call media_upload for empty list
        mock_api.media_upload.assert_not_called()
        assert result == []

    @patch.dict(os.environ, {}, clear=True)
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_without_credentials(self, mock_api_class, mock_oauth):
        """Test uploading media without credentials."""
        # Mock OAuth handler
        mock_auth = Mock()
        mock_oauth.return_value = mock_auth
        
        # Mock API
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        mock_media = Mock()
        mock_media.media_id_string = "123456789"
        mock_api.media_upload.return_value = mock_media
        
        result = upload_media(["/path/to/image.jpg"])
        
        # Verify OAuth setup with None values
        mock_oauth.assert_called_once_with(
            consumer_key=None,
            consumer_secret=None,
            access_token=None,
            access_token_secret=None
        )
        
        assert result == ["123456789"]


class TestSetAltText:
    """Test cases for set_alt_text function."""

    @patch('pixelbliss.twitter.client.get_client')
    def test_set_alt_text_success(self, mock_get_client):
        """Test setting alt text successfully."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        set_alt_text("123456789", "A beautiful sunset over mountains")
        
        mock_get_client.assert_called_once()
        mock_client.create_media_metadata.assert_called_once_with(
            "123456789",
            alt_text={"text": "A beautiful sunset over mountains"}
        )

    @patch('pixelbliss.twitter.client.get_client')
    def test_set_alt_text_empty_string(self, mock_get_client):
        """Test setting empty alt text."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        set_alt_text("123456789", "")
        
        mock_client.create_media_metadata.assert_called_once_with(
            "123456789",
            alt_text={"text": ""}
        )

    @patch('pixelbliss.twitter.client.get_client')
    def test_set_alt_text_long_description(self, mock_get_client):
        """Test setting long alt text description."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        long_alt = "A very detailed description of an image that contains multiple elements including people, buildings, nature, and various objects that would be important for accessibility purposes."
        
        set_alt_text("987654321", long_alt)
        
        mock_client.create_media_metadata.assert_called_once_with(
            "987654321",
            alt_text={"text": long_alt}
        )


class TestCreateTweet:
    """Test cases for create_tweet function."""

    @patch('pixelbliss.twitter.client.get_client')
    def test_create_tweet_with_media(self, mock_get_client):
        """Test creating tweet with media attachments."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = {'id': 'tweet_123456789'}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = create_tweet("Check out this amazing image!", ["123456789", "987654321"])
        
        mock_get_client.assert_called_once()
        mock_client.create_tweet.assert_called_once_with(
            text="Check out this amazing image!",
            media_ids=["123456789", "987654321"]
        )
        assert result == 'tweet_123456789'

    @patch('pixelbliss.twitter.client.get_client')
    def test_create_tweet_with_single_media(self, mock_get_client):
        """Test creating tweet with single media attachment."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = {'id': 'tweet_987654321'}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = create_tweet("Single image tweet", ["123456789"])
        
        mock_client.create_tweet.assert_called_once_with(
            text="Single image tweet",
            media_ids=["123456789"]
        )
        assert result == 'tweet_987654321'

    @patch('pixelbliss.twitter.client.get_client')
    def test_create_tweet_empty_media_list(self, mock_get_client):
        """Test creating tweet with empty media list."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = {'id': 'tweet_empty_media'}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = create_tweet("Text only tweet", [])
        
        mock_client.create_tweet.assert_called_once_with(
            text="Text only tweet",
            media_ids=[]
        )
        assert result == 'tweet_empty_media'

    @patch('pixelbliss.twitter.client.get_client')
    def test_create_tweet_long_text(self, mock_get_client):
        """Test creating tweet with long text."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = {'id': 'tweet_long_text'}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        long_text = "This is a very long tweet that might exceed normal character limits but we're testing to make sure our function handles it properly and passes it through to the Twitter API correctly."
        
        result = create_tweet(long_text, ["123456789"])
        
        mock_client.create_tweet.assert_called_once_with(
            text=long_text,
            media_ids=["123456789"]
        )
        assert result == 'tweet_long_text'

    @patch('pixelbliss.twitter.client.get_client')
    def test_create_tweet_special_characters(self, mock_get_client):
        """Test creating tweet with special characters and emojis."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = {'id': 'tweet_special_chars'}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        special_text = "Amazing AI art! 🎨✨ #AIArt #DigitalArt @pixelbliss"
        
        result = create_tweet(special_text, ["123456789"])
        
        mock_client.create_tweet.assert_called_once_with(
            text=special_text,
            media_ids=["123456789"]
        )
        assert result == 'tweet_special_chars'
