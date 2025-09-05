import os
import pytest
from unittest.mock import patch, Mock, MagicMock
from pixelbliss.twitter.client import (
    get_client,
    upload_media,
    set_alt_text,
    create_tweet,
    get_user_info,
    get_tweet_info
)


class TestGetClient:
    """Test cases for get_client function."""

    @patch.dict(os.environ, {
        'X_BEARER_TOKEN': 'test_bearer_token',
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
            bearer_token='test_bearer_token',
            consumer_key='test_api_key',
            consumer_secret='test_api_secret',
            access_token='test_access_token',
            access_token_secret='test_access_token_secret',
            wait_on_rate_limit=True
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
            bearer_token=None,
            consumer_key=None,
            consumer_secret=None,
            access_token=None,
            access_token_secret=None,
            wait_on_rate_limit=True
        )
        assert result == mock_client_instance

    @patch.dict(os.environ, {
        'X_BEARER_TOKEN': 'partial_bearer',
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
            bearer_token='partial_bearer',
            consumer_key='partial_key',
            consumer_secret='',
            access_token='partial_token',
            access_token_secret=None,
            wait_on_rate_limit=True
        )
        assert result == mock_client_instance


class TestUploadMedia:
    """Test cases for upload_media function."""

    @patch('pixelbliss.twitter.client.get_client')
    def test_upload_media_single_file_v2_success(self, mock_get_client):
        """Test uploading a single media file using v2 API successfully."""
        # Mock client and media upload
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_media = Mock()
        mock_media.media_id = 123456789
        mock_client.media_upload.return_value = mock_media
        
        result = upload_media(["/path/to/image.jpg"])
        
        mock_get_client.assert_called_once()
        mock_client.media_upload.assert_called_once_with(filename="/path/to/image.jpg")
        
        assert result == ["123456789"]

    @patch('pixelbliss.twitter.client.get_client')
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_single_file_v2_fallback(self, mock_api_class, mock_oauth, mock_get_client):
        """Test uploading a single media file with v2 failure and v1.1 fallback."""
        # Mock v2 client failure
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.media_upload.side_effect = Exception("v2 upload failed")
        
        # Mock v1.1 fallback
        mock_auth = Mock()
        mock_oauth.return_value = mock_auth
        
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        mock_media = Mock()
        mock_media.media_id_string = "123456789"
        mock_api.media_upload.return_value = mock_media
        
        result = upload_media(["/path/to/image.jpg"])
        
        # Verify v2 was tried first
        mock_get_client.assert_called_once()
        mock_client.media_upload.assert_called_once_with(filename="/path/to/image.jpg")
        
        # Verify v1.1 fallback was used
        mock_oauth.assert_called_once()
        mock_api_class.assert_called_once_with(mock_auth)
        mock_api.media_upload.assert_called_once_with("/path/to/image.jpg")
        
        assert result == ["123456789"]

    @patch('pixelbliss.twitter.client.get_client')
    def test_upload_media_multiple_files(self, mock_get_client):
        """Test uploading multiple media files."""
        # Mock client
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Mock multiple media uploads
        mock_media1 = Mock()
        mock_media1.media_id = 123456789
        mock_media2 = Mock()
        mock_media2.media_id = 987654321
        mock_media3 = Mock()
        mock_media3.media_id = 555666777
        
        mock_client.media_upload.side_effect = [mock_media1, mock_media2, mock_media3]
        
        paths = ["/path/to/image1.jpg", "/path/to/image2.png", "/path/to/image3.gif"]
        result = upload_media(paths)
        
        # Verify all files were uploaded
        assert mock_client.media_upload.call_count == 3
        mock_client.media_upload.assert_any_call(filename="/path/to/image1.jpg")
        mock_client.media_upload.assert_any_call(filename="/path/to/image2.png")
        mock_client.media_upload.assert_any_call(filename="/path/to/image3.gif")
        
        assert result == ["123456789", "987654321", "555666777"]

    @patch('pixelbliss.twitter.client.get_client')
    def test_upload_media_empty_list(self, mock_get_client):
        """Test uploading with empty file list."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        result = upload_media([])
        
        # Should not call media_upload for empty list
        mock_client.media_upload.assert_not_called()
        assert result == []


class TestSetAltText:
    """Test cases for set_alt_text function."""

    @patch.dict(os.environ, {
        'X_API_KEY': 'test_api_key',
        'X_API_SECRET': 'test_api_secret',
        'X_ACCESS_TOKEN': 'test_access_token',
        'X_ACCESS_TOKEN_SECRET': 'test_access_token_secret'
    })
    @patch('pixelbliss.twitter.client.get_client')
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_set_alt_text_success(self, mock_api_class, mock_oauth, mock_get_client):
        """Test setting alt text successfully using v1.1 fallback."""
        # Mock v2 client (will fail and fallback to v1.1)
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Mock v1.1 fallback
        mock_auth = Mock()
        mock_oauth.return_value = mock_auth
        
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        set_alt_text("123456789", "A beautiful sunset over mountains")
        
        # Verify v1.1 fallback was used
        mock_oauth.assert_called_once_with(
            consumer_key='test_api_key',
            consumer_secret='test_api_secret',
            access_token='test_access_token',
            access_token_secret='test_access_token_secret'
        )
        
        mock_api_class.assert_called_once_with(mock_auth)
        mock_api.create_media_metadata.assert_called_once_with(
            "123456789",
            alt_text={"text": "A beautiful sunset over mountains"}
        )

    @patch.dict(os.environ, {
        'X_API_KEY': 'test_api_key',
        'X_API_SECRET': 'test_api_secret',
        'X_ACCESS_TOKEN': 'test_access_token',
        'X_ACCESS_TOKEN_SECRET': 'test_access_token_secret'
    })
    @patch('pixelbliss.twitter.client.get_client')
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_set_alt_text_empty_string(self, mock_api_class, mock_oauth, mock_get_client):
        """Test setting empty alt text."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_auth = Mock()
        mock_oauth.return_value = mock_auth
        
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        set_alt_text("123456789", "")
        
        mock_api.create_media_metadata.assert_called_once_with(
            "123456789",
            alt_text={"text": ""}
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
            media_ids=[123456789, 987654321]
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
            media_ids=[123456789]
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
            media_ids=None
        )
        assert result == 'tweet_empty_media'

    @patch('pixelbliss.twitter.client.get_client')
    def test_create_tweet_alternative_response_format(self, mock_get_client):
        """Test creating tweet with alternative response format."""
        mock_client = Mock()
        mock_response = Mock()
        # Simulate response without .data attribute but with .id
        mock_response.data = None
        mock_response.id = 'tweet_alt_format'
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = create_tweet("Alternative format tweet", ["123456789"])
        
        mock_client.create_tweet.assert_called_once_with(
            text="Alternative format tweet",
            media_ids=[123456789]
        )
        assert result == 'tweet_alt_format'


class TestGetUserInfo:
    """Test cases for get_user_info function."""

    @patch('pixelbliss.twitter.client.get_client')
    def test_get_user_info_by_username(self, mock_get_client):
        """Test getting user info by username."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_user_data = Mock()
        mock_user_data.id = '123456789'
        mock_user_data.name = 'Test User'
        mock_user_data.username = 'testuser'
        mock_user_data.public_metrics = {'followers_count': 100, 'following_count': 50}
        
        mock_response = Mock()
        mock_response.data = mock_user_data
        mock_client.get_user.return_value = mock_response
        
        result = get_user_info('testuser')
        
        mock_client.get_user.assert_called_once_with(
            username='testuser',
            user_fields=['id', 'name', 'username', 'public_metrics']
        )
        
        expected = {
            'id': '123456789',
            'name': 'Test User',
            'username': 'testuser',
            'public_metrics': {'followers_count': 100, 'following_count': 50}
        }
        assert result == expected

    @patch('pixelbliss.twitter.client.get_client')
    def test_get_user_info_authenticated_user(self, mock_get_client):
        """Test getting authenticated user info."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_user_data = Mock()
        mock_user_data.id = '987654321'
        mock_user_data.name = 'Auth User'
        mock_user_data.username = 'authuser'
        mock_user_data.public_metrics = {'followers_count': 200, 'following_count': 75}
        
        mock_response = Mock()
        mock_response.data = mock_user_data
        mock_client.get_me.return_value = mock_response
        
        result = get_user_info()
        
        mock_client.get_me.assert_called_once_with(
            user_fields=['id', 'name', 'username', 'public_metrics']
        )
        
        expected = {
            'id': '987654321',
            'name': 'Auth User',
            'username': 'authuser',
            'public_metrics': {'followers_count': 200, 'following_count': 75}
        }
        assert result == expected

    @patch('pixelbliss.twitter.client.get_client')
    def test_get_user_info_user_not_found(self, mock_get_client):
        """Test getting user info when user is not found."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = None
        mock_client.get_user.return_value = mock_response
        
        with pytest.raises(Exception, match="User not found: nonexistentuser"):
            get_user_info('nonexistentuser')


class TestGetTweetInfo:
    """Test cases for get_tweet_info function."""

    @patch('pixelbliss.twitter.client.get_client')
    def test_get_tweet_info_success(self, mock_get_client):
        """Test getting tweet info successfully."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_tweet_data = Mock()
        mock_tweet_data.id = '123456789'
        mock_tweet_data.text = 'This is a test tweet'
        mock_tweet_data.created_at = '2023-01-01T12:00:00Z'
        mock_tweet_data.public_metrics = {'retweet_count': 5, 'like_count': 10}
        mock_tweet_data.author_id = '987654321'
        
        mock_author = Mock()
        mock_author.id = '987654321'
        mock_author.username = 'testuser'
        mock_author.name = 'Test User'
        
        mock_response = Mock()
        mock_response.data = mock_tweet_data
        mock_response.includes = {'users': [mock_author]}
        mock_client.get_tweet.return_value = mock_response
        
        result = get_tweet_info('123456789')
        
        mock_client.get_tweet.assert_called_once_with(
            '123456789',
            tweet_fields=['created_at', 'public_metrics', 'author_id', 'text'],
            expansions=['author_id'],
            user_fields=['username', 'name']
        )
        
        expected = {
            'id': '123456789',
            'text': 'This is a test tweet',
            'created_at': '2023-01-01T12:00:00Z',
            'public_metrics': {'retweet_count': 5, 'like_count': 10},
            'author_id': '987654321',
            'author': {
                'id': '987654321',
                'username': 'testuser',
                'name': 'Test User'
            }
        }
        assert result == expected

    @patch('pixelbliss.twitter.client.get_client')
    def test_get_tweet_info_without_author(self, mock_get_client):
        """Test getting tweet info without author expansion."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_tweet_data = Mock()
        mock_tweet_data.id = '123456789'
        mock_tweet_data.text = 'This is a test tweet'
        mock_tweet_data.created_at = '2023-01-01T12:00:00Z'
        mock_tweet_data.public_metrics = {'retweet_count': 5, 'like_count': 10}
        mock_tweet_data.author_id = '987654321'
        
        mock_response = Mock()
        mock_response.data = mock_tweet_data
        mock_response.includes = None
        mock_client.get_tweet.return_value = mock_response
        
        result = get_tweet_info('123456789')
        
        expected = {
            'id': '123456789',
            'text': 'This is a test tweet',
            'created_at': '2023-01-01T12:00:00Z',
            'public_metrics': {'retweet_count': 5, 'like_count': 10},
            'author_id': '987654321'
        }
        assert result == expected

    @patch('pixelbliss.twitter.client.get_client')
    def test_get_tweet_info_not_found(self, mock_get_client):
        """Test getting tweet info when tweet is not found."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = None
        mock_client.get_tweet.return_value = mock_response
        
        with pytest.raises(Exception, match="Tweet not found: nonexistenttweet"):
            get_tweet_info('nonexistenttweet')
