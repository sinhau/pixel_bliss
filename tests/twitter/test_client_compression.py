"""
Tests for Twitter client compression integration.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from pixelbliss.twitter.client import upload_media


class TestUploadMediaWithCompression:
    """Test media upload with automatic compression."""
    
    @patch('pixelbliss.twitter.client.prepare_for_twitter_upload')
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_calls_compression(self, mock_api_class, mock_auth_class, mock_prepare):
        """Test that upload_media calls compression preparation."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_auth = Mock()
        mock_auth_class.return_value = mock_auth
        
        # Mock media upload response
        mock_media = Mock()
        mock_media.media_id_string = "12345"
        mock_api.media_upload.return_value = mock_media
        
        # Mock compression preparation
        test_paths = ['/path/to/image1.jpg', '/path/to/image2.png']
        processed_paths = ['/path/to/compressed1.jpg', '/path/to/compressed2.jpg']
        mock_prepare.return_value = processed_paths
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'X_API_KEY': 'test_key',
            'X_API_SECRET': 'test_secret',
            'X_ACCESS_TOKEN': 'test_token',
            'X_ACCESS_TOKEN_SECRET': 'test_token_secret'
        }):
            # Call upload_media
            result = upload_media(test_paths)
        
        # Verify compression was called
        mock_prepare.assert_called_once_with(test_paths)
        
        # Verify API calls were made with processed paths
        assert mock_api.media_upload.call_count == 2
        mock_api.media_upload.assert_any_call('/path/to/compressed1.jpg')
        mock_api.media_upload.assert_any_call('/path/to/compressed2.jpg')
        
        # Verify return value
        assert result == ["12345", "12345"]
    
    @patch('pixelbliss.twitter.client.prepare_for_twitter_upload')
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_single_file(self, mock_api_class, mock_auth_class, mock_prepare):
        """Test upload_media with single file."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_auth = Mock()
        mock_auth_class.return_value = mock_auth
        
        # Mock media upload response
        mock_media = Mock()
        mock_media.media_id_string = "67890"
        mock_api.media_upload.return_value = mock_media
        
        # Mock compression preparation (no change needed)
        test_path = ['/path/to/image.jpg']
        mock_prepare.return_value = test_path
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'X_API_KEY': 'test_key',
            'X_API_SECRET': 'test_secret',
            'X_ACCESS_TOKEN': 'test_token',
            'X_ACCESS_TOKEN_SECRET': 'test_token_secret'
        }):
            # Call upload_media
            result = upload_media(test_path)
        
        # Verify compression was called
        mock_prepare.assert_called_once_with(test_path)
        
        # Verify API call
        mock_api.media_upload.assert_called_once_with('/path/to/image.jpg')
        
        # Verify return value
        assert result == ["67890"]
    
    @patch('pixelbliss.twitter.client.prepare_for_twitter_upload')
    @patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler')
    @patch('pixelbliss.twitter.client.tweepy.API')
    def test_upload_media_empty_list(self, mock_api_class, mock_auth_class, mock_prepare):
        """Test upload_media with empty file list."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_auth = Mock()
        mock_auth_class.return_value = mock_auth
        
        # Mock compression preparation
        mock_prepare.return_value = []
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'X_API_KEY': 'test_key',
            'X_API_SECRET': 'test_secret',
            'X_ACCESS_TOKEN': 'test_token',
            'X_ACCESS_TOKEN_SECRET': 'test_token_secret'
        }):
            # Call upload_media
            result = upload_media([])
        
        # Verify compression was called
        mock_prepare.assert_called_once_with([])
        
        # Verify no API calls were made
        mock_api.media_upload.assert_not_called()
        
        # Verify return value
        assert result == []
    
    @patch('pixelbliss.twitter.client.prepare_for_twitter_upload')
    def test_upload_media_compression_integration(self, mock_prepare):
        """Test that compression is properly integrated into upload workflow."""
        # Create a real image file for testing
        image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name, 'PNG')
            
            try:
                # Mock the compression to return the same file (no compression needed)
                mock_prepare.return_value = [tmp.name]
                
                # Mock the Twitter API parts
                with patch('pixelbliss.twitter.client.tweepy.OAuth1UserHandler') as mock_auth_class:
                    with patch('pixelbliss.twitter.client.tweepy.API') as mock_api_class:
                        mock_api = Mock()
                        mock_api_class.return_value = mock_api
                        mock_auth = Mock()
                        mock_auth_class.return_value = mock_auth
                        
                        # Mock media upload response
                        mock_media = Mock()
                        mock_media.media_id_string = "test123"
                        mock_api.media_upload.return_value = mock_media
                        
                        # Mock environment variables
                        with patch.dict(os.environ, {
                            'X_API_KEY': 'test_key',
                            'X_API_SECRET': 'test_secret',
                            'X_ACCESS_TOKEN': 'test_token',
                            'X_ACCESS_TOKEN_SECRET': 'test_token_secret'
                        }):
                            # Call upload_media
                            result = upload_media([tmp.name])
                        
                        # Verify compression was called with original path
                        mock_prepare.assert_called_once_with([tmp.name])
                        
                        # Verify API was called with processed path
                        mock_api.media_upload.assert_called_once_with(tmp.name)
                        
                        # Verify result
                        assert result == ["test123"]
            
            finally:
                os.unlink(tmp.name)


class TestTwitterClientImports:
    """Test that imports work correctly."""
    
    def test_compression_import(self):
        """Test that compression module is properly imported."""
        from pixelbliss.twitter.client import prepare_for_twitter_upload
        
        # Should be able to import the function
        assert callable(prepare_for_twitter_upload)
    
    def test_all_functions_available(self):
        """Test that all client functions are available."""
        from pixelbliss.twitter import client
        
        # Check that all expected functions exist
        assert hasattr(client, 'get_client')
        assert hasattr(client, 'upload_media')
        assert hasattr(client, 'set_alt_text')
        assert hasattr(client, 'create_tweet')
        
        # Check that they are callable
        assert callable(client.get_client)
        assert callable(client.upload_media)
        assert callable(client.set_alt_text)
        assert callable(client.create_tweet)


class TestUploadMediaDocumentation:
    """Test that the updated documentation is accurate."""
    
    def test_upload_media_docstring(self):
        """Test that upload_media docstring mentions compression."""
        from pixelbliss.twitter.client import upload_media
        
        docstring = upload_media.__doc__
        assert docstring is not None
        assert "compress" in docstring.lower() or "5mb" in docstring.lower()
        assert "quality" in docstring.lower()
