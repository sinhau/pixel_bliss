import pytest
import json
import os
from unittest.mock import Mock, patch
from pixelbliss.storage.manifest import _load_manifest, _save_manifest, append, update_tweet_id, load_recent_hashes


class TestManifest:
    @patch('pixelbliss.storage.manifest.os.path.exists')
    @patch('builtins.open')
    @patch('pixelbliss.storage.manifest.json.load')
    def test_load_manifest_exists(self, mock_json_load, mock_open, mock_exists):
        mock_exists.return_value = True
        mock_json_load.return_value = [{"id": "test"}]
        result = _load_manifest()
        assert result == [{"id": "test"}]

    @patch('pixelbliss.storage.manifest.os.path.exists')
    def test_load_manifest_not_exists(self, mock_exists):
        mock_exists.return_value = False
        result = _load_manifest()
        assert result == []

    @patch('pixelbliss.storage.manifest.Path')
    @patch('builtins.open')
    @patch('pixelbliss.storage.manifest.json.dump')
    def test_save_manifest(self, mock_json_dump, mock_open, mock_path):
        data = [{"id": "test"}]
        _save_manifest(data)
        mock_json_dump.assert_called_once_with(data, mock_open().__enter__(), indent=2)

    @patch('pixelbliss.storage.manifest._load_manifest')
    @patch('pixelbliss.storage.manifest._save_manifest')
    def test_append(self, mock_save, mock_load):
        mock_load.return_value = [{"id": "existing"}]
        entry = {"id": "new"}
        append(entry)
        mock_save.assert_called_once_with([{"id": "existing"}, {"id": "new"}])

    @patch('pixelbliss.storage.manifest._load_manifest')
    @patch('pixelbliss.storage.manifest._save_manifest')
    def test_update_tweet_id(self, mock_save, mock_load):
        mock_load.return_value = [{"id": "test_id", "tweet_id": "old"}]
        update_tweet_id("test_id", "new_tweet_id")
        mock_save.assert_called_once_with([{"id": "test_id", "tweet_id": "new_tweet_id"}])

    @patch('pixelbliss.storage.manifest._load_manifest')
    def test_load_recent_hashes(self, mock_load):
        mock_load.return_value = [
            {"phash": "hash1"},
            {"phash": "hash2"},
            {"phash": None},
            {"phash": "hash3"}
        ]
        result = load_recent_hashes(4)
        assert result == ["hash1", "hash2", "hash3"]

    @patch('pixelbliss.storage.manifest._load_manifest')
    def test_load_recent_hashes_limit(self, mock_load):
        mock_load.return_value = [{"phash": f"hash{i}"} for i in range(5)]
        result = load_recent_hashes(3)
        assert result == ["hash2", "hash3", "hash4"]
