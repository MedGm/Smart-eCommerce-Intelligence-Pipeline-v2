import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch


def test_is_minio_configured_false_when_no_env(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    from src.storage.minio_client import is_minio_configured
    assert not is_minio_configured()


def test_is_minio_configured_true_when_env_set(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    from src.storage.minio_client import is_minio_configured
    assert is_minio_configured()


def test_upload_file_calls_s3_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    local_file = tmp_path / "test.json"
    local_file.write_text('{"x": 1}')

    mock_client = MagicMock()
    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import upload_file
        upload_file(local_file, bucket="raw-data", key="raw/test.json")

    mock_client.upload_file.assert_called_once_with(
        str(local_file), "raw-data", "raw/test.json"
    )


def test_upload_file_noop_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    local_file = tmp_path / "test.json"
    local_file.write_text("{}")

    mock_client = MagicMock()
    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import upload_file
        upload_file(local_file, bucket="raw-data", key="test.json")

    mock_client.upload_file.assert_not_called()


def test_list_objects_returns_keys(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "raw/a.json"}, {"Key": "raw/b.json"}]},
        {"Contents": [{"Key": "raw/c.json"}]},
    ]
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator

    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import list_objects
        keys = list_objects("raw-data", prefix="raw/")

    assert keys == ["raw/a.json", "raw/b.json", "raw/c.json"]


def test_list_objects_returns_empty_when_not_configured(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    from src.storage.minio_client import list_objects
    assert list_objects("raw-data") == []


def test_sync_to_local_downloads_objects(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "raw/shopify/ruggable/run1.json"}]},
    ]
    mock_client.get_paginator.return_value = mock_paginator

    with patch("src.storage.minio_client._client", return_value=mock_client):
        from src.storage.minio_client import sync_to_local
        downloaded = sync_to_local(
            bucket="raw-data", prefix="raw/", local_dir=tmp_path
        )

    assert len(downloaded) == 1
    assert downloaded[0] == tmp_path / "shopify/ruggable/run1.json"
    mock_client.download_file.assert_called_once_with(
        "raw-data",
        "raw/shopify/ruggable/run1.json",
        str(tmp_path / "shopify/ruggable/run1.json"),
    )
