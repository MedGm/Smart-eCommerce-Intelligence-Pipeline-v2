"""
MinIO / S3-compatible storage client.
All operations are no-ops when MINIO_ENDPOINT is not set — local dev works unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.config import get_logger

logger = get_logger(__name__)


def is_minio_configured() -> bool:
    return bool(os.environ.get("MINIO_ENDPOINT"))


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    )


def upload_file(local_path: Path, bucket: str, key: str) -> None:
    if not is_minio_configured():
        return
    try:
        _client().upload_file(str(local_path), bucket, key)
        logger.debug("Uploaded %s → s3://%s/%s", local_path, bucket, key)
    except Exception as exc:
        logger.warning("MinIO upload failed for %s: %s", local_path, exc)


def download_file(bucket: str, key: str, local_path: Path) -> None:
    if not is_minio_configured():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(local_path))
    logger.debug("Downloaded s3://%s/%s → %s", bucket, key, local_path)


def list_objects(bucket: str, prefix: str = "") -> list[str]:
    if not is_minio_configured():
        return []
    paginator = _client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def sync_to_local(bucket: str, prefix: str, local_dir: Path) -> list[Path]:
    """Download all objects matching prefix into local_dir, preserving key sub-path."""
    if not is_minio_configured():
        return []
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for key in list_objects(bucket, prefix):
        rel = key[len(prefix):].lstrip("/")
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(bucket, key, str(dest))
        downloaded.append(dest)
    return downloaded
