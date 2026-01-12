"""
WhatsApp Media Cache
- Cache media_id for uploaded files to avoid re-uploading
- Uses file hash + phone number ID as cache key
- Stores in local JSON file (dev) or can be extended to DB (prod)
"""
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

def _ensure_cache_dir(cache_path: str) -> None:
    """Ensure the cache file directory exists"""
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

def get_cached_media_id(file_hash: str, phone_number_id: str, cache_path: str) -> Optional[str]:
    """
    Get cached media_id for a file hash and phone number ID.
    
    Args:
        file_hash: SHA256 hash of the file
        phone_number_id: WhatsApp Business Phone Number ID
        cache_path: Path to cache JSON file
    
    Returns:
        media_id if found in cache, None otherwise
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        
        cache_key = f"{file_hash}_{phone_number_id}"
        entry = cache.get(cache_key)
        
        if entry and entry.get('media_id'):
            return entry['media_id']
        
        return None
    except (json.JSONDecodeError, IOError, KeyError):
        # Cache file corrupted or missing - return None
        return None

def save_cached_media_id(
    file_hash: str,
    phone_number_id: str,
    media_id: str,
    cache_path: str
) -> None:
    """
    Save media_id to cache.
    
    Args:
        file_hash: SHA256 hash of the file
        phone_number_id: WhatsApp Business Phone Number ID
        media_id: Media ID returned from WhatsApp API
        cache_path: Path to cache JSON file
    """
    _ensure_cache_dir(cache_path)
    
    # Load existing cache
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            cache = {}
    
    # Add new entry
    cache_key = f"{file_hash}_{phone_number_id}"
    cache[cache_key] = {
        "media_id": media_id,
        "file_hash": file_hash,
        "phone_number_id": phone_number_id,
        "created_at": datetime.now().isoformat()
    }
    
    # Save cache
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError:
        # If cache write fails, log but don't crash
        pass

