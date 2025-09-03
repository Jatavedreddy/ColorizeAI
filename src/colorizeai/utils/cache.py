"""
Cache management utilities
"""

import hashlib
import os
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

class CacheManager:
    """Manages image processing cache"""
    
    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.max_size = max_size
    
    def _sha_key(self, arr: np.ndarray, strength: float, **kwargs) -> str:
        h = hashlib.sha256()
        h.update(arr.tobytes())
        h.update(str(strength).encode())
        # Include additional parameters in cache key
        for key, value in sorted(kwargs.items()):
            h.update(f"{key}:{value}".encode())
        return h.hexdigest()
    
    def get(self, key: str) -> Tuple[np.ndarray, np.ndarray] | None:
        return self.cache.get(key)
    
    def put(self, key: str, value: Tuple[np.ndarray, np.ndarray]):
        self.cache[key] = value
        self._manage_cache()
    
    def _manage_cache(self):
        """Remove oldest cache entries if cache is too large"""
        if len(self.cache) > self.max_size:
            # Remove 20% of oldest entries
            keys_to_remove = list(self.cache.keys())[:self.max_size // 5]
            for key in keys_to_remove:
                del self.cache[key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()

class VideoCacheManager:
    """Manages video processing cache with persistent registry"""
    
    def __init__(self, cache_dir: Path, max_size: int = 10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.registry_file = self.cache_dir / "cache_registry.json"
        
        # Load existing cache registry
        self._load_registry()
        
        # Clean up orphaned files and broken entries
        self._cleanup_cache()
    
    def _load_registry(self):
        """Load cache registry from disk"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"📚 Loaded video cache registry: {len(self.cache)} entries")
            else:
                print("📚 No existing cache registry found, starting fresh")
        except Exception as e:
            print(f"⚠️ Could not load cache registry: {e}")
            self.cache = {}
    
    def _save_registry(self):
        """Save cache registry to disk"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save cache registry: {e}")
    
    def _cleanup_cache(self):
        """Remove orphaned files and broken cache entries"""
        # Remove cache entries for files that no longer exist
        broken_keys = []
        for key, path in self.cache.items():
            if not os.path.exists(path):
                broken_keys.append(key)
        
        for key in broken_keys:
            del self.cache[key]
            print(f"🧹 Removed broken cache entry: {key[:16]}...")
        
        # Remove orphaned files (files without cache entries)
        cache_files = set(self.cache_dir.glob("*.mp4"))
        registered_files = {Path(path) for path in self.cache.values()}
        orphaned_files = cache_files - registered_files
        
        for orphan in orphaned_files:
            if orphan.name != "cache_registry.json":  # Don't delete registry file
                try:
                    # Try to register orphaned files by scanning for their potential keys
                    self._try_register_orphan(orphan)
                except Exception:
                    # If can't register, optionally remove (commented out for safety)
                    # orphan.unlink()
                    print(f"🔍 Found orphaned file (keeping): {orphan.name}")
        
        # Save updated registry
        if broken_keys:
            self._save_registry()
    
    def _try_register_orphan(self, orphan_file: Path):
        """Try to register an orphaned file by reverse-engineering its key"""
        # This is complex since we'd need to guess the original parameters
        # For now, just log the orphan - user can re-process to re-cache
        print(f"🔍 Orphaned cache file: {orphan_file.name}")
    
    def _video_cache_key(self, video_path: str, strength: float, frame_skip: int, 
                        resolution: str, custom_width: int, custom_height: int, 
                        fast_mode: bool, use_temporal_consistency: bool = False, 
                        style_type: str = 'none') -> str:
        """Generate robust cache key for video processing.

        Improvements over previous version:
        - Multi-region sampling (start, middle, end) instead of only first 1MB.
        - Includes file size and modification time for quick invalidation.
        - Falls back gracefully if file operations fail.
        - Normalizes optional custom dimensions (None -> 0) for stable hashing.
        - Separates file fingerprint hash from parameters hash then combines.
        """
        file_hasher = hashlib.sha256()
        param_hasher = hashlib.sha256()

        # --- File fingerprint -------------------------------------------------
        try:
            file_size = os.path.getsize(video_path)
            mtime = int(os.path.getmtime(video_path))
            # Encode basic metadata first (size + mtime)
            file_hasher.update(f"{file_size}:{mtime}".encode())

            # Open once and sample regions
            with open(video_path, 'rb') as f:
                # Sample up to 3 regions: start, middle, end
                regions = []
                sample_size = 256 * 1024  # 256KB per region (tunable)
                if file_size <= sample_size * 3:
                    # Small file: read entirely
                    regions.append(f.read())
                else:
                    # Start
                    regions.append(f.read(sample_size))
                    # Middle
                    mid_pos = max(sample_size, (file_size // 2) - (sample_size // 2))
                    f.seek(mid_pos)
                    regions.append(f.read(sample_size))
                    # End
                    end_pos = max(0, file_size - sample_size)
                    f.seek(end_pos)
                    regions.append(f.read(sample_size))
                for chunk in regions:
                    file_hasher.update(chunk)
        except Exception:
            # Fallback: path string (still deterministic but weaker)
            file_hasher.update(video_path.encode())
        
        # --- Parameter fingerprint -------------------------------------------
        # Normalize None dimensions to 0 for deterministic representation
        cw = custom_width if custom_width is not None else 0
        ch = custom_height if custom_height is not None else 0
        # Round strength to avoid floating noise differences
        norm_strength = round(float(strength), 5)
        param_tuple = (
            norm_strength,
            int(frame_skip),
            str(resolution),
            int(cw),
            int(ch),
            int(bool(fast_mode)),
            int(bool(use_temporal_consistency)),
            str(style_type or 'none')
        )
        param_hasher.update(json.dumps(param_tuple, separators=(',', ':')).encode())

        # Combine both hashes to produce final key
        combined = hashlib.sha256()
        combined.update(file_hasher.hexdigest().encode())
        combined.update(param_hasher.hexdigest().encode())
        return combined.hexdigest()
    
    def get(self, key: str) -> str | None:
        """Get cached video path if it exists"""
        cached_path = self.cache.get(key)
        if cached_path and os.path.exists(cached_path):
            return cached_path
        elif key in self.cache:
            # Remove invalid cache entry
            del self.cache[key]
        return None
    
    def put(self, key: str, video_path: str):
        """Add video to cache and save registry"""
        self.cache[key] = video_path
        self._save_registry()  # Persist to disk
        self._manage_cache()
    
    def _manage_cache(self):
        """Remove oldest video cache entries and clean up files"""
        keys_to_remove = []
        
        if len(self.cache) > self.max_size:
            # Remove 30% of oldest entries
            keys_to_remove = list(self.cache.keys())[:self.max_size // 3]
            for key in keys_to_remove:
                video_path = self.cache[key]
                # Try to remove the cached video file
                try:
                    if os.path.exists(video_path):
                        os.unlink(video_path)
                except Exception as e:
                    print(f"Warning: Could not remove cached video {video_path}: {e}")
                del self.cache[key]
        
        # Save registry after cleanup
        if keys_to_remove:
            self._save_registry()
    
    def clear(self):
        """Clear all cache entries and remove files"""
        for video_path in self.cache.values():
            try:
                if os.path.exists(video_path):
                    os.unlink(video_path)
            except Exception as e:
                print(f"Warning: Could not remove cached video {video_path}: {e}")
        self.cache.clear()
        
        # Remove registry file
        try:
            if self.registry_file.exists():
                self.registry_file.unlink()
        except Exception as e:
            print(f"Warning: Could not remove registry file: {e}")

# Global cache instances
_image_cache = None
_video_cache = None

def get_image_cache() -> CacheManager:
    """Get the global image cache instance"""
    global _image_cache
    if _image_cache is None:
        _image_cache = CacheManager()
    return _image_cache

def get_video_cache() -> VideoCacheManager:
    """Get the global video cache instance"""
    global _video_cache
    if _video_cache is None:
        # Default cache directory
        cache_dir = Path(__file__).parent.parent.parent.parent / "cache" / "videos"
        _video_cache = VideoCacheManager(cache_dir)
    return _video_cache
