# backend/app/cache_manager.py
# CREATE THIS NEW FILE

import hashlib
import pickle
import os
import time
from typing import Dict, Optional

class DocumentCache:
    def __init__(self):
        self.cache = {}
        self.cache_file = "./cache/document_cache.pkl"
        self._load_cache()
    
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
        except:
            self.cache = {}
    
    def _save_cache(self):
        os.makedirs("./cache", exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def get_doc_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def get_cached(self, url: str) -> Optional[Dict]:
        doc_hash = self.get_doc_hash(url)
        if doc_hash in self.cache:
            return self.cache[doc_hash]
        return None
    
    def set_cache(self, url: str, data: Dict):
        doc_hash = self.get_doc_hash(url)
        self.cache[doc_hash] = data
        self._save_cache()

doc_cache = DocumentCache()