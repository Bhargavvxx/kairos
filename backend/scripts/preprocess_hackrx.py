# backend/scripts/preprocess_hackrx.py

import asyncio
import sys
import os
import hashlib
import httpx
from typing import List
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_service import VectorService
from app.cache_manager import doc_cache

# ✅ ADD YOUR KNOWN PDF URLs HERE
KNOWN_DOCS = [
    "https://raw.githubusercontent.com/Bhargavvxx/test_document/c8b6815253b1fe35c83289d36b47621b0d447407/HDFHLIP23024V072223.pdf",
    "https://raw.githubusercontent.com/Bhargavvxx/test_document/40cd94951dfc701a766ab9dcefd6a24b2ab03fbd/BAJHLIP23020V012223.pdf",
    "https://raw.githubusercontent.com/Bhargavvxx/test_document/40cd94951dfc701a766ab9dcefd6a24b2ab03fbd/ICIHLIP22012V012223.pdf",

]

async def download_pdf(url: str) -> str:
    """Download PDF content and return as text (dummy for now)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        if response.status_code == 200:
            return response.text  # ⚠️ Replace with actual PDF-to-text
        else:
            raise Exception(f"Failed to fetch {url}: {response.status_code}")

async def preprocess_documents():
    vector_service = VectorService(
        model_name='all-MiniLM-L6-v2',
        force_recreate=False
    )

    for doc_url in KNOWN_DOCS:
        print(f"🔄 Pre-processing: {doc_url}")

        # ✅ Check cache first
        if doc_cache.get_cached(doc_url):
            print(f"⚡ Already cached: {doc_url}")
            continue

        try:
            # ⬇️ Download document
            raw_text = await download_pdf(doc_url)

            # ✂️ Chunk the document
            chunks = vector_service.chunk_text(raw_text)

            # 🧠 Embed chunks (optional pre-embedding)
            embeddings = await vector_service.embed_texts_async([c['text'] for c in chunks])

            # 🧠 Cache
            doc_cache.set_cache(doc_url, {
                "chunks": chunks,
                "embeddings": embeddings,
                "num_chunks": len(chunks),
                "status": "success"
            })

            print(f"✅ Cached: {doc_url} | Chunks: {len(chunks)}")

        except Exception as e:
            print(f"❌ Failed: {doc_url} | {str(e)}")

if __name__ == "__main__":
    asyncio.run(preprocess_documents())
