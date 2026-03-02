# backend/scripts/preprocess_hackrx.py

import asyncio
import sys
import os
import hashlib
import io
import httpx
from typing import List
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_service import VectorService
from app.cache_manager import doc_cache

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Add known PDF URLs here
KNOWN_DOCS = [
    "https://raw.githubusercontent.com/Bhargavvxx/test_document/c8b6815253b1fe35c83289d36b47621b0d447407/HDFHLIP23024V072223.pdf",
    "https://raw.githubusercontent.com/Bhargavvxx/test_document/40cd94951dfc701a766ab9dcefd6a24b2ab03fbd/BAJHLIP23020V012223.pdf",
    "https://raw.githubusercontent.com/Bhargavvxx/test_document/40cd94951dfc701a766ab9dcefd6a24b2ab03fbd/ICIHLIP22012V012223.pdf",

]

async def download_pdf(url: str) -> str:
    """Download PDF content and extract text."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch {url}: {response.status_code}")
        
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF text extraction. Install with: pip install PyPDF2")
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
        text_parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        if not text_parts:
            raise Exception(f"No text could be extracted from {url}")
        
        return "\n".join(text_parts)

async def preprocess_documents():
    vector_service = VectorService(
        model_name='all-MiniLM-L6-v2',
        force_recreate=False
    )

    for doc_url in KNOWN_DOCS:
        print(f"Pre-processing: {doc_url}")

        # Check cache first
        if doc_cache.get_cached(doc_url):
            print(f"Already cached: {doc_url}")
            continue

        try:
            # Download document and extract text
            raw_text = await download_pdf(doc_url)

            # Chunk the document - returns List[Tuple[str, int, int]]
            chunk_tuples = vector_service._smart_chunk_text(raw_text)
            chunk_texts = [text for text, start, end in chunk_tuples]

            # Embed chunks
            embeddings = await vector_service._encode_texts_async(chunk_texts)

            # Cache results
            doc_cache.set_cache(doc_url, {
                "chunks": [{"text": text, "start": start, "end": end} for text, start, end in chunk_tuples],
                "embeddings": embeddings,
                "num_chunks": len(chunk_tuples),
                "status": "success"
            })

            print(f"Cached: {doc_url} | Chunks: {len(chunk_tuples)}")

        except Exception as e:
            print(f"Failed: {doc_url} | {str(e)}")

if __name__ == "__main__":
    asyncio.run(preprocess_documents())
