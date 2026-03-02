// src/api.js
// Talks to the FastAPI backend through the Vite dev-server proxy (/api → localhost:8000)

const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

/** Upload files for processing. Returns tracking IDs. */
export async function uploadFiles(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  const res = await fetch(`${BASE}/documents/upload/batch`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Upload failed (${res.status}): ${body}`);
  }
  return res.json();
}

/** Upload a single file. */
export async function uploadSingleFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Upload failed (${res.status}): ${body}`);
  }
  return res.json();
}

/** Poll processing status for a tracking id. */
export async function getProcessingStatus(trackingId) {
  return request(`/documents/status/${trackingId}`);
}

/** Ask a question via the RAG chat endpoint. */
export async function askKairos(query) {
  return request('/query/chat', {
    method: 'POST',
    body: JSON.stringify({ query }),
  });
}

/** Fetch dashboard intelligence (risks & opportunities). */
export async function getDashboardData() {
  return request('/query/dashboard');
}

/** Fetch the knowledge graph for visualisation. */
export async function getGraphData() {
  return request('/query/graph');
}

/** Health check. */
export async function healthCheck() {
  return request('/health');
}

/** Get system stats (documents, chunks, entities, relationships). */
export async function getStats() {
  return request('/stats');
}
