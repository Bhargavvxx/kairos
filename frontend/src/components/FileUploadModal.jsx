import React, { useState, useRef } from 'react';
import { UploadCloud, Loader } from 'lucide-react';
import { uploadSingleFile, getProcessingStatus } from '../api';

export default function FileUploadModal({ onFilesProcessed, setNotification }) {
  const [files, setFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files));
    }
  };

  const pollStatus = async (trackingId) => {
    const maxAttempts = 120; // 2 minutes
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const status = await getProcessingStatus(trackingId);
        setProgress(status.progress);
        if (status.status === 'completed') return true;
        if (status.status === 'failed') return false;
      } catch {
        // ignore transient errors
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    return false;
  };

  const handleProcessClick = async () => {
    if (files.length === 0) {
      setNotification({ type: 'error', message: 'Please select files to process.' });
      return;
    }
    setIsProcessing(true);
    setProgress(0);

    let successCount = 0;
    for (const file of files) {
      try {
        const result = await uploadSingleFile(file);
        const ok = await pollStatus(result.tracking_id);
        if (ok) successCount++;
      } catch (err) {
        console.error('Upload error:', err);
      }
    }

    if (successCount > 0) {
      setNotification({
        type: 'success',
        message: `${successCount} of ${files.length} document(s) processed successfully.`,
      });
      onFilesProcessed();
    } else {
      setNotification({ type: 'error', message: 'Processing failed. Please try again.' });
    }

    setIsProcessing(false);
    setFiles([]);
    setProgress(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer.files) {
      setFiles(Array.from(event.dataTransfer.files));
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  return (
    <div
      className="border border-dashed border-gray-300 rounded-lg p-6 text-center"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
      <p className="mt-2 text-sm text-gray-600">
        <button
          type="button"
          className="font-medium text-indigo-600 hover:text-indigo-500 focus:outline-none"
          onClick={() => fileInputRef.current?.click()}
        >
          Click to upload
        </button>{' '}
        or drag and drop
      </p>
      <p className="text-xs text-gray-400 mt-1">PDF, TXT, DOCX supported (max 10 MB)</p>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.doc,.docx,.html,.md"
        className="hidden"
        onChange={handleFileChange}
      />
      {files.length > 0 && (
        <div className="mt-4 text-sm text-gray-500">
          <p className="font-semibold">Selected files:</p>
          <ul className="list-disc list-inside">
            {files.map((file) => (
              <li key={file.name}>{file.name}</li>
            ))}
          </ul>
        </div>
      )}
      {progress !== null && (
        <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
      <button
        onClick={handleProcessClick}
        disabled={isProcessing || files.length === 0}
        className="mt-6 w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-300 disabled:cursor-not-allowed"
      >
        {isProcessing ? (
          <>
            <Loader className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
            Processing{progress !== null ? ` (${progress}%)` : '...'}
          </>
        ) : (
          'Process Documents'
        )}
      </button>
    </div>
  );
}
