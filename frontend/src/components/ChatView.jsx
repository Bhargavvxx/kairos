import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Bot, User, Copy, Check, Clock, FileText, Trash2, Lightbulb, Send, ChevronDown, ChevronUp, BookOpen } from 'lucide-react';
import { askKairos } from '../api';

const SUGGESTED_QUESTIONS = [
  'What is the waiting period for pre-existing diseases?',
  'What are the permanent exclusions under this policy?',
  'Explain the claim settlement process.',
  'What is the grace period for premium payment?',
  'Is maternity or newborn coverage included?',
  'What are the sub-limits on room rent?',
];

function formatMarkdown(text) {
  return text
    .replace(/\n/g, '<br />')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code class="bg-gray-200 text-gray-800 px-1 rounded text-xs">$1</code>')
    // Highlight citation markers like [1], [2] etc.
    .replace(/\[(\d+)\]/g, '<span class="inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold bg-indigo-100 text-indigo-700 rounded-full cursor-help" title="See citation $1">$1</span>');
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-gray-600 p-1 rounded"
      title="Copy to clipboard"
    >
      {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
    </button>
  );
}

function CitationList({ citations }) {
  const [expanded, setExpanded] = useState(false);

  if (!citations || citations.length === 0) return null;

  const shown = expanded ? citations : citations.slice(0, 2);

  return (
    <div className="mt-2 px-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-[11px] font-medium text-indigo-600 hover:text-indigo-800 transition-colors mb-1.5"
      >
        <BookOpen size={12} />
        {citations.length} source{citations.length > 1 ? 's' : ''} cited
        {citations.length > 2 && (expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />)}
      </button>
      <div className="space-y-1.5">
        {shown.map((cit) => (
          <div
            key={cit.id}
            className="flex items-start gap-2 bg-gray-50 border border-gray-100 rounded-lg px-3 py-2 text-xs"
          >
            <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center bg-indigo-100 text-indigo-700 rounded-full font-bold text-[10px]">
              {cit.id}
            </span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="font-medium text-gray-700 truncate">{cit.source}</span>
                {cit.page && <span className="text-gray-400">p.{cit.page}</span>}
                <span className={`ml-auto text-[10px] px-1.5 py-0.5 rounded-full ${
                  cit.score >= 0.7 ? 'bg-green-50 text-green-600' :
                  cit.score >= 0.4 ? 'bg-yellow-50 text-yellow-600' :
                  'bg-red-50 text-red-500'
                }`}>
                  {Math.round(cit.score * 100)}%
                </span>
              </div>
              <p className="text-gray-500 leading-snug line-clamp-2">{cit.snippet}</p>
            </div>
          </div>
        ))}
      </div>
      {!expanded && citations.length > 2 && (
        <button
          onClick={() => setExpanded(true)}
          className="text-[11px] text-gray-400 hover:text-indigo-500 mt-1 transition-colors"
        >
          + {citations.length - 2} more
        </button>
      )}
    </div>
  );
}

export default function ChatView({ messages, setMessages }) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({ top: chatContainerRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = useCallback(async (queryText) => {
    const query = queryText || input.trim();
    if (!query) return;

    const userMessage = { from: 'user', text: query, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await askKairos(query);
      const answer = response.answer || 'No answer was returned.';

      const botMessage = {
        from: 'bot',
        text: '',
        sources: response.sources || [],
        citations: response.citations || [],
        confidence: response.confidence || 0,
        processingTime: response.processing_time || 0,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, botMessage]);

      // Word-by-word streaming
      const words = answer.split(' ');
      let streamedText = '';
      for (let i = 0; i < words.length; i++) {
        streamedText += (i > 0 ? ' ' : '') + words[i];
        const captured = streamedText;
        setMessages((prev) =>
          prev.map((msg, index) =>
            index === prev.length - 1 ? { ...msg, text: captured } : msg,
          ),
        );
        await new Promise((r) => setTimeout(r, 15));
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { from: 'bot', text: `Sorry, something went wrong: ${err.message}`, timestamp: Date.now() },
      ]);
    }

    setIsLoading(false);
    inputRef.current?.focus();
  }, [input, setMessages]);

  const handleClearChat = () => {
    setMessages([
      { from: 'bot', text: 'Chat cleared. Ask me anything about your documents!', timestamp: Date.now() },
    ]);
  };

  const onlyWelcome = messages.length === 1 && messages[0].from === 'bot';

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-100 bg-gray-50/50">
        <div className="flex items-center gap-2">
          <Bot size={20} className="text-indigo-600" />
          <h2 className="text-sm font-semibold text-gray-700">KAIROS Chat</h2>
          <span className="text-xs text-gray-400">— Ask questions about your documents</span>
        </div>
        {messages.length > 1 && (
          <button
            onClick={handleClearChat}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-red-500 transition-colors px-2 py-1 rounded hover:bg-red-50"
            title="Clear chat history"
          >
            <Trash2 size={14} />
            Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex items-start gap-3 ${msg.from === 'user' ? 'justify-end' : ''} group`}
          >
            {msg.from === 'bot' && (
              <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-sm">
                <Bot size={18} />
              </div>
            )}
            <div className={`max-w-2xl ${msg.from === 'user' ? 'order-first' : ''}`}>
              <div
                className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.from === 'bot'
                    ? 'bg-gray-50 text-gray-800 rounded-tl-sm border border-gray-100'
                    : 'bg-indigo-600 text-white rounded-br-sm shadow-sm'
                }`}
              >
                <div
                  className="whitespace-pre-wrap"
                  dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.text) }}
                />
              </div>
              {/* Meta info for bot messages */}
              {msg.from === 'bot' && msg.text && (
                <div className="flex items-center gap-3 mt-1.5 px-1">
                  <CopyButton text={msg.text} />
                  {msg.processingTime > 0 && (
                    <span className="text-[11px] text-gray-300 flex items-center gap-1">
                      <Clock size={10} />
                      {msg.processingTime.toFixed(1)}s
                    </span>
                  )}
                  {msg.confidence > 0 && (
                    <span className={`text-[11px] flex items-center gap-1 ${
                      msg.confidence >= 0.7 ? 'text-green-400' : msg.confidence >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {Math.round(msg.confidence * 100)}% confidence
                    </span>
                  )}
                </div>
              )}
              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2 px-1">
                  {msg.sources.map((source) => (
                    <span
                      key={source}
                      className="inline-flex items-center gap-1 text-[11px] bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded-full border border-indigo-100"
                    >
                      <FileText size={10} />
                      {source}
                    </span>
                  ))}
                </div>
              )}
              {/* Chunk-level citations */}
              {msg.citations && msg.citations.length > 0 && (
                <CitationList citations={msg.citations} />
              )}
            </div>
            {msg.from === 'user' && (
              <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gray-200 flex items-center justify-center text-gray-600">
                <User size={18} />
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && messages[messages.length - 1]?.from !== 'bot' && (
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-sm">
              <Bot size={18} />
            </div>
            <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-gray-50 border border-gray-100">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        {/* Suggested questions on empty chat */}
        {onlyWelcome && !isLoading && (
          <div className="pt-2">
            <div className="flex items-center gap-2 mb-3">
              <Lightbulb size={14} className="text-amber-500" />
              <span className="text-xs font-medium text-gray-400 uppercase tracking-wide">Suggested Questions</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  className="text-left text-sm text-gray-600 bg-gray-50 hover:bg-indigo-50 hover:text-indigo-700 border border-gray-200 hover:border-indigo-200 rounded-xl px-4 py-3 transition-all duration-200"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="px-4 py-3 border-t border-gray-100 bg-white">
        <div className="flex items-center gap-2 max-w-4xl mx-auto">
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && !isLoading && handleSend()}
              placeholder="Ask a question about your documents..."
              className="w-full pl-4 pr-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition text-sm bg-gray-50 focus:bg-white"
              disabled={isLoading}
            />
          </div>
          <button
            onClick={() => handleSend()}
            disabled={isLoading || !input.trim()}
            className="flex items-center justify-center w-11 h-11 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-200 text-white disabled:text-gray-400 rounded-xl transition-all duration-200 shadow-sm disabled:shadow-none"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
