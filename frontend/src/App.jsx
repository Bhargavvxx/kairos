import { useState, useEffect, useCallback } from 'react';
import { FileText, LayoutDashboard, Share2, X, UploadCloud, Loader } from 'lucide-react';
import FileUploadModal from './components/FileUploadModal';
import ChatView from './components/ChatView';
import DashboardView from './components/DashboardView';
import GraphView from './components/GraphView';
import { getStats } from './api';

export default function App() {
  const [activeView, setActiveView] = useState('chat');
  const [notification, setNotification] = useState(null);
  const [showUpload, setShowUpload] = useState(false);
  const [loading, setLoading] = useState(true);
  const [systemStats, setSystemStats] = useState(null);

  // Lift chat messages to App so they persist across tab switches
  const [chatMessages, setChatMessages] = useState([
    { from: 'bot', text: 'Hello! I\'m **KAIROS** — your intelligent document analysis assistant.\n\nAsk me anything about your uploaded documents, or try one of the suggested questions below.' },
  ]);

  // On mount: check if documents already exist in the backend
  useEffect(() => {
    const checkExistingDocs = async () => {
      try {
        const stats = await getStats();
        setSystemStats(stats.system);
        if ((stats.system?.total_chunks || 0) === 0) {
          setShowUpload(true);
        }
      } catch (err) {
        console.error('Failed to fetch stats:', err);
      } finally {
        setLoading(false);
      }
    };
    checkExistingDocs();
  }, []);

  const refreshStats = useCallback(async () => {
    try {
      const stats = await getStats();
      setSystemStats(stats.system);
    } catch (err) {
      console.error('Failed to refresh stats:', err);
    }
  }, []);

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const NavItem = ({ icon: Icon, label, viewName, badge }) => (
    <button
      onClick={() => { setActiveView(viewName); setShowUpload(false); }}
      className={`flex items-center w-full px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
        !showUpload && activeView === viewName
          ? 'bg-indigo-600 text-white shadow-md shadow-indigo-200'
          : 'text-gray-600 hover:bg-gray-100'
      }`}
    >
      <Icon className="mr-3 h-5 w-5" />
      {label}
      {badge && (
        <span className="ml-auto text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full">
          {badge}
        </span>
      )}
    </button>
  );

  const renderView = () => {
    if (loading) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-3">
          <Loader className="w-10 h-10 animate-spin text-indigo-600" />
          <span className="text-gray-500 text-sm">Connecting to KAIROS...</span>
        </div>
      );
    }

    if (showUpload) {
      return (
        <div className="p-8 max-w-2xl mx-auto">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Upload Documents</h1>
          <p className="text-gray-600 mb-6">
            Upload new documents to add to the knowledge base.
            {systemStats && systemStats.total_chunks > 0 && (
              <span className="block mt-2 text-sm text-indigo-600 bg-indigo-50 px-3 py-2 rounded-lg">
                {systemStats.total_documents} document(s) already indexed — {systemStats.total_chunks} chunks, {systemStats.total_entities} entities
              </span>
            )}
          </p>
          <FileUploadModal
            onFilesProcessed={() => {
              setShowUpload(false);
              setActiveView('chat');
              refreshStats();
            }}
            setNotification={setNotification}
          />
          {systemStats && systemStats.total_chunks > 0 && (
            <button
              onClick={() => { setShowUpload(false); setActiveView('chat'); }}
              className="mt-4 text-sm text-indigo-600 hover:text-indigo-800 underline"
            >
              Skip — use existing documents
            </button>
          )}
        </div>
      );
    }

    switch (activeView) {
      case 'chat':
        return <ChatView messages={chatMessages} setMessages={setChatMessages} />;
      case 'dashboard':
        return <DashboardView systemStats={systemStats} refreshStats={refreshStats} />;
      case 'graph':
        return <GraphView />;
      default:
        return <ChatView messages={chatMessages} setMessages={setChatMessages} />;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 font-sans antialiased">
      {/* Notification */}
      {notification && (
        <div
          className={`fixed top-5 right-5 z-50 flex items-center p-4 rounded-lg shadow-lg text-white animate-slide-in ${
            notification.type === 'success' ? 'bg-green-500' : 'bg-red-500'
          }`}
        >
          <p className="text-sm">{notification.message}</p>
          <button onClick={() => setNotification(null)} className="ml-4 hover:opacity-80">
            <X size={18} />
          </button>
        </div>
      )}

      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col p-4">
        <div className="flex items-center mb-8">
          <svg
            className="w-10 h-10 text-indigo-600"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M12 2L2 7V17L12 22L22 17V7L12 2Z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M2 7L12 12L22 7"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M12 22V12"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <div className="ml-2">
            <h1 className="text-xl font-bold text-gray-800 leading-tight">KAIROS</h1>
            <p className="text-[10px] text-gray-400 leading-tight">Document Intelligence</p>
          </div>
        </div>
        <nav className="space-y-1">
          <NavItem icon={FileText} label="Chat" viewName="chat"
            badge={chatMessages.length > 1 ? chatMessages.filter(m => m.from === 'user').length : null} />
          <NavItem icon={LayoutDashboard} label="Dashboard" viewName="dashboard" />
          <NavItem icon={Share2} label="Knowledge Graph" viewName="graph" />
        </nav>
        <div className="mt-auto space-y-2">
          {systemStats && systemStats.total_chunks > 0 && (
            <div className="text-xs text-gray-400 px-3 py-2 border border-gray-100 rounded-lg bg-gray-50">
              <div className="grid grid-cols-2 gap-1">
                <span>{systemStats.total_documents} docs</span>
                <span>{systemStats.total_chunks} chunks</span>
                <span>{systemStats.total_entities} entities</span>
                <span>{systemStats.total_relationships} relations</span>
              </div>
            </div>
          )}
          <button
            onClick={() => setShowUpload(true)}
            className={`flex items-center w-full px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
              showUpload
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-200'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <UploadCloud className="mr-3 h-5 w-5" />
            Upload Documents
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        <div className="flex-1 bg-white m-4 rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          {renderView()}
        </div>
      </main>
    </div>
  );
}
