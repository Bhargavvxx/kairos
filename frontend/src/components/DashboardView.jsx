import React, { useState, useEffect } from 'react';
import { Loader, AlertTriangle, Sparkles, RefreshCw, FileText, Database, Share2, Box } from 'lucide-react';
import { getDashboardData } from '../api';

function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 flex items-center gap-4 hover:shadow-md transition-shadow">
      <div className={`w-11 h-11 rounded-lg flex items-center justify-center ${color}`}>
        <Icon size={20} />
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-800">{value ?? '—'}</p>
        <p className="text-xs text-gray-400 font-medium">{label}</p>
      </div>
    </div>
  );
}

export default function DashboardView({ systemStats, refreshStats }) {
  const [data, setData] = useState({ risks: [], opportunities: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const dashboardData = await getDashboardData();
      setData(dashboardData);
    } catch (err) {
      console.error('Dashboard fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchData(), refreshStats?.()]);
    setRefreshing(false);
  };

  const severityColor = {
    High: 'bg-red-100 text-red-800',
    Medium: 'bg-yellow-100 text-yellow-800',
    Low: 'bg-green-100 text-green-800',
  };

  const potentialColor = {
    High: 'bg-green-100 text-green-800',
    Medium: 'bg-blue-100 text-blue-800',
    Low: 'bg-gray-100 text-gray-800',
  };

  const noInsights = data.risks.length === 0 && data.opportunities.length === 0;

  return (
    <div className="h-full overflow-y-auto bg-gray-50/50">
      <div className="p-6 max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Intelligence Dashboard</h1>
            <p className="text-sm text-gray-400 mt-0.5">System overview, risks, and opportunities</p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-600 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition disabled:opacity-50"
          >
            <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>

        {/* System Stats Cards */}
        {systemStats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard icon={FileText} label="Documents" value={systemStats.total_documents} color="bg-blue-50 text-blue-600" />
            <StatCard icon={Database} label="Chunks" value={systemStats.total_chunks} color="bg-indigo-50 text-indigo-600" />
            <StatCard icon={Box} label="Entities" value={systemStats.total_entities} color="bg-purple-50 text-purple-600" />
            <StatCard icon={Share2} label="Relationships" value={systemStats.total_relationships} color="bg-pink-50 text-pink-600" />
          </div>
        )}

        {/* Loading state for insights */}
        {loading && (
          <div className="flex items-center justify-center py-16">
            <Loader className="w-7 h-7 animate-spin text-indigo-600" />
            <span className="ml-3 text-sm text-gray-500">Analyzing documents for insights...</span>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700 flex items-center gap-2">
            <AlertTriangle size={16} />
            Failed to load insights: {error}
          </div>
        )}

        {/* No insights */}
        {!loading && !error && noInsights && (
          <div className="bg-white border border-gray-200 rounded-xl p-8 text-center">
            <Sparkles className="w-10 h-10 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 text-sm">
              No insights yet. Upload and process more documents to generate risk and opportunity analysis.
            </p>
          </div>
        )}

        {/* Insights Grid */}
        {!loading && !error && !noInsights && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Risks */}
            {data.risks.length > 0 && (
              <div>
                <h2 className="text-base font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <AlertTriangle size={18} className="text-red-500" />
                  Identified Risks
                  <span className="ml-auto text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full">{data.risks.length}</span>
                </h2>
                <div className="space-y-3">
                  {data.risks.map((risk) => (
                    <div
                      key={risk.id}
                      className="bg-white p-4 rounded-xl border border-gray-200 hover:shadow-md transition-shadow"
                    >
                      <div className="flex justify-between items-start gap-2">
                        <h3 className="font-semibold text-sm text-gray-800">{risk.title}</h3>
                        <span
                          className={`flex-shrink-0 px-2 py-0.5 text-[11px] font-medium rounded-full ${severityColor[risk.severity] || 'bg-gray-100 text-gray-800'}`}
                        >
                          {risk.severity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1.5 leading-relaxed">{risk.description}</p>
                      {risk.document && (
                        <p className="text-[11px] text-gray-400 mt-2 flex items-center gap-1">
                          <FileText size={10} />
                          {risk.document}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Opportunities */}
            {data.opportunities.length > 0 && (
              <div>
                <h2 className="text-base font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <Sparkles size={18} className="text-green-500" />
                  Growth Opportunities
                  <span className="ml-auto text-xs bg-green-100 text-green-600 px-2 py-0.5 rounded-full">{data.opportunities.length}</span>
                </h2>
                <div className="space-y-3">
                  {data.opportunities.map((opp) => (
                    <div
                      key={opp.id}
                      className="bg-white p-4 rounded-xl border border-gray-200 hover:shadow-md transition-shadow"
                    >
                      <div className="flex justify-between items-start gap-2">
                        <h3 className="font-semibold text-sm text-gray-800">{opp.title}</h3>
                        <span
                          className={`flex-shrink-0 px-2 py-0.5 text-[11px] font-medium rounded-full ${potentialColor[opp.potential] || 'bg-gray-100 text-gray-800'}`}
                        >
                          {opp.potential} Potential
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1.5 leading-relaxed">{opp.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
