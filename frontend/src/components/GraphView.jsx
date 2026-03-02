import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Loader, Search, ZoomIn, ZoomOut, Maximize2, Info, X } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import { getGraphData } from '../api';

const colorMap = {
  Person: '#3B82F6',
  Company: '#10B981',
  Policy: '#F59E0B',
  Claim: '#EF4444',
  Contract: '#8B5CF6',
  Document: '#6B7280',
  Amount: '#EC4899',
  Date: '#14B8A6',
  Location: '#F97316',
  Entity: '#9CA3AF',
};

export default function GraphView() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const fgRef = useRef();

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getGraphData();
        setGraphData({ nodes: data.nodes || [], links: data.links || [] });
      } catch (err) {
        console.error('Graph fetch error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Search filtering
  const filteredData = useMemo(() => {
    if (!search.trim()) {
      setHighlightNodes(new Set());
      return graphData;
    }
    const q = search.toLowerCase();
    const matchedIds = new Set();
    graphData.nodes.forEach((n) => {
      if ((n.label || n.id || '').toLowerCase().includes(q) || (n.group || '').toLowerCase().includes(q)) {
        matchedIds.add(n.id);
      }
    });
    setHighlightNodes(matchedIds);
    return graphData; // Keep all nodes visible but highlight matches
  }, [search, graphData]);

  const handleNodeClick = useCallback((node) => {
    setSelectedNode(node);
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 400);
      fgRef.current.zoom(3, 400);
    }
  }, []);

  const handleZoomIn = () => fgRef.current?.zoom(fgRef.current.zoom() * 1.4, 300);
  const handleZoomOut = () => fgRef.current?.zoom(fgRef.current.zoom() / 1.4, 300);
  const handleFit = () => fgRef.current?.zoomToFit(400, 40);

  // Find connected edges for selected node
  const connectedEdges = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.links.filter(
      (l) =>
        (l.source?.id ?? l.source) === selectedNode.id ||
        (l.target?.id ?? l.target) === selectedNode.id,
    );
  }, [selectedNode, graphData.links]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2">
        <Loader className="w-7 h-7 animate-spin text-indigo-600" />
        <span className="text-sm text-gray-500">Loading knowledge graph...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-500 p-6 text-sm">
        Failed to load graph: {error}
      </div>
    );
  }

  const noData = graphData.nodes.length === 0;

  return (
    <div className="h-full overflow-hidden bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-100 bg-gray-50/50 flex-shrink-0">
        <div>
          <h2 className="text-sm font-semibold text-gray-700">Knowledge Graph</h2>
          <p className="text-xs text-gray-400">
            {graphData.nodes.length} entities · {graphData.links.length} relationships
          </p>
        </div>
        {!noData && (
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search entities..."
                className="pl-8 pr-3 py-1.5 text-xs border border-gray-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 w-48 bg-white"
              />
            </div>
            <div className="flex items-center border border-gray-200 rounded-lg bg-white overflow-hidden">
              <button onClick={handleZoomIn} className="p-1.5 hover:bg-gray-100 transition" title="Zoom in">
                <ZoomIn size={14} className="text-gray-600" />
              </button>
              <button onClick={handleZoomOut} className="p-1.5 hover:bg-gray-100 transition border-x border-gray-200" title="Zoom out">
                <ZoomOut size={14} className="text-gray-600" />
              </button>
              <button onClick={handleFit} className="p-1.5 hover:bg-gray-100 transition" title="Fit to view">
                <Maximize2 size={14} className="text-gray-600" />
              </button>
            </div>
          </div>
        )}
      </div>

      {noData ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-gray-400 text-sm">
            No graph data yet. Upload and process documents to build the knowledge graph.
          </p>
        </div>
      ) : (
        <div className="flex-1 relative">
          <ForceGraph2D
            ref={fgRef}
            graphData={filteredData}
            nodeLabel={(node) => `${node.label || node.id} (${node.group || 'Entity'})`}
            nodeColor={(node) => {
              if (highlightNodes.size > 0 && !highlightNodes.has(node.id)) return '#E5E7EB';
              if (selectedNode?.id === node.id) return '#4F46E5';
              return node.color || colorMap[node.group] || '#9CA3AF';
            }}
            nodeVal={(node) => {
              if (highlightNodes.size > 0 && highlightNodes.has(node.id)) return (node.val || 5) * 1.5;
              return node.val || 5;
            }}
            linkDirectionalArrowLength={3.5}
            linkDirectionalArrowRelPos={1}
            linkCurvature={0.25}
            linkLabel={(link) => link.label || ''}
            linkColor={(link) => {
              if (highlightNodes.size > 0) {
                const srcId = link.source?.id ?? link.source;
                const tgtId = link.target?.id ?? link.target;
                if (highlightNodes.has(srcId) || highlightNodes.has(tgtId)) return '#6366F1';
                return '#F3F4F6';
              }
              return '#D1D5DB';
            }}
            linkWidth={(link) => {
              if (highlightNodes.size > 0) {
                const srcId = link.source?.id ?? link.source;
                const tgtId = link.target?.id ?? link.target;
                if (highlightNodes.has(srcId) || highlightNodes.has(tgtId)) return 2;
              }
              return 1;
            }}
            onNodeClick={handleNodeClick}
            onBackgroundClick={() => setSelectedNode(null)}
            cooldownTicks={100}
            nodeCanvasObjectMode={() => 'after'}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const label = node.label || node.id;
              const fontSize = Math.max(11 / globalScale, 2);
              ctx.font = `${fontSize}px Inter, sans-serif`;
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = highlightNodes.size > 0 && !highlightNodes.has(node.id) ? '#D1D5DB' : '#374151';
              ctx.fillText(label, node.x, node.y + (node.val || 5) + fontSize + 1);
            }}
          />

          {/* Node Detail Panel */}
          {selectedNode && (
            <div className="absolute top-3 right-3 w-64 bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden z-10">
              <div className="flex items-center justify-between px-4 py-2.5 bg-gray-50 border-b border-gray-100">
                <div className="flex items-center gap-2">
                  <Info size={14} className="text-indigo-600" />
                  <span className="text-xs font-semibold text-gray-700">Node Details</span>
                </div>
                <button onClick={() => setSelectedNode(null)} className="text-gray-400 hover:text-gray-600">
                  <X size={14} />
                </button>
              </div>
              <div className="p-4 space-y-3">
                <div>
                  <p className="text-[11px] text-gray-400 font-medium uppercase tracking-wide">Label</p>
                  <p className="text-sm font-semibold text-gray-800">{selectedNode.label || selectedNode.id}</p>
                </div>
                <div>
                  <p className="text-[11px] text-gray-400 font-medium uppercase tracking-wide">Type</p>
                  <span
                    className="inline-block text-xs px-2 py-0.5 rounded-full mt-0.5 font-medium"
                    style={{
                      backgroundColor: (colorMap[selectedNode.group] || '#9CA3AF') + '20',
                      color: colorMap[selectedNode.group] || '#9CA3AF',
                    }}
                  >
                    {selectedNode.group || 'Entity'}
                  </span>
                </div>
                {connectedEdges.length > 0 && (
                  <div>
                    <p className="text-[11px] text-gray-400 font-medium uppercase tracking-wide mb-1">
                      Connections ({connectedEdges.length})
                    </p>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {connectedEdges.slice(0, 10).map((edge, i) => {
                        const srcId = edge.source?.id ?? edge.source;
                        const tgtId = edge.target?.id ?? edge.target;
                        const other = srcId === selectedNode.id ? tgtId : srcId;
                        const otherNode = graphData.nodes.find((n) => n.id === other);
                        return (
                          <div key={i} className="text-xs text-gray-600 flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />
                            <span className="truncate">{edge.label || '→'} {otherNode?.label || other}</span>
                          </div>
                        );
                      })}
                      {connectedEdges.length > 10 && (
                        <p className="text-[11px] text-gray-400">+{connectedEdges.length - 10} more</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-sm rounded-lg border border-gray-200 px-3 py-2 z-10">
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {Object.entries(colorMap).map(([type, color]) => (
                <span key={type} className="flex items-center gap-1 text-[11px] text-gray-500">
                  <span className="w-2.5 h-2.5 rounded-full inline-block flex-shrink-0" style={{ backgroundColor: color }} />
                  {type}
                </span>
              ))}
            </div>
          </div>

          {/* Search highlight count */}
          {search && highlightNodes.size > 0 && (
            <div className="absolute top-3 left-3 bg-indigo-600 text-white text-xs px-3 py-1.5 rounded-lg shadow z-10">
              {highlightNodes.size} match{highlightNodes.size !== 1 ? 'es' : ''} found
            </div>
          )}
        </div>
      )}
    </div>
  );
}
