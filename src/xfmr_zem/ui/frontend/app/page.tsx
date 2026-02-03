"use client";

import { useCallback, useState, useEffect } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  MarkerType,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Plus, Download, Settings, X } from 'lucide-react';

interface Tool {
  name: string;
  description: string;
}

interface ServerTools {
  [serverName: string]: Tool[];
}

interface NodeData {
  label: React.ReactNode;
  server: string;
  tool: string;
  params: Record<string, any>;
}

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

// Custom Node Component with handles
const CustomNode = ({ data }: { data: NodeData }) => {
  return (
    <div className="px-4 py-3 bg-gradient-to-br from-purple-600 to-violet-700 border-2 border-purple-400 rounded-xl shadow-lg">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-green-400 border-2 border-white" />
      <div className="font-semibold text-sm text-white">{data.server}.{data.tool}</div>
      <div className="text-xs text-purple-200 mt-1">
        {Object.keys(data.params || {}).length} params
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-blue-400 border-2 border-white" />
    </div>
  );
};

const nodeTypes = {
  custom: CustomNode,
};

export default function Home() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [tools, setTools] = useState<ServerTools>({});
  const [selectedTool, setSelectedTool] = useState<{ server: string; tool: Tool } | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node<NodeData> | null>(null);
  const [pipelineName, setPipelineName] = useState('my_pipeline');
  const [nodeParams, setNodeParams] = useState<Record<string, any>>({});

  useEffect(() => {
    fetch('http://localhost:8878/api/tools')
      .then(res => res.json())
      .then(data => {
        console.log('Fetched tools:', data);
        setTools(data || {});
      })
      .catch(err => {
        console.error('Error fetching tools:', err);
        setTools({});
      });
  }, []);

  const onConnect = useCallback(
    (params: Edge | Connection) =>
      setEdges((eds) =>
        addEdge({
          ...params,
          markerEnd: { type: MarkerType.ArrowClosed },
          style: { stroke: '#8b5cf6', strokeWidth: 2 },
        }, eds)
      ),
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node<NodeData>) => {
    setSelectedNode(node);
    setNodeParams(node.data.params || {});
  }, []);

  const addToolNode = () => {
    if (!selectedTool) return;

    const newNodeId = `${Date.now()}`;
    const newNode: Node<NodeData> = {
      id: newNodeId,
      type: 'custom',
      position: { x: 250 + nodes.length * 50, y: 100 + nodes.length * 80 },
      data: {
        label: `${selectedTool.server}.${selectedTool.tool.name}`,
        server: selectedTool.server,
        tool: selectedTool.tool.name,
        params: {},
      },
    };

    setNodes((nds) => [...nds, newNode]);
  };

  const updateNodeParams = () => {
    if (!selectedNode) return;

    setNodes((nds) =>
      nds.map((node) =>
        node.id === selectedNode.id
          ? { ...node, data: { ...node.data, params: nodeParams } }
          : node
      )
    );
  };

  const generateYAML = async () => {
    const pipeline = nodes.map((node) => {
      const stepDef: any = {};
      const toolKey = `${node.data.server}.${node.data.tool}`;

      if (Object.keys(node.data.params || {}).length > 0) {
        stepDef[toolKey] = { input: node.data.params };
      } else {
        stepDef[toolKey] = {};
      }

      return stepDef;
    });

    const config = {
      name: pipelineName,
      parameters: {},
      servers: Object.keys(tools).reduce((acc, server) => ({
        ...acc,
        [server]: `servers/${server}`,
      }), {}),
      pipeline,
    };

    try {
      const response = await fetch('http://localhost:8878/api/generate-yaml', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await response.json();

      const blob = new Blob([data.yaml], { type: 'text/yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${pipelineName}.yaml`;
      a.click();
    } catch (error) {
      console.error('Error generating YAML:', error);
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 flex">
      {/* Sidebar */}
      <div className="w-80 bg-black/40 backdrop-blur-xl border-r border-white/10 p-6 overflow-y-auto">
        <h1 className="text-3xl font-bold text-white mb-2">Zem</h1>
        <p className="text-gray-400 text-sm mb-6">Pipeline Visual Configurator</p>

        <div className="mb-6">
          <label className="text-sm text-gray-300 mb-2 block">Pipeline Name</label>
          <input
            type="text"
            value={pipelineName}
            onChange={(e) => setPipelineName(e.target.value)}
            className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-500"
          />
        </div>

        <h2 className="text-lg font-semibold text-white mb-4">Available Tools</h2>
        {Object.entries(tools).map(([serverName, serverTools]) => (
          <div key={serverName} className="mb-6">
            <h3 className="text-sm font-medium text-purple-300 mb-2">{serverName}</h3>
            <div className="space-y-2">
              {Array.isArray(serverTools) && serverTools.map((tool) => (
                <button
                  key={tool.name}
                  onClick={() => setSelectedTool({ server: serverName, tool })}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-all ${selectedTool?.tool.name === tool.name
                      ? 'bg-purple-600 text-white'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10'
                    }`}
                >
                  <div className="font-medium text-sm">{tool.name}</div>
                  <div className="text-xs opacity-70 mt-1">{tool.description}</div>
                </button>
              ))}
            </div>
          </div>
        ))}

        <div className="mt-8 space-y-3">
          <button
            onClick={addToolNode}
            disabled={!selectedTool}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
          >
            <Plus size={20} />
            Add to Canvas
          </button>
          <button
            onClick={generateYAML}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
          >
            <Download size={20} />
            Export YAML
          </button>
        </div>
      </div>

      {/* Main Canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          fitView
          className="bg-transparent"
        >
          <Background color="#8b5cf6" gap={20} className="opacity-20" />
          <Controls className="bg-black/40 backdrop-blur-xl border border-white/10" />
          <MiniMap
            className="bg-black/40 backdrop-blur-xl border border-white/10"
            nodeColor="#667eea"
          />
        </ReactFlow>
      </div>

      {/* Parameter Panel */}
      {selectedNode && (
        <div className="w-96 bg-black/40 backdrop-blur-xl border-l border-white/10 p-6 overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-xl font-bold text-white">Configure Node</h2>
              <p className="text-sm text-purple-300">{selectedNode.data.server}.{selectedNode.data.tool}</p>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <X size={20} className="text-gray-400" />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-300 mb-2 block">Parameters (JSON)</label>
              <textarea
                value={JSON.stringify(nodeParams, null, 2)}
                onChange={(e) => {
                  try {
                    setNodeParams(JSON.parse(e.target.value));
                  } catch (err) {
                    // Invalid JSON, ignore
                  }
                }}
                className="w-full h-64 px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white font-mono text-sm focus:outline-none focus:border-purple-500"
                placeholder='{"file_path": "data/sample.pdf"}'
              />
            </div>

            <button
              onClick={updateNodeParams}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-colors"
            >
              <Settings size={20} />
              Update Parameters
            </button>
          </div>

          <div className="mt-6 p-4 bg-white/5 rounded-lg">
            <h3 className="text-sm font-semibold text-white mb-2">Current Parameters</h3>
            <pre className="text-xs text-gray-300 overflow-auto">
              {JSON.stringify(selectedNode.data.params, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
