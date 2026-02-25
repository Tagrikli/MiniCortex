/**
 * MiniCortex - State Management
 * 
 * Centralized state for the node editor application.
 */

// ─────────────────────────────────────────────────────────────────────────────
// State Variables
// ─────────────────────────────────────────────────────────────────────────────

/** @type {Array} List of all nodes in the editor */
export let nodes = [];

/** @type {Array} list of all connections between nodes */
export let connections = [];

/** @type {WebSocket|null} WebSocket connection for real-time updates */
export let ws = null;

/** @type {Object} Map of node_id -> { output_key -> { canvas, ctx, type } } */
export let outputCanvases = {};

// ─────────────────────────────────────────────────────────────────────────────
// Editor State
// ─────────────────────────────────────────────────────────────────────────────

export let editor = {
    pan: { x: 0, y: 0 },
    zoom: 1,
    isPanning: false,
    panStart: { x: 0, y: 0 },
    isDragging: false,
    draggedNode: null,
    dragOffset: { x: 0, y: 0 },
    isConnecting: false,
    connectingFrom: null,  // { nodeId, portName, portType, x, y }
    connectionPreview: null,
    selectedNode: null,
    selectedConnection: null,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 3;
export const ZOOM_SPEED = 0.001;
export const GRID_SIZE = 20;

// ─────────────────────────────────────────────────────────────────────────────
// State Mutation Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Set the entire nodes array
 * @param {Array} newNodes - New nodes array
 */
export function setNodes(newNodes) {
    nodes = newNodes;
}

/**
 * Add a single node to the nodes array
 * @param {Object} node - Node object to add
 */
export function addNode(node) {
    nodes.push(node);
}

/**
 * Remove a node by ID
 * @param {string} nodeId - ID of node to remove
 */
export function removeNode(nodeId) {
    nodes = nodes.filter(n => n.node_id !== nodeId);
}

/**
 * Update a specific node's data
 * @param {string} nodeId - ID of node to update
 * @param {Object} updates - Partial node data to merge
 */
export function updateNode(nodeId, updates) {
    const index = nodes.findIndex(n => n.node_id === nodeId);
    if (index !== -1) {
        nodes[index] = { ...nodes[index], ...updates };
    }
}

/**
 * Set the entire connections array
 * @param {Array} newConnections - New connections array
 */
export function setConnections(newConnections) {
    connections = newConnections;
}

/**
 * Add a connection
 * @param {Object} connection - Connection object to add
 */
export function addConnection(connection) {
    connections.push(connection);
}

/**
 * Remove a connection by index
 * @param {number} index - Index of connection to remove
 */
export function removeConnection(index) {
    connections.splice(index, 1);
}

/**
 * Remove all connections for a specific node
 * @param {string} nodeId - ID of node
 */
export function removeConnectionsForNode(nodeId) {
    connections = connections.filter(c => c.from_node !== nodeId && c.to_node !== nodeId);
}

/**
 * Set the WebSocket connection
 * @param {WebSocket|null} websocket - WebSocket instance
 */
export function setWebSocket(websocket) {
    ws = websocket;
}

/**
 * Initialize output canvases for a node
 * @param {string} nodeId - Node ID
 */
export function initOutputCanvases(nodeId) {
    if (!outputCanvases[nodeId]) {
        outputCanvases[nodeId] = {};
    }
}

/**
 * Set canvas info for a specific output
 * @param {string} nodeId - Node ID
 * @param {string} outputKey - Output key
 * @param {Object} info - { canvas, ctx, type }
 */
export function setOutputCanvas(nodeId, outputKey, info) {
    initOutputCanvases(nodeId);
    outputCanvases[nodeId][outputKey] = info;
}

/**
 * Get canvas info for a specific output
 * @param {string} nodeId - Node ID
 * @param {string} outputKey - Output key
 * @returns {Object|null} Canvas info or null
 */
export function getOutputCanvas(nodeId, outputKey) {
    return outputCanvases[nodeId]?.[outputKey] || null;
}

/**
 * Clear all output canvases
 */
export function clearOutputCanvases() {
    outputCanvases = {};
}

/**
 * Update editor state
 * @param {Object} updates - Partial editor state to merge
 */
export function updateEditor(updates) {
    Object.assign(editor, updates);
}

/**
 * Reset editor state to defaults
 */
export function resetEditor() {
    editor = {
        pan: { x: 0, y: 0 },
        zoom: 1,
        isPanning: false,
        panStart: { x: 0, y: 0 },
        isDragging: false,
        draggedNode: null,
        dragOffset: { x: 0, y: 0 },
        isConnecting: false,
        connectingFrom: null,
        connectionPreview: null,
        selectedNode: null,
        selectedConnection: null,
    };
}

/**
 * Find a node by ID
 * @param {string} nodeId - Node ID
 * @returns {Object|undefined} Node object or undefined
 */
export function findNode(nodeId) {
    return nodes.find(n => n.node_id === nodeId);
}

/**
 * Check if a port is connected
 * @param {string} nodeId - Node ID
 * @param {string} portName - Port name
 * @param {string} portType - 'input' or 'output'
 * @returns {boolean} True if connected
 */
export function isPortConnected(nodeId, portName, portType) {
    return connections.some(conn => {
        if (portType === 'input') {
            return conn.to_node === nodeId && conn.to_input === portName;
        } else {
            return conn.from_node === nodeId && conn.from_output === portName;
        }
    });
}
