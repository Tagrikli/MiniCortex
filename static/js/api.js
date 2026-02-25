/**
 * MiniCortex - API Communication
 * 
 * Handles all API calls and WebSocket communication.
 */

import { 
    setNodes, 
    setConnections, 
    setWebSocket,
    editor,
    updateEditor,
    addNode,
    removeNode,
    removeConnectionsForNode,
    addConnection,
    removeConnection
} from './state.js';
import { updateConnectionStatus, showUiError } from './utils.js';

// Callbacks to avoid circular dependencies
let renderNodesCallback = null;
let renderConnectionsCallback = null;
let updateOutputsCallback = null;
let updateNetworkStateCallback = null;

/**
 * Set render callbacks (called from editor.js to avoid circular deps)
 */
export function setRenderCallbacks(renderNodes, renderConnections, updateOutputs) {
    renderNodesCallback = renderNodes;
    renderConnectionsCallback = renderConnections;
    updateOutputsCallback = updateOutputs;
}

export function setNetworkStateCallback(callback) {
    updateNetworkStateCallback = callback;
}

function getViewportPayload() {
    return {
        pan: {
            x: editor.pan.x,
            y: editor.pan.y,
        },
        zoom: editor.zoom,
    };
}

function applyTopologySnapshot(snapshot) {
    if (!snapshot) return false;
    
    if (Array.isArray(snapshot.nodes)) {
        setNodes(snapshot.nodes);
    }
    if (Array.isArray(snapshot.connections)) {
        setConnections(snapshot.connections);
    }
    if (snapshot.viewport?.pan && typeof snapshot.viewport.zoom === 'number') {
        updateEditor({
            pan: {
                x: snapshot.viewport.pan.x,
                y: snapshot.viewport.pan.y,
            },
            zoom: snapshot.viewport.zoom,
        });
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Load initial configuration from server
 * @returns {Promise<Object>} Configuration object
 */
export async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        if (!applyTopologySnapshot(config)) {
            setNodes(config.nodes || []);
            setConnections(config.connections || []);
        }
        if (renderNodesCallback) renderNodesCallback();
        if (renderConnectionsCallback) renderConnectionsCallback();
        // Don't update connection status here - it's managed by WebSocket
        return config;
    } catch (error) {
        console.error('Failed to load config:', error);
        throw error;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch node schema from server
 * @param {string} nodeType - Type of node
 * @returns {Promise<Object>} Node schema
 */
export async function fetchNodeSchema(nodeType) {
    const response = await fetch(`/api/nodes/schema/${nodeType}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch schema for ${nodeType}`);
    }
    return response.json();
}

/**
 * Create a new node
 * @param {string} type - Node type
 * @param {Object} position - { x, y } position
 * @returns {Promise<Object>} Created node
 */
export async function createNode(type, position) {
    try {
        const response = await fetch('/api/nodes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: type,
                position: position,
                viewport: getViewportPayload(),
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to create node');
        }
        
        const data = await response.json();
        if (applyTopologySnapshot(data.snapshot)) {
            if (renderNodesCallback) renderNodesCallback();
            if (renderConnectionsCallback) renderConnectionsCallback();
        } else if (data.node) {
            addNode(data.node);
            if (renderNodesCallback) renderNodesCallback();
        }
        return data.node;
    } catch (error) {
        console.error('Failed to create node:', error);
        throw error;
    }
}

/**
 * Update a node's property
 * @param {string} nodeId - Node ID
 * @param {string} propKey - Property key
 * @param {*} value - New value
 */
export async function setProperty(nodeId, propKey, value) {
    try {
        await fetch(`/api/nodes/${nodeId}/properties/${propKey}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value })
        });
    } catch (error) {
        console.error('Failed to set property:', error);
    }
}

/**
 * Update a node's position
 * @param {string} nodeId - Node ID
 * @param {Object} position - { x, y } position
 */
export async function updateNodePosition(nodeId, position) {
    try {
        await fetch(`/api/nodes/${nodeId}/position`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(position)
        });
    } catch (error) {
        console.error('Failed to update position:', error);
    }
}

/**
 * Delete a node
 * @param {string} nodeId - Node ID
 */
export async function deleteNode(nodeId) {
    try {
        const response = await fetch(`/api/nodes/${nodeId}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                viewport: getViewportPayload(),
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete node');
        }
        
        const data = await response.json();
        
        if (applyTopologySnapshot(data.snapshot)) {
            if (renderNodesCallback) renderNodesCallback();
            if (renderConnectionsCallback) renderConnectionsCallback();
        } else {
            removeNode(nodeId);
            removeConnectionsForNode(nodeId);
            if (renderNodesCallback) renderNodesCallback();
        }
    } catch (error) {
        console.error('Failed to delete node:', error);
    }
}

/**
 * Execute a node action
 * @param {string} nodeId - Node ID
 * @param {string} actionKey - Action key
 * @param {Object} params - Action parameters
 */
export async function executeAction(nodeId, actionKey, params = {}) {
    try {
        await fetch(`/api/nodes/${nodeId}/actions/${actionKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ params })
        });
    } catch (error) {
        console.error('Action failed:', error);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a connection between nodes
 * @param {string} fromNode - Source node ID
 * @param {string} fromOutput - Source output port name
 * @param {string} toNode - Target node ID
 * @param {string} toInput - Target input port name
 */
export async function createConnection(fromNode, fromOutput, toNode, toInput) {
    try {
        const response = await fetch('/api/connections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                from_node: fromNode,
                from_output: fromOutput,
                to_node: toNode,
                to_input: toInput,
                viewport: getViewportPayload(),
            })
        });
        
        if (!response.ok) {
            const errText = await response.text();
            let detail = errText;
            try {
                const parsed = JSON.parse(errText);
                detail = parsed?.detail || errText;
            } catch {
                // ignore parse errors
            }
            throw new Error(detail || 'Failed to create connection');
        }

        const data = await response.json();
        if (applyTopologySnapshot(data.snapshot)) {
            if (renderNodesCallback) renderNodesCallback();
            if (renderConnectionsCallback) renderConnectionsCallback();
        } else {
            addConnection({
                from_node: fromNode,
                from_output: fromOutput,
                to_node: toNode,
                to_input: toInput,
            });
            
            if (renderConnectionsCallback) renderConnectionsCallback();
            if (renderNodesCallback) renderNodesCallback();  // Re-render to update port states
        }
    } catch (error) {
        console.error('Failed to create connection:', error);
        if (showUiError) {
            showUiError(error?.message || 'Connection failed');
        }
    }
}

/**
 * Delete a connection
 * @param {Object} connection - Connection object
 * @param {number} index - Connection index in array
 */
export async function deleteConnection(connection, index) {
    try {
        const response = await fetch('/api/connections', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                from_node: connection.from_node,
                from_output: connection.from_output,
                to_node: connection.to_node,
                to_input: connection.to_input,
                viewport: getViewportPayload(),
            })
        });
        
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText || 'Failed to delete connection');
        }
        
        const data = await response.json();
        if (applyTopologySnapshot(data.snapshot)) {
            if (renderNodesCallback) renderNodesCallback();
            if (renderConnectionsCallback) renderConnectionsCallback();
        } else {
            removeConnection(index);
            if (renderConnectionsCallback) renderConnectionsCallback();
            if (renderNodesCallback) renderNodesCallback();  // Re-render to update port states
        }
    } catch (error) {
        console.error('Failed to delete connection:', error);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Palette API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch the node palette configuration
 * @returns {Promise<Object>} Palette object with categories
 */
export async function fetchPalette() {
    try {
        const response = await fetch('/api/palette');
        if (!response.ok) {
            throw new Error('Failed to fetch palette');
        }
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch palette:', error);
        return {};
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Network Runtime Controls API
// ─────────────────────────────────────────────────────────────────────────────

export async function getNetworkState() {
    const response = await fetch('/api/network');
    if (!response.ok) {
        throw new Error('Failed to fetch network state');
    }
    return response.json();
}

export async function startNetwork() {
    const response = await fetch('/api/network/start', { method: 'POST' });
    if (!response.ok) {
        throw new Error('Failed to start network');
    }
    return response.json();
}

export async function stopNetwork() {
    const response = await fetch('/api/network/stop', { method: 'POST' });
    if (!response.ok) {
        throw new Error('Failed to stop network');
    }
    return response.json();
}

export async function stepNetwork() {
    const response = await fetch('/api/network/step', { method: 'POST' });
    if (!response.ok) {
        throw new Error('Failed to step network');
    }
    return response.json();
}

export async function setNetworkSpeed(speed) {
    const response = await fetch('/api/network/speed', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed }),
    });
    if (!response.ok) {
        throw new Error('Failed to set network speed');
    }
    return response.json();
}

export async function setOutputEnabled(nodeId, outputKey, enabled) {
    const response = await fetch(`/api/nodes/${nodeId}/outputs/${outputKey}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
    });
    if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'Failed to update output');
    }
    return response.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Connect to WebSocket for real-time updates
 */
export function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const websocket = new WebSocket(`${protocol}//${window.location.host}/ws`);
    
    // Set initial status to connecting
    updateConnectionStatus('connecting');
    
    websocket.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus('connected');
    };
    
    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus('disconnected');
        // Attempt to reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('disconnected');
    };
    
    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'state') {
                if (updateOutputsCallback) updateOutputsCallback(data.nodes);
                if (data.network && updateNetworkStateCallback) {
                    updateNetworkStateCallback(data.network);
                }
            } else if (data.type === 'error') {
                // Handle node processing error
                console.error(`Node error in "${data.node_name}":`, data.error);
                console.error('Traceback:', data.traceback);
                
                // Show error notification to user
                showUiError(`Node "${data.node_name}" error: ${data.error}`);
                
                // Update network state if provided
                if (data.network && updateNetworkStateCallback) {
                    updateNetworkStateCallback(data.network);
                }
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    };
    
    setWebSocket(websocket);
}
