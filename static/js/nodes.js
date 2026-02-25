/**
 * MiniCortex - Node Rendering
 * 
 * Handles node creation, rendering, and manipulation.
 */

import { nodes, editor, updateEditor, isPortConnected } from './state.js';
import { renderConnections } from './connections.js';
import { renderProperties, renderActions } from './properties.js';
import { renderOutputs, initializeCanvases, clearOutputCanvases, reapplyCachedOutputs } from './outputs.js';
import { deselectAll } from './utils.js';

// Callbacks to avoid circular dependencies
let updateNodePositionCallback = null;
let deleteNodeCallback = null;

/**
 * Set API callbacks (called from editor.js to avoid circular deps)
 */
export function setNodeApiCallbacks(updateNodePosition, deleteNode) {
    updateNodePositionCallback = updateNodePosition;
    deleteNodeCallback = deleteNode;
}

// ─────────────────────────────────────────────────────────────────────────────
// Node Rendering
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render all nodes to the canvas
 */
export function renderNodes() {
    const canvasEl = document.getElementById('node-canvas');
    canvasEl.innerHTML = '';
    clearOutputCanvases();
    
    // Create SVG layer for connections
    const svgLayer = document.createElement('div');
    svgLayer.id = 'connections-layer';
    svgLayer.className = 'connections-layer';
    svgLayer.innerHTML = '<svg></svg>';
    canvasEl.appendChild(svgLayer);
    
    // Render each node
    for (const node of nodes) {
        const nodeEl = createNodeElement(node);
        canvasEl.appendChild(nodeEl);
    }
    
    renderConnections();
    reapplyCachedOutputs();
}

/**
 * Create a node DOM element
 * @param {Object} node - Node object
 * @returns {HTMLElement} Node element
 */
export function createNodeElement(node) {
    const nodeEl = document.createElement('div');
    nodeEl.className = 'node';
    nodeEl.dataset.type = node.node_type;
    nodeEl.dataset.id = node.node_id;
    
    // Position
    nodeEl.style.left = `${node.position.x}px`;
    nodeEl.style.top = `${node.position.y}px`;
    
    // Get node title
    const title = node.name || node.node_type.charAt(0).toUpperCase() + node.node_type.slice(1);
    
    // Build combined I/O rows HTML (inputs/outputs share the same rows)
    const ioRowsHtml = renderIoRows(node);
    
    // Build properties HTML
    const propertiesHtml = renderProperties(node);
    
    // Build actions HTML
    const actionsHtml = renderActions(node);
    
    // Build outputs HTML
    const outputsHtml = renderOutputs(node);
    
    // Build dynamic node refresh button
    const dynamicButton = node.dynamic ? `
        <button class="node-reload-btn" data-node-id="${node.node_id}" title="Reload Node Code">
            ↻
        </button>
    ` : '';
    
    nodeEl.innerHTML = `
        <div class="node-header">
            <span class="node-title">${title}</span>
            ${dynamicButton}
        </div>
        <div class="node-body">
            ${ioRowsHtml}
            <div class="node-content">
                ${propertiesHtml}
                ${actionsHtml}
                ${outputsHtml}
            </div>
        </div>
    `;
    
    // Initialize canvases for this node
    initializeCanvases(node, nodeEl);
    
    // Attach reload button handler
    if (node.dynamic) {
        const reloadBtn = nodeEl.querySelector('.node-reload-btn');
        if (reloadBtn) {
            reloadBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                reloadNode(node.node_id);
            });
        }
    }
    
    return nodeEl;
}

/**
 * Render shared input/output rows for a node.
 * Inputs are shown on the left, outputs on the right, aligned by row index.
 * @param {Object} node
 * @returns {string}
 */
function renderIoRows(node) {
    const inputs = node.input_ports || [];
    const outputs = node.output_ports || [];
    const hasPorts = inputs.length > 0 || outputs.length > 0;
    
    if (!hasPorts) {
        return '';
    }
    
    const inputCells = inputs.map(port => `
        <div class="node-io-cell left">
            ${renderPort(node.node_id, port, 'input')}
        </div>
    `).join('');
    
    const outputCells = outputs.map(port => `
        <div class="node-io-cell right">
            ${renderPort(node.node_id, port, 'output')}
        </div>
    `).join('');
    
    return `
        <div class="node-io">
            <div class="node-io-column left">
                ${inputCells}
            </div>
            <div class="node-io-column right">
                ${outputCells}
            </div>
        </div>
    `;
}

/**
 * Render a single input/output port.
 * @param {string} nodeId
 * @param {Object} port
 * @param {'input'|'output'} type
 * @returns {string}
 */
function renderPort(nodeId, port, type) {
    const portKey = port.key || port.name;
    const connected = isPortConnected(nodeId, portKey, type);
    const baseLabel = port.label || port.name;
    const typeLabel = port.data_type ? ` (${port.data_type})` : '';
    const label = `${baseLabel}${typeLabel}`;
    
    if (type === 'input') {
        return `
            <div class="port input ${connected ? 'connected' : ''}" 
                 data-node="${nodeId}" 
                 data-port="${portKey}" 
                 data-type="input">
                <div class="port-dot"></div>
                <span class="port-label">${label}</span>
            </div>
        `;
    }
    
    return `
        <div class="port output ${connected ? 'connected' : ''}" 
             data-node="${nodeId}" 
             data-port="${portKey}" 
             data-type="output">
            <span class="port-label">${label}</span>
            <div class="port-dot"></div>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Node Selection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Select a node
 * @param {string} nodeId - Node ID
 */
export function selectNode(nodeId) {
    deselectAll();
    updateEditor({ selectedNode: nodeId });
    
    const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
    if (nodeEl) {
        nodeEl.classList.add('selected');
    }
}

/**
 * Deselect all nodes and connections
 */
export function deselectAllNodes() {
    updateEditor({ selectedNode: null, selectedConnection: null });
    deselectAll();
}

// ─────────────────────────────────────────────────────────────────────────────
// Node Dragging
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Start dragging a node
 * @param {MouseEvent} e - Mouse event
 * @param {HTMLElement} nodeEl - Node element
 */
export function startDraggingNode(e, nodeEl) {
    const nodeId = nodeEl.dataset.id;
    const node = nodes.find(n => n.node_id === nodeId);
    
    if (!node) return;
    
    updateEditor({
        isDragging: true,
        draggedNode: nodeId,
        selectedNode: nodeId
    });
    
    // Calculate offset from mouse to node position
    const editorEl = document.getElementById('node-editor');
    const rect = editorEl.getBoundingClientRect();
    const dragOffset = {
        x: (e.clientX - rect.left - editor.pan.x) / editor.zoom - node.position.x,
        y: (e.clientY - rect.top - editor.pan.y) / editor.zoom - node.position.y
    };
    updateEditor({ dragOffset });
    
    // Select node
    selectNode(nodeId);
}

/**
 * Handle node dragging
 * @param {MouseEvent} e - Mouse event
 */
export function handleNodeDrag(e) {
    if (!editor.isDragging || !editor.draggedNode) return;
    
    const nodeEl = document.querySelector(`.node[data-id="${editor.draggedNode}"]`);
    if (!nodeEl) return;
    
    const editorEl = document.getElementById('node-editor');
    const rect = editorEl.getBoundingClientRect();
    
    const x = (e.clientX - rect.left - editor.pan.x) / editor.zoom - editor.dragOffset.x;
    const y = (e.clientY - rect.top - editor.pan.y) / editor.zoom - editor.dragOffset.y;
    
    nodeEl.style.left = `${x}px`;
    nodeEl.style.top = `${y}px`;
    
    // Update node data
    const node = nodes.find(n => n.node_id === editor.draggedNode);
    if (node) {
        node.position = { x, y };
    }
    
    // Update connections
    renderConnections();
}

/**
 * End node dragging
 */
export function endNodeDrag() {
    if (editor.isDragging && editor.selectedNode) {
        // Send position update to server
        const node = nodes.find(n => n.node_id === editor.selectedNode);
        if (node && updateNodePositionCallback) {
            updateNodePositionCallback(editor.selectedNode, node.position);
        }
    }
    
    updateEditor({
        isDragging: false,
        draggedNode: null
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Node Deletion
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Delete the selected node
 */
export async function deleteSelectedNode() {
    if (!editor.selectedNode) return;
    
    if (deleteNodeCallback) {
        await deleteNodeCallback(editor.selectedNode);
    }
    updateEditor({ selectedNode: null });
}

// ─────────────────────────────────────────────────────────────────────────────
// Node Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get bounding box of all nodes
 * @returns {Object} { minX, minY, maxX, maxY, width, height }
 */
export function getNodesBoundingBox() {
    if (nodes.length === 0) {
        return { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0 };
    }
    
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    for (const node of nodes) {
        minX = Math.min(minX, node.position.x);
        minY = Math.min(minY, node.position.y);
        maxX = Math.max(maxX, node.position.x + 200);  // Approximate node width
        maxY = Math.max(maxY, node.position.y + 150);  // Approximate node height
    }
    
    return {
        minX,
        minY,
        maxX,
        maxY,
        width: maxX - minX,
        height: maxY - minY
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic Node Reload
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reload a dynamic node's code from the server
 * @param {string} nodeId - Node ID to reload
 */
async function reloadNode(nodeId) {
    try {
        const response = await fetch(`/api/nodes/${nodeId}/reload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to reload node');
        }
        
        const result = await response.json();
        console.log('Node reloaded:', result.message);
        
        // Update the node in the local state
        const nodeIndex = nodes.findIndex(n => n.node_id === nodeId);
        if (nodeIndex !== -1 && result.node) {
            nodes[nodeIndex] = result.node;
        }
        
        // Re-render nodes
        renderNodes();
        
    } catch (error) {
        console.error('Failed to reload node:', error);
        alert(`Failed to reload node: ${error.message}`);
    }
}
