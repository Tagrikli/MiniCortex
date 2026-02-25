/**
 * MiniCortex - Connection Management
 *
 * Handles connection rendering, creation, and manipulation.
 *
 * Port positions are calculated from screen-space dot centers and converted
 * to SVG-space coordinates using the connections layer's screen CTM.
 */

import { 
    editor, 
    connections, 
    nodes,
    isPortConnected 
} from './state.js';
import { showUiError } from './utils.js';

// Callbacks to avoid circular dependencies
let createConnectionCallback = null;
let deleteConnectionCallback = null;

/**
 * Set API callbacks (called from editor.js to avoid circular deps)
 */
export function setConnectionApiCallbacks(createConnection, deleteConnection) {
    createConnectionCallback = createConnection;
    deleteConnectionCallback = deleteConnection;
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection Rendering
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render all connections
 */
export function renderConnections() {
    const svg = document.querySelector('#connections-layer svg');
    if (!svg) return;
    
    // Clear existing connections
    svg.innerHTML = '';
    
    // Render each connection
    connections.forEach((conn, index) => {
        const path = createConnectionPathFromData(conn);
        if (path) {
            const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            pathEl.setAttribute('d', path);
            pathEl.classList.add('connection');
            pathEl.dataset.index = index;
            svg.appendChild(pathEl);
        }
    });
}

/**
 * Create SVG path string from connection data
 * @param {Object} conn - Connection object
 * @returns {string|null} SVG path string or null
 */
function createConnectionPathFromData(conn) {
    const fromPos = getPortPosition(conn.from_node, conn.from_output, 'output');
    const toPos = getPortPosition(conn.to_node, conn.to_input, 'input');
    
    if (!fromPos || !toPos) return null;
    
    return createConnectionPath(fromPos.x, fromPos.y, toPos.x, toPos.y);
}

function screenToSvgPoint(screenX, screenY) {
    const svg = document.querySelector('#connections-layer svg');
    if (!svg) return null;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = screenX;
    pt.y = screenY;
    const result = pt.matrixTransform(ctm.inverse());
    return { x: result.x, y: result.y };
}

/**
 * Get the position of a port dot in SVG coordinates.
 *
 * Uses the dot's screen-space bounding box and converts to SVG-space
 * coordinates via the connections layer. This stays accurate even with
 * transforms or layout tweaks.
 *
 * @param {string} nodeId - Node ID
 * @param {string} portName - Port name
 * @param {string} portType - 'input' or 'output'
 * @returns {Object|null} { x, y } position or null
 */
export function getPortPosition(nodeId, portName, portType) {
    const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
    if (!nodeEl) return null;

    const portEl = nodeEl.querySelector(`.port.${portType}[data-port="${portName}"]`);
    if (!portEl) return null;

    const dotEl = portEl.querySelector('.port-dot');
    if (!dotEl) return null;

    const rect = dotEl.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    return screenToSvgPoint(centerX, centerY);
}

/**
 * Create a bezier curve path string
 * @param {number} x1 - Start X
 * @param {number} y1 - Start Y
 * @param {number} x2 - End X
 * @param {number} y2 - End Y
 * @returns {string} SVG path string
 */
export function createConnectionPath(x1, y1, x2, y2) {
    // Create a bezier curve
    const dx = Math.abs(x2 - x1);
    const controlOffset = Math.max(50, dx * 0.5);
    
    return `M ${x1} ${y1} C ${x1 + controlOffset} ${y1}, ${x2 - controlOffset} ${y2}, ${x2} ${y2}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection Creation (Drag)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Start creating a connection from a port
 * @param {MouseEvent} e - Mouse event
 * @param {HTMLElement} portEl - Port element
 */
export function startConnection(e, portEl) {
    const nodeEl = portEl.closest('.node');
    const nodeId = nodeEl.dataset.id;
    const portName = portEl.dataset.port;
    const portType = portEl.dataset.type;  // 'input' or 'output'
    
    // Get port position using the fixed function
    const pos = getPortPosition(nodeId, portName, portType);
    if (!pos) return;
    
    editor.isConnecting = true;
    editor.connectingFrom = {
        nodeId,
        portName,
        portType,
        x: pos.x,
        y: pos.y,
    };
    
    document.getElementById('node-editor').classList.add('connecting');
}

/**
 * Update the connection preview during drag
 * @param {MouseEvent} e - Mouse event
 */
export function updateConnectionPreview(e) {
    if (!editor.connectingFrom) return;
    const mousePos = screenToSvgPoint(e.clientX, e.clientY);
    if (!mousePos) return;
    
    // Create or update preview path
    let previewEl = document.getElementById('connection-preview');
    if (!previewEl) {
        previewEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        previewEl.id = 'connection-preview';
        previewEl.classList.add('connection-preview');
        document.querySelector('#connections-layer svg').appendChild(previewEl);
    }
    
    const path = createConnectionPath(
        editor.connectingFrom.x,
        editor.connectingFrom.y,
        mousePos.x,
        mousePos.y
    );
    previewEl.setAttribute('d', path);
}

/**
 * End connection creation on mouse up
 * @param {MouseEvent} e - Mouse event
 */
export function endConnection(e) {
    editor.isConnecting = false;
    document.getElementById('node-editor').classList.remove('connecting');
    
    // Remove preview
    const previewEl = document.getElementById('connection-preview');
    if (previewEl) {
        previewEl.remove();
    }
    
    // Check if we're over a compatible port
    const portEl = e.target.closest('.port');
    if (portEl) {
        const nodeEl = portEl.closest('.node');
        const toNodeId = nodeEl.dataset.id;
        const toPortName = portEl.dataset.port;
        const toPortType = portEl.dataset.type;
        
        // Validate connection
        if (editor.connectingFrom.nodeId !== toNodeId && 
            editor.connectingFrom.portType !== toPortType) {
            
            // Determine from/to based on port types
            let fromNode, fromOutput, toNode, toInput;
            
            if (editor.connectingFrom.portType === 'output') {
                fromNode = editor.connectingFrom.nodeId;
                fromOutput = editor.connectingFrom.portName;
                toNode = toNodeId;
                toInput = toPortName;
            } else {
                fromNode = toNodeId;
                fromOutput = toPortName;
                toNode = editor.connectingFrom.nodeId;
                toInput = editor.connectingFrom.portName;
            }

            const fromType = getPortDataType(fromNode, fromOutput, 'output');
            const toType = getPortDataType(toNode, toInput, 'input');
            if (!isTypeCompatible(fromType, toType)) {
                showUiError(`Type mismatch: ${fromType || 'any'} → ${toType || 'any'}`);
                editor.connectingFrom = null;
                return;
            }

            // Create connection
            if (createConnectionCallback) {
                createConnectionCallback(fromNode, fromOutput, toNode, toInput);
            }
        }
    }
    
    editor.connectingFrom = null;
}

function getPortDataType(nodeId, portName, portType) {
    const node = nodes.find(n => n.node_id === nodeId);
    if (!node) return null;
    const ports = portType === 'input' ? node.input_ports : node.output_ports;
    if (!Array.isArray(ports)) return null;
    const port = ports.find(p => (p.key || p.name) === portName);
    return port?.data_type || null;
}

function isTypeCompatible(fromType, toType) {
    if (isAnyType(fromType) || isAnyType(toType)) {
        return true;
    }
    return String(fromType) === String(toType);
}

function isAnyType(typeLabel) {
    if (!typeLabel) return true;
    return String(typeLabel).toLowerCase() === 'any';
}

/**
 * Cancel connection creation
 */
export function cancelConnection() {
    editor.isConnecting = false;
    editor.connectingFrom = null;
    document.getElementById('node-editor').classList.remove('connecting');
    
    const previewEl = document.getElementById('connection-preview');
    if (previewEl) {
        previewEl.remove();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection Selection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Select a connection
 * @param {SVGElement} connectionEl - Connection path element
 */
export function selectConnection(connectionEl) {
    // Deselect all first
    document.querySelectorAll('.connection.selected').forEach(el => el.classList.remove('selected'));
    document.querySelectorAll('.node.selected').forEach(el => el.classList.remove('selected'));
    
    const connIndex = parseInt(connectionEl.dataset.index);
    editor.selectedConnection = connIndex;
    editor.selectedNode = null;
    connectionEl.classList.add('selected');
}

/**
 * Delete the selected connection
 */
export async function deleteSelectedConnection() {
    if (editor.selectedConnection === null) return;
    
    const conn = connections[editor.selectedConnection];
    if (!conn) return;
    
    if (deleteConnectionCallback) {
        await deleteConnectionCallback(conn, editor.selectedConnection);
    }
    editor.selectedConnection = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Port Rendering Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render input ports HTML for a node
 * @param {Object} node - Node object
 * @returns {string} HTML string
 */
export function renderInputPorts(node) {
    if (!node.input_ports || node.input_ports.length === 0) {
        return '';
    }
    
    return node.input_ports.map(port => {
        const connected = isPortConnected(node.node_id, port.name, 'input');
        const baseLabel = port.label || port.name;
        const typeLabel = port.data_type ? ` (${port.data_type})` : '';
        const label = `${baseLabel}${typeLabel}`;
        return `
            <div class="port input ${connected ? 'connected' : ''}" 
                 data-node="${node.node_id}" 
                 data-port="${port.name}" 
                 data-type="input">
                <div class="port-dot"></div>
                <span class="port-label">${label}</span>
            </div>
        `;
    }).join('');
}

/**
 * Render output ports HTML for a node
 * @param {Object} node - Node object
 * @returns {string} HTML string
 */
export function renderOutputPorts(node) {
    if (!node.output_ports || node.output_ports.length === 0) {
        return '';
    }
    
    return node.output_ports.map(port => {
        const connected = isPortConnected(node.node_id, port.name, 'output');
        const baseLabel = port.label || port.name;
        const typeLabel = port.data_type ? ` (${port.data_type})` : '';
        const label = `${baseLabel}${typeLabel}`;
        return `
            <div class="port output ${connected ? 'connected' : ''}" 
                 data-node="${node.node_id}" 
                 data-port="${port.name}" 
                 data-type="output">
                <span class="port-label">${label}</span>
                <div class="port-dot"></div>
            </div>
        `;
    }).join('');
}
