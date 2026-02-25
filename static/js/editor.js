/**
 * MiniCortex - Editor Initialization
 * 
 * Main editor interactions: pan, zoom, event handlers.
 * Initializes all modules and sets up the node editor.
 */

import { 
    editor, 
    MIN_ZOOM, 
    MAX_ZOOM, 
    ZOOM_SPEED, 
    updateEditor 
} from './state.js';
import {
    loadConfig,
    connectWebSocket,
    setRenderCallbacks,
    getNetworkState,
    startNetwork,
    stopNetwork,
    stepNetwork,
    setNetworkSpeed,
    setNetworkStateCallback,
} from './api.js';
import { initDrawer } from './drawer.js';
import { initWorkspaces, setUpdateTransformCallback } from './workspaces.js';
import { 
    startConnection, 
    updateConnectionPreview, 
    endConnection, 
    cancelConnection,
    selectConnection,
    deleteSelectedConnection,
    renderConnections,
    setConnectionApiCallbacks 
} from './connections.js';
import { 
    renderNodes, 
    selectNode, 
    deselectAllNodes, 
    startDraggingNode, 
    handleNodeDrag, 
    endNodeDrag, 
    deleteSelectedNode,
    getNodesBoundingBox,
    setNodeApiCallbacks 
} from './nodes.js';
import { updateOutputs } from './outputs.js';
import { deselectAll, clamp } from './utils.js';
import { showContextMenu, hideContextMenu } from './context-menu.js';
import { updateNodePosition, deleteNode, createConnection, deleteConnection } from './api.js';

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialize the editor
 */
export function initEditor() {
    // Set render callbacks in api.js to avoid circular dependencies
    setRenderCallbacks(renderNodes, renderConnections, updateOutputs);
    
    // Set API callbacks in nodes.js to avoid circular dependencies
    setNodeApiCallbacks(updateNodePosition, deleteNode);
    
    // Set API callbacks in connections.js to avoid circular dependencies
    setConnectionApiCallbacks(createConnection, deleteConnection);
    
    // Set updateTransform callback in workspaces.js to avoid circular dependencies
    setUpdateTransformCallback(updateTransform);
    
    const editorEl = document.getElementById('node-editor');
    const canvasEl = document.getElementById('node-canvas');

    
    // Mouse events for panning
    editorEl.addEventListener('mousedown', onMouseDown);
    editorEl.addEventListener('mousemove', onMouseMove);
    editorEl.addEventListener('mouseup', onMouseUp);
    editorEl.addEventListener('mouseleave', onMouseUp);
    
    // Wheel event for zooming
    editorEl.addEventListener('wheel', onWheel, { passive: false });
    
    // Context menu
    editorEl.addEventListener('contextmenu', onContextMenu);
    
    // Keyboard events
    document.addEventListener('keydown', onKeyDown);
    
    // Zoom controls
    document.getElementById('zoom-in').addEventListener('click', () => setZoom(editor.zoom * 1.2));
    document.getElementById('zoom-out').addEventListener('click', () => setZoom(editor.zoom / 1.2));
    document.getElementById('zoom-reset').addEventListener('click', () => {
        updateEditor({ zoom: 1, pan: { x: 0, y: 0 } });
        updateTransform();
    });
    
    // Close context menu on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.context-menu')) {
            hideContextMenu();
        }
    });
}


/**
 * Initialize the application
 */
export async function initApp() {
    initEditor();
    connectWebSocket();  // Connect immediately, don't wait for REST calls
    await initDrawer();
    await initWorkspaces();
    const config = await loadConfig();
    await initNetworkControls(config?.network);
}

async function initNetworkControls(initialNetworkState = null) {
    const toggleBtn = document.getElementById('network-toggle-btn');
    const iterateBtn = document.getElementById('network-iterate-btn');
    const speedSlider = document.getElementById('network-speed-slider');
    const speedValue = document.getElementById('network-speed-value');
    const actualValue = document.getElementById('network-actual-value');
    if (!toggleBtn || !iterateBtn || !speedSlider || !speedValue || !actualValue) return;
    
    const applyNetworkUi = (network) => {
        if (!network) return;
        const running = Boolean(network.running);
        const speed = Math.round(Number(network.speed) || 10);
        const actual = Number(network.actual_hz) || 0;
        
        toggleBtn.textContent = running ? 'Stop' : 'Start';
        toggleBtn.classList.toggle('running', running);
        speedSlider.value = String(speed);
        speedValue.textContent = String(speed);
        actualValue.textContent = `${actual.toFixed(1)} Hz`;
    };
    
    try {
        applyNetworkUi(initialNetworkState || await getNetworkState());
    } catch (error) {
        console.error('Failed to initialize network controls:', error);
    }

    setNetworkStateCallback(applyNetworkUi);
    
    toggleBtn.addEventListener('click', async () => {
        try {
            const isRunning = toggleBtn.classList.contains('running');
            const result = isRunning ? await stopNetwork() : await startNetwork();
            applyNetworkUi(result.network);
        } catch (error) {
            console.error('Failed to toggle network:', error);
        }
    });

    iterateBtn.addEventListener('click', async () => {
        try {
            const result = await stepNetwork();
            applyNetworkUi(result.network);
        } catch (error) {
            console.error('Failed to iterate network:', error);
        }
    });
    
    speedSlider.addEventListener('input', () => {
        speedValue.textContent = speedSlider.value;
    });
    
    speedSlider.addEventListener('change', async () => {
        try {
            const result = await setNetworkSpeed(parseInt(speedSlider.value, 10));
            applyNetworkUi(result.network);
        } catch (error) {
            console.error('Failed to set network speed:', error);
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Transform & View
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Update the canvas transform based on pan and zoom
 */
export function updateTransform() {
    const canvasEl = document.getElementById('node-canvas');
    canvasEl.style.transform = `translate(${editor.pan.x}px, ${editor.pan.y}px) scale(${editor.zoom})`;
    
    const gridEl = document.querySelector('.node-editor-grid');
    if (gridEl) {
        const minor = 20 * editor.zoom;
        const major = 100 * editor.zoom;
        const position = `${editor.pan.x}px ${editor.pan.y}px`;
        gridEl.style.backgroundSize = `${minor}px ${minor}px, ${minor}px ${minor}px, ${major}px ${major}px, ${major}px ${major}px`;
        gridEl.style.backgroundPosition = `${position}, ${position}, ${position}, ${position}`;
    }
    
    // Update zoom display
    document.getElementById('zoom-level').textContent = `${Math.round(editor.zoom * 100)}%`;
}

/**
 * Set the zoom level
 * @param {number} newZoom - New zoom level
 */
export function setZoom(newZoom) {
    const editorEl = document.getElementById('node-editor');
    const rect = editorEl.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const zoomRatio = newZoom / editor.zoom;
    const pan = {
        x: centerX - (centerX - editor.pan.x) * zoomRatio,
        y: centerY - (centerY - editor.pan.y) * zoomRatio
    };
    
    const zoom = clamp(newZoom, MIN_ZOOM, MAX_ZOOM);
    updateEditor({ zoom, pan });
    updateTransform();
}

/**
 * Fit the view to show all nodes
 */
export function fitToView() {
    const nodes = getNodesBoundingBox();
    if (nodes.width === 0 && nodes.height === 0) return;
    
    const editorEl = document.getElementById('node-editor');
    const editorRect = editorEl.getBoundingClientRect();
    
    // Calculate zoom to fit
    const zoomX = editorRect.width / (nodes.width + 100);
    const zoomY = editorRect.height / (nodes.height + 100);
    const zoom = Math.min(zoomX, zoomY, 1);
    
    // Center the view
    const pan = {
        x: (editorRect.width - nodes.width * zoom) / 2 - nodes.minX * zoom,
        y: (editorRect.height - nodes.height * zoom) / 2 - nodes.minY * zoom
    };
    
    updateEditor({ zoom, pan });
    updateTransform();
}

/**
 * Reset the view to default
 */
export function resetView() {
    updateEditor({ zoom: 1, pan: { x: 0, y: 0 } });
    updateTransform();
}

// ─────────────────────────────────────────────────────────────────────────────
// Mouse Event Handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Handle mouse down event
 * @param {MouseEvent} e - Mouse event
 */
function onMouseDown(e) {
    const editorEl = document.getElementById('node-editor');
    
    // Left mouse button
    if (e.button === 0) {
        // Check if clicking on a port
        const portEl = e.target.closest('.port');
        if (portEl) {
            startConnection(e, portEl);
            return;
        }
        
        // Check if clicking on a node header (for dragging)
        const nodeHeader = e.target.closest('.node-header');
        if (nodeHeader) {
            const nodeEl = nodeHeader.closest('.node');
            startDraggingNode(e, nodeEl);
            return;
        }
        
        // Check if clicking on a connection
        const connectionEl = e.target.closest('.connection');
        if (connectionEl) {
            selectConnection(connectionEl);
            return;
        }
        
        // Ignore clicks inside node content (properties/actions/outputs)
        if (e.target.closest('.node')) {
            return;
        }
        
        // Left-drag on empty canvas pans the view
        deselectAllNodes();
        updateEditor({
            isPanning: true,
            panStart: { x: e.clientX - editor.pan.x, y: e.clientY - editor.pan.y }
        });
        editorEl.classList.add('grabbing');
    }
}

/**
 * Handle mouse move event
 * @param {MouseEvent} e - Mouse event
 */
function onMouseMove(e) {
    const editorEl = document.getElementById('node-editor');
    
    // Panning
    if (editor.isPanning) {
        const pan = {
            x: e.clientX - editor.panStart.x,
            y: e.clientY - editor.panStart.y
        };
        updateEditor({ pan });
        updateTransform();
        return;
    }
    
    // Dragging node
    if (editor.isDragging && editor.draggedNode) {
        handleNodeDrag(e);
        return;
    }
    
    // Connecting
    if (editor.isConnecting) {
        updateConnectionPreview(e);
    }
}

/**
 * Handle mouse up event
 * @param {MouseEvent} e - Mouse event
 */
function onMouseUp(e) {
    const editorEl = document.getElementById('node-editor');
    
    if (editor.isPanning) {
        updateEditor({ isPanning: false });
        editorEl.classList.remove('grabbing');
    }
    
    if (editor.isDragging) {
        endNodeDrag();
    }
    
    if (editor.isConnecting) {
        endConnection(e);
    }
}

/**
 * Handle wheel event for zooming
 * @param {WheelEvent} e - Wheel event
 */
function onWheel(e) {
    e.preventDefault();
    
    const editorEl = document.getElementById('node-editor');
    const rect = editorEl.getBoundingClientRect();
    
    // Get mouse position relative to editor
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Calculate zoom
    const delta = -e.deltaY * ZOOM_SPEED;
    const newZoom = clamp(editor.zoom * (1 + delta), MIN_ZOOM, MAX_ZOOM);
    
    // Adjust pan to zoom towards mouse position
    const zoomRatio = newZoom / editor.zoom;
    const pan = {
        x: mouseX - (mouseX - editor.pan.x) * zoomRatio,
        y: mouseY - (mouseY - editor.pan.y) * zoomRatio
    };
    
    updateEditor({ zoom: newZoom, pan });
    updateTransform();
}

/**
 * Handle key down event
 * @param {KeyboardEvent} e - Keyboard event
 */
function onKeyDown(e) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) {
        return;
    }

    // Delete selected node or connection
    if (e.key === 'Delete' || e.key === 'Backspace' || e.key.toLowerCase() === 'x') {
        if (editor.selectedConnection !== null) {
            deleteSelectedConnection();
        } else if (editor.selectedNode) {
            deleteSelectedNode();
        }
    }
    
    // Escape to cancel actions
    if (e.key === 'Escape') {
        cancelConnection();
        deselectAllNodes();
        hideContextMenu();
    }

    if (e.key === ' ') {
        e.preventDefault();
        const toggleBtn = document.getElementById('network-toggle-btn');
        if (toggleBtn) toggleBtn.click();
    }

    if (e.key === 'ArrowRight') {
        e.preventDefault();
        const iterateBtn = document.getElementById('network-iterate-btn');
        if (iterateBtn) iterateBtn.click();
    }
}

/**
 * Handle context menu event
 * @param {MouseEvent} e - Mouse event
 */
function onContextMenu(e) {
    e.preventDefault();
    showContextMenu(e.clientX, e.clientY);
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for other modules
// ─────────────────────────────────────────────────────────────────────────────

export { renderNodes };
