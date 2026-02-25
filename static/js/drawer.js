/**
 * MiniCortex - Drawer Component
 * 
 * Implements the left-side drawer with node palette.
 * Fetches palette from /api/palette and enables drag-and-drop node creation.
 */

import { fetchPalette, createNode } from './api.js';
import { editor } from './state.js';

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/** @type {Object|null} Cached palette data */
let palette = null;

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialize the drawer component
 */
export async function initDrawer() {
    try {
        palette = await fetchPalette();
        renderDrawer();
        setupDropZone();
        setupRediscoverButton();
    } catch (error) {
        console.error('Failed to initialize drawer:', error);
    }
}

/**
 * Setup the rediscover button to re-scan for new nodes
 */
function setupRediscoverButton() {
    const btn = document.getElementById('drawer-rediscover-btn');
    if (!btn) return;
    
    btn.addEventListener('click', async () => {
        btn.classList.add('loading');
        btn.disabled = true;
        
        try {
            const response = await fetch('/api/nodes/rediscover', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                throw new Error('Failed to rediscover nodes');
            }
            
            const result = await response.json();
            console.log('Rediscovered nodes:', result);
            
            // Refresh the drawer with new palette
            palette = await fetchPalette();
            renderDrawer();
            
        } catch (error) {
            console.error('Failed to rediscover nodes:', error);
            alert('Failed to rediscover nodes: ' + error.message);
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render the drawer content
 */
export function renderDrawer() {
    const container = document.getElementById('drawer-content');
    if (!container) return;
    
    container.innerHTML = '';
    
    const panels = normalizePalettePanels(palette);
    if (panels.length === 0) {
        container.innerHTML = '<div class="drawer-empty">No nodes available</div>';
        return;
    }
    
    for (const panelDef of panels) {
        const panel = createPanel(panelDef.name, panelDef.nodes || []);
        container.appendChild(panel);
    }
}

/**
 * Normalize palette payloads to a [{ name, nodes }] array.
 * Supports both the new { panels: [...] } shape and legacy object maps.
 * @param {Object|null} paletteData
 * @returns {Array<{name: string, nodes: Array}>}
 */
function normalizePalettePanels(paletteData) {
    if (!paletteData || typeof paletteData !== 'object') {
        return [];
    }
    
    if (Array.isArray(paletteData.panels)) {
        return paletteData.panels.filter(panel => panel && panel.name && Array.isArray(panel.nodes));
    }
    
    return Object.entries(paletteData).map(([name, nodes]) => ({
        name,
        nodes: Array.isArray(nodes) ? nodes : [],
    }));
}

/**
 * Create a collapsible panel
 * @param {string} name - Panel name
 * @param {Array} nodes - Array of node definitions
 * @returns {HTMLElement} Panel element
 */
function createPanel(name, nodes) {
    const panel = document.createElement('div');
    panel.className = 'drawer-panel';
    panel.dataset.category = name.toLowerCase();
    
    panel.innerHTML = `
        <div class="drawer-panel-header">
            <span class="drawer-panel-title">${name}</span>
            <span class="drawer-panel-toggle">▼</span>
        </div>
        <div class="drawer-panel-items">
            ${nodes.map(node => createDrawerItem(node)).join('')}
        </div>
    `;
    
    // Toggle collapse on header click
    const header = panel.querySelector('.drawer-panel-header');
    header.addEventListener('click', () => {
        panel.classList.toggle('collapsed');
    });
    
    // Setup drag for each item
    panel.querySelectorAll('.drawer-item').forEach(item => {
        setupDrag(item);
    });
    
    return panel;
}

/**
 * Create a draggable drawer item
 * @param {Object} node - Node definition { type, name, category }
 * @returns {string} HTML string
 */
function createDrawerItem(node) {
    const iconClass = getCategoryIconClass(node.category);
    const initial = node.name.charAt(0).toUpperCase();
    
    return `
        <div class="drawer-item" draggable="true" data-type="${node.type}">
            <div class="drawer-item-icon ${iconClass}">${initial}</div>
            <span class="drawer-item-name">${node.name}</span>
        </div>
    `;
}

/**
 * Get the icon class based on category
 * @param {string} category - Node category
 * @returns {string} CSS class name
 */
function getCategoryIconClass(category) {
    const categoryMap = {
        'Input': 'input',
        'Utilities': 'utility',
        'Processing': 'processing'
    };
    return categoryMap[category] || 'default';
}

// ─────────────────────────────────────────────────────────────────────────────
// Drag and Drop
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Setup drag behavior for a drawer item
 * @param {HTMLElement} item - Drawer item element
 */
function setupDrag(item) {
    item.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('node-type', item.dataset.type);
        e.dataTransfer.effectAllowed = 'copy';
        
        // Add dragging class for visual feedback
        item.classList.add('dragging');
        
        // Create custom drag image
        const dragImage = document.createElement('div');
        dragImage.className = 'drag-preview';
        dragImage.textContent = item.querySelector('.drawer-item-name').textContent;
        document.body.appendChild(dragImage);
        e.dataTransfer.setDragImage(dragImage, 0, 0);
        
        // Remove drag image after drag starts
        setTimeout(() => dragImage.remove(), 0);
    });
    
    item.addEventListener('dragend', () => {
        item.classList.remove('dragging');
    });
}

/**
 * Setup the drop zone on the editor
 */
export function setupDropZone() {
    const editorEl = document.getElementById('node-editor');
    if (!editorEl) return;
    
    editorEl.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });
    
    editorEl.addEventListener('drop', async (e) => {
        e.preventDefault();
        
        const nodeType = e.dataTransfer.getData('node-type');
        if (!nodeType) return;
        
        // Calculate drop position in canvas coordinates
        const rect = editorEl.getBoundingClientRect();
        const x = (e.clientX - rect.left - editor.pan.x) / editor.zoom;
        const y = (e.clientY - rect.top - editor.pan.y) / editor.zoom;
        
        // Create node at drop position
        await createNode(nodeType, { x, y });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel Controls
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Collapse a specific panel
 * @param {string} panelName - Panel name to collapse
 */
export function collapsePanel(panelName) {
    const panel = document.querySelector(`.drawer-panel[data-category="${panelName.toLowerCase()}"]`);
    if (panel) {
        panel.classList.add('collapsed');
    }
}

/**
 * Expand a specific panel
 * @param {string} panelName - Panel name to expand
 */
export function expandPanel(panelName) {
    const panel = document.querySelector(`.drawer-panel[data-category="${panelName.toLowerCase()}"]`);
    if (panel) {
        panel.classList.remove('collapsed');
    }
}

/**
 * Expand all panels
 */
export function expandAllPanels() {
    document.querySelectorAll('.drawer-panel').forEach(panel => {
        panel.classList.remove('collapsed');
    });
}

/**
 * Collapse all panels
 */
export function collapseAllPanels() {
    document.querySelectorAll('.drawer-panel').forEach(panel => {
        panel.classList.add('collapsed');
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Refresh
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Refresh the drawer content from server
 */
export async function refreshDrawer() {
    palette = await fetchPalette();
    renderDrawer();
}
