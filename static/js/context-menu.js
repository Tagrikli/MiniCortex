/**
 * MiniCortex - Context Menu
 * 
 * Handles right-click context menu for the node editor.
 */

import { createNode } from './api.js';
import { fitToView, resetView } from './editor.js';

// ─────────────────────────────────────────────────────────────────────────────
// Context Menu Display
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Show the context menu at the specified position
 * @param {number} x - Screen X coordinate
 * @param {number} y - Screen Y coordinate
 */
export function showContextMenu(x, y) {
    let menu = document.getElementById('context-menu');
    if (!menu) {
        menu = document.createElement('div');
        menu.id = 'context-menu';
        menu.className = 'context-menu';
        document.body.appendChild(menu);
    }
    
    menu.innerHTML = `
        <div class="context-menu-item" onclick="window.contextMenuCreateNode('input_rotating_line')">
            + Add Rotating Line
        </div>
        <div class="context-menu-item" onclick="window.contextMenuCreateNode('input_scanning_square')">
            + Add Scanning Square
        </div>
        <div class="context-menu-item" onclick="window.contextMenuCreateNode('input_mnist')">
            + Add MNIST
        </div>
        <div class="context-menu-item" onclick="window.contextMenuCreateNode('area')">
            + Add Area
        </div>
        <div class="context-menu-divider"></div>
        <div class="context-menu-item" onclick="window.contextMenuFitToView()">
            ⊞ Fit to View
        </div>
        <div class="context-menu-item" onclick="window.contextMenuResetView()">
            ↺ Reset View
        </div>
    `;
    
    // Position menu, ensuring it stays within viewport
    const menuWidth = 150;
    const menuHeight = 200;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    let posX = x;
    let posY = y;
    
    if (x + menuWidth > viewportWidth) {
        posX = viewportWidth - menuWidth - 10;
    }
    if (y + menuHeight > viewportHeight) {
        posY = viewportHeight - menuHeight - 10;
    }
    
    menu.style.left = `${posX}px`;
    menu.style.top = `${posY}px`;
    menu.style.display = 'block';
}

/**
 * Hide the context menu
 */
export function hideContextMenu() {
    const menu = document.getElementById('context-menu');
    if (menu) {
        menu.style.display = 'none';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Context Menu Actions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a node from context menu (at center of view)
 * @param {string} type - Node type
 */
async function contextMenuCreateNode(type) {
    hideContextMenu();
    
    // Import editor state dynamically to avoid circular dependency
    const { editor } = await import('./state.js');
    
    // Calculate position at center of view
    const editorRect = document.getElementById('node-editor').getBoundingClientRect();
    const x = (-editor.pan.x + editorRect.width / 2) / editor.zoom;
    const y = (-editor.pan.y + editorRect.height / 2) / editor.zoom;
    
    await createNode(type, { x, y });
}

/**
 * Fit view to all nodes (wrapper for import)
 */
function contextMenuFitToView() {
    hideContextMenu();
    fitToView();
}

/**
 * Reset view to default (wrapper for import)
 */
function contextMenuResetView() {
    hideContextMenu();
    resetView();
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Registration (for inline event handlers)
// ─────────────────────────────────────────────────────────────────────────────

if (typeof window !== 'undefined') {
    window.contextMenuCreateNode = contextMenuCreateNode;
    window.contextMenuFitToView = contextMenuFitToView;
    window.contextMenuResetView = contextMenuResetView;
}
