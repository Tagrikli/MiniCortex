/**
 * MiniCortex - Utility Functions
 * 
 * Common utility functions used across the application.
 */

// ─────────────────────────────────────────────────────────────────────────────
// ID Generation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate a unique ID
 * @returns {string} Unique ID string
 */
export function generateId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Object Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Deep clone an object
 * @param {Object} obj - Object to clone
 * @returns {Object} Deep cloned object
 */
export function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Format a value based on scale type
 * @param {number} value - Value to format
 * @param {string} scale - Scale type ('log' or 'linear')
 * @returns {string} Formatted value string
 */
export function formatValue(value, scale) {
    if (scale === 'log') {
        return value.toFixed(6);
    }
    return value.toFixed(3);
}

/**
 * Format a numeric value based on format string
 * @param {number|string} value - Value to format
 * @param {string} format - Format string (e.g., '.4f', 'd')
 * @returns {string} Formatted value string
 */
export function formatNumeric(value, format) {
    if (typeof value !== 'number') return String(value);
    
    if (format === 'd') {
        return Math.round(value).toString();
    }
    
    const match = format.match(/^\.(\d+)f$/);
    if (match) {
        return value.toFixed(parseInt(match[1]));
    }
    
    return value.toString();
}

// ─────────────────────────────────────────────────────────────────────────────
// DOM Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Deselect all nodes and connections in the UI
 */
export function deselectAll() {
    document.querySelectorAll('.node.selected').forEach(el => el.classList.remove('selected'));
    document.querySelectorAll('.connection.selected').forEach(el => el.classList.remove('selected'));
}

/**
 * Update connection status indicator
 * @param {string} status - 'connected', 'disconnected', or 'connecting'
 */
export function updateConnectionStatus(status) {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;
    
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');
    
    dot.className = 'status-dot';
    if (status === 'connected') {
        dot.classList.add('connected');
        text.textContent = 'Connected';
    } else if (status === 'disconnected') {
        dot.classList.add('disconnected');
        text.textContent = 'Disconnected';
    } else {
        // connecting state - use warning color (default)
        dot.classList.add('connecting');
        text.textContent = 'Connecting...';
    }
}

export function showUiError(message) {
    let el = document.getElementById('ui-error');
    if (!el) {
        el = document.createElement('div');
        el.id = 'ui-error';
        el.className = 'ui-error';
        document.body.appendChild(el);
    }
    el.textContent = message;
    el.classList.add('visible');
    if (el._hideTimeout) {
        clearTimeout(el._hideTimeout);
    }
    el._hideTimeout = setTimeout(() => {
        el.classList.remove('visible');
    }, 2500);
}

// ─────────────────────────────────────────────────────────────────────────────
// Math Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Clamp a value between min and max
 * @param {number} value - Value to clamp
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} Clamped value
 */
export function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

/**
 * Linear interpolation between two values
 * @param {number} a - Start value
 * @param {number} b - End value
 * @param {number} t - Interpolation factor (0-1)
 * @returns {number} Interpolated value
 */
export function lerp(a, b, t) {
    return a + (b - a) * t;
}

/**
 * Snap a value to grid
 * @param {number} value - Value to snap
 * @param {number} gridSize - Grid size
 * @returns {number} Snapped value
 */
export function snapToGrid(value, gridSize) {
    return Math.round(value / gridSize) * gridSize;
}

// ─────────────────────────────────────────────────────────────────────────────
// Event Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get mouse position relative to an element
 * @param {MouseEvent} e - Mouse event
 * @param {HTMLElement} element - Reference element
 * @returns {Object} { x, y } position
 */
export function getRelativeMousePosition(e, element) {
    const rect = element.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

/**
 * Convert screen coordinates to canvas coordinates
 * @param {number} screenX - Screen X coordinate
 * @param {number} screenY - Screen Y coordinate
 * @param {HTMLElement} editorEl - Editor element
 * @param {Object} editor - Editor state { pan, zoom }
 * @returns {Object} { x, y } canvas coordinates
 */
export function screenToCanvas(screenX, screenY, editorEl, editor) {
    const rect = editorEl.getBoundingClientRect();
    return {
        x: (screenX - rect.left - editor.pan.x) / editor.zoom,
        y: (screenY - rect.top - editor.pan.y) / editor.zoom
    };
}

/**
 * Convert canvas coordinates to screen coordinates
 * @param {number} canvasX - Canvas X coordinate
 * @param {number} canvasY - Canvas Y coordinate
 * @param {HTMLElement} editorEl - Editor element
 * @param {Object} editor - Editor state { pan, zoom }
 * @returns {Object} { x, y } screen coordinates
 */
export function canvasToScreen(canvasX, canvasY, editorEl, editor) {
    const rect = editorEl.getBoundingClientRect();
    return {
        x: canvasX * editor.zoom + editor.pan.x + rect.left,
        y: canvasY * editor.zoom + editor.pan.y + rect.top
    };
}
