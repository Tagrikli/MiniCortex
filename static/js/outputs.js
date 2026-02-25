/**
 * MiniCortex - Outputs Panel
 * 
 * Handles output rendering for nodes (Vector2D, Text, Numeric, etc.)
 */

import { getOutputCanvas, setOutputCanvas, clearOutputCanvases, findNode, updateNode } from './state.js';
import { formatNumeric } from './utils.js';
import { setOutputEnabled } from './api.js';

/** @type {Object<string, Object<string, any>>} */
let cachedNodeOutputs = {};

// ─────────────────────────────────────────────────────────────────────────────
// Output Rendering
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render all outputs for a node
 * @param {Object} node - Node object
 * @returns {string} HTML string
 */
export function renderOutputs(node) {
    if (!node.outputs || node.outputs.length === 0) {
        return '';
    }
    
    const outputsHtml = node.outputs.map(output => {
        switch (output.type) {
            case 'numeric':
                return renderNumericOutput(node.node_id, output);
            case 'text':
                return renderTextOutput(node.node_id, output);
            case 'vector1d':
                return renderVector1DOutput(node.node_id, output);
            case 'vector2d':
                return renderVector2DOutput(node.node_id, output);
            default:
                return '';
        }
    }).join('');
    
    return `<div class="node-outputs">${outputsHtml}</div>`;
}

/**
 * Render a numeric output display
 * @param {Object} output - Output definition
 * @returns {string} HTML string
 */
export function renderNumericOutput(nodeId, output) {
    const value = output.formatted || output.value || '0';
    const hiddenClass = output.enabled === false ? 'output-disabled' : '';
    
    return `
        <div class="output-item">
            ${renderOutputLabel(nodeId, output)}
            <div class="output-body ${hiddenClass}">
                <div class="output-value" data-output="${output.key}">${value}</div>
            </div>
        </div>
    `;
}

/**
 * Render a text output display
 * @param {Object} output - Output definition
 * @returns {string} HTML string
 */
export function renderTextOutput(nodeId, output) {
    const value = output.value || output.default || '';
    const hiddenClass = output.enabled === false ? 'output-disabled' : '';
    
    return `
        <div class="output-item">
            ${renderOutputLabel(nodeId, output)}
            <div class="output-body ${hiddenClass}">
                <div class="output-value text-output" data-output="${output.key}">${value}</div>
            </div>
        </div>
    `;
}

/**
 * Render a 1D vector output display (canvas)
 * @param {string} nodeId - Node ID
 * @param {Object} output - Output definition
 * @returns {string} HTML string
 */
export function renderVector1DOutput(nodeId, output) {
    const hiddenClass = output.enabled === false ? 'output-disabled' : '';
    return `
        <div class="output-item">
            ${renderOutputLabel(nodeId, output)}
            <div class="output-body ${hiddenClass}">
                <div class="output-canvas">
                    <canvas data-node="${nodeId}" 
                            data-output="${output.key}" 
                            data-type="vector1d"
                            height="30"></canvas>
                </div>
            </div>
        </div>
    `;
}

/**
 * Render a 2D vector output display (canvas)
 * @param {string} nodeId - Node ID
 * @param {Object} output - Output definition
 * @returns {string} HTML string
 */
export function renderVector2DOutput(nodeId, output) {
    const hiddenClass = output.enabled === false ? 'output-disabled' : '';
    return `
        <div class="output-item">
            ${renderOutputLabel(nodeId, output)}
            <div class="output-body ${hiddenClass}">
                <div class="output-canvas">
                    <canvas data-node="${nodeId}" 
                            data-output="${output.key}" 
                            data-type="vector2d"></canvas>
                </div>
            </div>
        </div>
    `;
}

function renderOutputLabel(nodeId, output) {
    const checked = output.enabled !== false ? 'checked' : '';
    return `
        <label class="output-label output-toggle-label">
            <input type="checkbox"
                   class="output-toggle"
                   data-node="${nodeId}"
                   data-output="${output.key}"
                   ${checked}
                   onchange="window.onOutputToggle(this)">
            <span>${output.label}</span>
        </label>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas Initialization
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialize canvases for a node's outputs
 * @param {Object} node - Node object
 * @param {HTMLElement} nodeEl - Node element
 */
export function initializeCanvases(node, nodeEl) {
    const canvases = nodeEl.querySelectorAll('canvas');
    
    canvases.forEach(canvas => {
        const outputKey = canvas.dataset.output;
        const ctx = canvas.getContext('2d');
        setOutputCanvas(node.node_id, outputKey, {
            canvas,
            ctx,
            type: canvas.dataset.type
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Output Updates (WebSocket)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Update outputs from WebSocket data
 * @param {Object} nodesData - Map of nodeId -> { outputs: {...} }
 */
export function updateOutputs(nodesData) {
    for (const [nodeId, data] of Object.entries(nodesData)) {
        const outputs = data.outputs;
        if (!outputs) continue;
        cachedNodeOutputs[nodeId] = { ...(cachedNodeOutputs[nodeId] || {}), ...outputs };
        
        const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
        if (!nodeEl) continue;
        
        for (const [key, value] of Object.entries(outputs)) {
            updateOutput(nodeEl, nodeId, key, value);
        }
    }
}

/**
 * Re-apply last known outputs to currently rendered nodes/canvases.
 * Used after node re-renders, which recreate canvas elements.
 */
export function reapplyCachedOutputs() {
    updateOutputs(Object.fromEntries(
        Object.entries(cachedNodeOutputs).map(([nodeId, outputs]) => [nodeId, { outputs }])
    ));
}

/**
 * Update a single output display
 * @param {HTMLElement} nodeEl - Node element
 * @param {string} nodeId - Node ID
 * @param {string} key - Output key
 * @param {*} value - Output value
 */
function updateOutput(nodeEl, nodeId, key, value) {
    const outputEl = nodeEl.querySelector(`.output-value[data-output="${key}"], canvas[data-output="${key}"]`);
    if (!outputEl) return;
    
    if (outputEl.tagName === 'CANVAS') {
        renderCanvasOutput(nodeId, key, value);
    } else if (outputEl.classList.contains('output-value')) {
        outputEl.textContent = formatNumeric(value, '.4f');
    }
}

/**
 * Render output to a canvas
 * @param {string} nodeId - Node ID
 * @param {string} outputKey - Output key
 * @param {*} data - Output data (array)
 */
export function renderCanvasOutput(nodeId, outputKey, data) {
    const canvasInfo = getOutputCanvas(nodeId, outputKey);
    if (!canvasInfo) return;
    
    const { canvas, ctx, type } = canvasInfo;
    const node = findNode(nodeId);
    const outputSpec = node?.outputs?.find(output => output.key === outputKey);
    const colorMode = outputSpec?.color_mode || 'grayscale';
    
    if (type === 'vector2d' && Array.isArray(data)) {
        render2DArray(ctx, canvas, data, colorMode);
    } else if (type === 'vector1d' && Array.isArray(data)) {
        render1DArray(ctx, canvas, data);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Array Rendering
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render a 2D array to a canvas
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Array} data - 2D array of values
 * @param {number} size - Canvas size (width and height)
 */
export function render2DArray(ctx, canvas, data, colorMode = 'grayscale') {
    const rows = data.length;
    const cols = data[0]?.length || 1;

    const displayWidth = canvas.clientWidth || 60;
    const cellSize = displayWidth / cols;
    const width = Math.max(1, Math.round(cols * cellSize));
    const height = Math.max(1, Math.round(rows * cellSize));

    canvas.width = width;
    canvas.height = height;
    canvas.style.height = `${height}px`;
    
    const cellWidth = width / cols;
    const cellHeight = height / rows;
    
    const diverging = String(colorMode).toLowerCase();
    const useDiverging = diverging === 'bwr' || diverging === 'diverging';
    
    // Clear canvas
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, width, height);
    
    // Draw cells
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const raw = data[r][c];
            if (useDiverging) {
                const t = Math.min(1, Math.max(0, (raw + 1) / 2));
                let rCol, gCol, bCol;
                if (t < 0.5) {
                    const k = t / 0.5;
                    rCol = Math.round(255 * k);
                    gCol = Math.round(255 * k);
                    bCol = 255;
                } else {
                    const k = (t - 0.5) / 0.5;
                    rCol = 255;
                    gCol = Math.round(255 * (1 - k));
                    bCol = Math.round(255 * (1 - k));
                }
                ctx.fillStyle = `rgb(${rCol},${gCol},${bCol})`;
            } else {
                const v = Math.min(1, Math.max(0, raw));
                const gray = Math.floor(v * 255);
                ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
            }
            ctx.fillRect(
                Math.floor(c * cellWidth),
                Math.floor(r * cellHeight),
                Math.ceil(cellWidth),
                Math.ceil(cellHeight)
            );
        }
    }
}

/**
 * Render a 1D array to a canvas (bar chart)
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Array} data - 1D array of values
 */
export function render1DArray(ctx, canvas, data) {
    const len = data.length;
    canvas.width = len;
    canvas.height = 30;
    
    // Clear canvas
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Find min/max for normalization
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < len; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
    const range = max - min || 1;
    
    // Draw bars
    ctx.fillStyle = '#e94560';
    for (let i = 0; i < len; i++) {
        const v = (data[i] - min) / range;
        const height = v * 28;
        ctx.fillRect(i, 30 - height, 1, height);
    }
}

export async function onOutputToggle(inputEl) {
    const nodeId = inputEl.dataset.node;
    const outputKey = inputEl.dataset.output;
    const enabled = inputEl.checked;
    const outputItem = inputEl.closest('.output-item');
    const outputBody = outputItem ? outputItem.querySelector('.output-body') : null;
    try {
        await setOutputEnabled(nodeId, outputKey, enabled);
        const node = findNode(nodeId);
        if (node && Array.isArray(node.outputs)) {
            const outputs = node.outputs.map(output => {
                if (output.key === outputKey) {
                    return { ...output, enabled };
                }
                return output;
            });
            updateNode(nodeId, { outputs });
        }
        if (outputBody) {
            outputBody.classList.toggle('output-disabled', !enabled);
        }
    } catch (err) {
        console.error('Failed to toggle output:', err);
        inputEl.checked = !enabled;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exports for other modules
// ─────────────────────────────────────────────────────────────────────────────

export { clearOutputCanvases };

if (typeof window !== 'undefined') {
    window.onOutputToggle = onOutputToggle;
}
