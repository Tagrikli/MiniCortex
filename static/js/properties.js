/**
 * MiniCortex - Properties Panel
 * 
 * Handles property rendering and change handlers for nodes.
 */

import { findNode } from './state.js';
import { setProperty } from './api.js';
import { formatValue } from './utils.js';

// ─────────────────────────────────────────────────────────────────────────────
// Property Rendering
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render all properties for a node
 * @param {Object} node - Node object
 * @returns {string} HTML string
 */
export function renderProperties(node) {
    if (!node.properties || node.properties.length === 0) {
        return '';
    }
    
    const propertiesHtml = node.properties.map(prop => {
        switch (prop.type) {
            case 'range':
            case 'slider': // Legacy support
                return renderRange(node.node_id, prop);
            case 'integer':
                return renderInteger(node.node_id, prop);
            case 'bool':
            case 'checkbox': // Legacy support
                return renderBool(node.node_id, prop);
            case 'enum':
            case 'dropdown': // Legacy support
                return renderEnum(node.node_id, prop);
            default:
                return '';
        }
    }).join('');
    
    return `<div class="node-properties">${propertiesHtml}</div>`;
}

/**
 * Render a range property
 * @param {string} nodeId - Node ID
 * @param {Object} prop - Property definition
 * @returns {string} HTML string
 */
export function renderRange(nodeId, prop) {
    const value = prop.value !== undefined ? prop.value : prop.default;
    const displayValue = formatValue(value, prop.scale);
    const useLog = prop.scale === 'log' && prop.min > 0 && prop.max > 0;
    const sliderMin = useLog ? Math.log10(prop.min) : prop.min;
    const sliderMax = useLog ? Math.log10(prop.max) : prop.max;
    const sliderValue = useLog ? Math.log10(value) : value;
    const step = prop.step || (prop.scale === 'log' ? 'any' : '0.01');
    
    return `
        <div class="property-group">
            <div class="property-label">
                <span>${prop.label}</span>
                <span class="property-value">${displayValue}</span>
            </div>
            <input type="range" 
                   class="property-slider" 
                   data-node="${nodeId}" 
                   data-property="${prop.key}"
                   min="${sliderMin}" 
                   max="${sliderMax}" 
                   value="${sliderValue}"
                   step="${step}"
                   oninput="window.onRangeChange(this)">
        </div>
    `;
}

/**
 * Render an integer property
 * @param {string} nodeId - Node ID
 * @param {Object} prop - Property definition
 * @returns {string} HTML string
 */
export function renderInteger(nodeId, prop) {
    const value = prop.value !== undefined ? prop.value : prop.default;
    const minAttr = prop.min !== null && prop.min !== undefined ? `min="${prop.min}"` : '';
    const maxAttr = prop.max !== null && prop.max !== undefined ? `max="${prop.max}"` : '';
    
    return `
        <div class="property-group">
            <div class="property-label">
                <span>${prop.label}</span>
                <span class="property-value">${value}</span>
            </div>
            <input type="number"
                   class="property-number input"
                   data-node="${nodeId}"
                   data-property="${prop.key}"
                   step="1"
                   value="${value}"
                   ${minAttr}
                   ${maxAttr}
                   oninput="window.onIntegerChange(this)">
        </div>
    `;
}

/**
 * Render a bool property
 * @param {string} nodeId - Node ID
 * @param {Object} prop - Property definition
 * @returns {string} HTML string
 */
export function renderBool(nodeId, prop) {
    const checked = prop.value !== undefined ? prop.value : prop.default;
    
    return `
        <div class="property-group">
            <div class="property-checkbox" onclick="window.onBoolChange('${nodeId}', '${prop.key}')">
                <div class="checkbox ${checked ? 'checked' : ''}" 
                     data-node="${nodeId}" 
                     data-property="${prop.key}"></div>
                <span>${prop.label}</span>
            </div>
        </div>
    `;
}

/**
 * Render an enum property
 * @param {string} nodeId - Node ID
 * @param {Object} prop - Property definition
 * @returns {string} HTML string
 */
export function renderEnum(nodeId, prop) {
    const selected = prop.value !== undefined ? prop.value : prop.default;
    
    const options = prop.options.map(opt => `
        <div class="enum-option ${opt === selected ? 'selected' : ''}"
             data-node="${nodeId}"
             data-property="${prop.key}"
             data-value="${opt}"
             onclick="window.onEnumOptionClick(this)">
            ${opt}
        </div>
    `).join('');
    
    return `
        <div class="property-group">
            <div class="property-label">
                <span>${prop.label}</span>
            </div>
            <div class="property-enum-container">
                <div class="property-enum-header"
                     data-node="${nodeId}"
                     data-property="${prop.key}"
                     onclick="window.onEnumHeaderClick(this)">
                    <span class="enum-selected-value">${selected}</span>
                    <span class="enum-arrow">▼</span>
                </div>
                <div class="property-enum-options">
                    ${options}
                </div>
            </div>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Change Handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Handle range value change
 * @param {HTMLInputElement} slider - Slider element
 */
export async function onRangeChange(slider) {
    const nodeId = slider.dataset.node;
    const propKey = slider.dataset.property;
    let value = parseFloat(slider.value);
    
    // Update display
    const label = slider.closest('.property-group').querySelector('.property-value');
    const node = findNode(nodeId);
    const prop = node?.properties?.find(p => p.key === propKey);
    const scale = prop?.scale || 'linear';
    if (scale === 'log' && prop?.min > 0 && prop?.max > 0) {
        value = Math.pow(10, value);
    }
    label.textContent = formatValue(value, scale);
    
    // Send to server
    await setProperty(nodeId, propKey, value);
}

/**
 * Handle integer value change
 * @param {HTMLInputElement} input - Number input element
 */
export async function onIntegerChange(input) {
    const nodeId = input.dataset.node;
    const propKey = input.dataset.property;
    const value = parseInt(input.value, 10);
    if (Number.isNaN(value)) return;
    
    // Update display
    const label = input.closest('.property-group').querySelector('.property-value');
    label.textContent = String(value);
    
    // Send to server
    await setProperty(nodeId, propKey, value);
}

/**
 * Handle bool change
 * @param {string} nodeId - Node ID
 * @param {string} propKey - Property key
 */
export async function onBoolChange(nodeId, propKey) {
    const checkbox = document.querySelector(`.checkbox[data-node="${nodeId}"][data-property="${propKey}"]`);
    const checked = !checkbox.classList.contains('checked');
    
    checkbox.classList.toggle('checked');
    
    // Send to server
    await setProperty(nodeId, propKey, checked);
}

/**
 * Handle enum header click (toggle dropdown)
 * @param {HTMLElement} header - Enum header element
 */
export function onEnumHeaderClick(header) {
    const container = header.closest('.property-enum-container');
    const options = container.querySelector('.property-enum-options');
    const isOpen = options.classList.contains('open');
    
    // Close all other dropdowns
    document.querySelectorAll('.property-enum-options.open').forEach(el => {
        el.classList.remove('open');
    });
    
    if (!isOpen) {
        options.classList.add('open');
    }
}

/**
 * Handle enum option click
 * @param {HTMLElement} option - Enum option element
 */
export async function onEnumOptionClick(option) {
    const nodeId = option.dataset.node;
    const propKey = option.dataset.property;
    const value = option.dataset.value;
    
    // Update UI
    const container = option.closest('.property-enum-container');
    const header = container.querySelector('.property-enum-header');
    const selectedValue = header.querySelector('.enum-selected-value');
    const options = container.querySelector('.property-enum-options');
    
    // Update selected value display
    selectedValue.textContent = value;
    
    // Update selected state
    options.querySelectorAll('.enum-option').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.value === value);
    });
    
    // Close dropdown
    options.classList.remove('open');
    
    // Send to server
    await setProperty(nodeId, propKey, value);
}

// Close dropdowns when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.property-enum-container')) {
        document.querySelectorAll('.property-enum-options.open').forEach(el => {
            el.classList.remove('open');
        });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Actions Rendering
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render action buttons for a node
 * @param {Object} node - Node object
 * @returns {string} HTML string
 */
export function renderActions(node) {
    if (!node.actions || node.actions.length === 0) {
        return '';
    }
    
    const buttons = node.actions.map(action => `
        <button class="action-btn" 
                data-node="${node.node_id}" 
                data-action="${action.key}"
                onclick="window.onActionClick('${node.node_id}', '${action.key}')">
            ${action.label}
        </button>
    `).join('');
    
    return `<div class="node-actions">${buttons}</div>`;
}

/**
 * Handle action button click
 * @param {string} nodeId - Node ID
 * @param {string} actionKey - Action key
 */
export async function onActionClick(nodeId, actionKey) {
    const { executeAction } = await import('./api.js');
    await executeAction(nodeId, actionKey);
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Registration (for inline event handlers)
// ─────────────────────────────────────────────────────────────────────────────

// Register handlers on window for inline event handlers
if (typeof window !== 'undefined') {
    window.onRangeChange = onRangeChange;
    window.onBoolChange = onBoolChange;
    window.onEnumHeaderClick = onEnumHeaderClick;
    window.onEnumOptionClick = onEnumOptionClick;
    window.onActionClick = onActionClick;
    window.onIntegerChange = onIntegerChange;
}
