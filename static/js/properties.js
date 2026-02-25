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
            case 'slider':
                return renderSlider(node.node_id, prop);
            case 'integer':
                return renderInteger(node.node_id, prop);
            case 'checkbox':
                return renderCheckbox(node.node_id, prop);
            case 'radio':
                return renderRadio(node.node_id, prop);
            default:
                return '';
        }
    }).join('');
    
    return `<div class="node-properties">${propertiesHtml}</div>`;
}

/**
 * Render a slider property
 * @param {string} nodeId - Node ID
 * @param {Object} prop - Property definition
 * @returns {string} HTML string
 */
export function renderSlider(nodeId, prop) {
    const value = prop.value !== undefined ? prop.value : prop.default;
    const displayValue = formatValue(value, prop.scale);
    const useLog = prop.scale === 'log' && prop.min > 0 && prop.max > 0;
    const sliderMin = useLog ? Math.log10(prop.min) : prop.min;
    const sliderMax = useLog ? Math.log10(prop.max) : prop.max;
    const sliderValue = useLog ? Math.log10(value) : value;
    
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
                   step="${prop.scale === 'log' ? 'any' : '0.01'}"
                   oninput="window.onSliderChange(this)">
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
 * Render a checkbox property
 * @param {string} nodeId - Node ID
 * @param {Object} prop - Property definition
 * @returns {string} HTML string
 */
export function renderCheckbox(nodeId, prop) {
    const checked = prop.value !== undefined ? prop.value : prop.default;
    
    return `
        <div class="property-group">
            <div class="property-checkbox" onclick="window.onCheckboxClick('${nodeId}', '${prop.key}')">
                <div class="checkbox ${checked ? 'checked' : ''}" 
                     data-node="${nodeId}" 
                     data-property="${prop.key}"></div>
                <span>${prop.label}</span>
            </div>
        </div>
    `;
}

/**
 * Render a radio button group property
 * @param {string} nodeId - Node ID
 * @param {Object} prop - Property definition
 * @returns {string} HTML string
 */
export function renderRadio(nodeId, prop) {
    const selected = prop.value !== undefined ? prop.value : prop.default;
    
    const options = prop.options.map(opt => `
        <div class="radio-option ${opt === selected ? 'selected' : ''}"
             data-node="${nodeId}"
             data-property="${prop.key}"
             data-value="${opt}"
             onclick="window.onRadioClick(this)">
            ${opt}
        </div>
    `).join('');
    
    return `
        <div class="property-group">
            <div class="property-label">
                <span>${prop.label}</span>
            </div>
            <div class="property-radio">
                ${options}
            </div>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Change Handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Handle slider value change
 * @param {HTMLInputElement} slider - Slider element
 */
export async function onSliderChange(slider) {
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
 * Handle checkbox click
 * @param {string} nodeId - Node ID
 * @param {string} propKey - Property key
 */
export async function onCheckboxClick(nodeId, propKey) {
    const checkbox = document.querySelector(`.checkbox[data-node="${nodeId}"][data-property="${propKey}"]`);
    const checked = !checkbox.classList.contains('checked');
    
    checkbox.classList.toggle('checked');
    
    // Send to server
    await setProperty(nodeId, propKey, checked);
}

/**
 * Handle radio option click
 * @param {HTMLElement} element - Radio option element
 */
export async function onRadioClick(element) {
    const nodeId = element.dataset.node;
    const propKey = element.dataset.property;
    const value = element.dataset.value;
    
    // Update UI
    const siblings = element.parentElement.querySelectorAll('.radio-option');
    siblings.forEach(s => s.classList.remove('selected'));
    element.classList.add('selected');
    
    // Send to server
    await setProperty(nodeId, propKey, value);
}

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
    window.onSliderChange = onSliderChange;
    window.onCheckboxClick = onCheckboxClick;
    window.onRadioClick = onRadioClick;
    window.onActionClick = onActionClick;
    window.onIntegerChange = onIntegerChange;
}
