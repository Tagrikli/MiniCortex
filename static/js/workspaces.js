/**
 * MiniCortex - Workspace Management
 * 
 * Handles workspace save, load, list, and delete functionality.
 * Workspace dropdown is in the header. Ctrl+S saves.
 */

import { loadConfig } from './api.js';

// Current workspace state
let currentWorkspaceName = null;
let hasUnsavedChanges = false;

// Callback to avoid circular dependency with editor.js
let updateTransformCallback = null;

export function setUpdateTransformCallback(callback) {
    updateTransformCallback = callback;
}

// ─────────────────────────────────────────────────────────────────────────────
// Workspace API Functions
// ─────────────────────────────────────────────────────────────────────────────

async function fetchWorkspaces() {
    const response = await fetch('/api/workspaces');
    if (!response.ok) {
        throw new Error('Failed to fetch workspaces');
    }
    return response.json();
}

async function saveWorkspaceAPI(name) {
    const response = await fetch('/api/workspaces/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    });
    if (!response.ok) {
        throw new Error('Failed to save workspace');
    }
    return response.json();
}

async function loadWorkspaceAPI(name) {
    const response = await fetch('/api/workspaces/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    });
    if (!response.ok) {
        throw new Error('Failed to load workspace');
    }
    return response.json();
}

async function deleteWorkspaceAPI(name) {
    const response = await fetch('/api/workspaces', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    });
    if (!response.ok) {
        throw new Error('Failed to delete workspace');
    }
    return response.json();
}

async function getCurrentWorkspaceAPI() {
    const response = await fetch('/api/workspaces/current');
    if (!response.ok) {
        return null;
    }
    return response.json();
}

async function clearWorkspaceAPI() {
    const response = await fetch('/api/workspaces/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) {
        throw new Error('Failed to clear workspace');
    }
    return response.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// UI Functions
// ─────────────────────────────────────────────────────────────────────────────

function updateCurrentWorkspaceDisplay(name) {
    const nameEl = document.getElementById('workspace-current-name');
    if (nameEl) {
        nameEl.textContent = name || 'Unsaved Workspace';
    }
    currentWorkspaceName = name;
}

async function refreshWorkspaceList() {
    const listEl = document.getElementById('workspace-dropdown-list');
    if (!listEl) return;
    
    try {
        const workspaces = await fetchWorkspaces();
        
        if (workspaces.length === 0) {
            listEl.innerHTML = '<div class="workspace-dropdown-empty">No saved workspaces</div>';
            return;
        }
        
        listEl.innerHTML = workspaces.map(ws => `
            <div class="workspace-dropdown-item ${ws.name === currentWorkspaceName ? 'active' : ''}" data-name="${escapeHtml(ws.name)}">
                <span class="workspace-item-name">${escapeHtml(ws.name)}</span>
                <button class="workspace-item-delete" data-name="${escapeHtml(ws.name)}" title="Delete">✕</button>
            </div>
        `).join('');
        
        // Attach click handlers for loading
        listEl.querySelectorAll('.workspace-dropdown-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // Don't load if clicking delete button
                if (e.target.classList.contains('workspace-item-delete')) {
                    return;
                }
                loadWorkspace(item.dataset.name);
                closeDropdown();
            });
        });
        
        // Attach delete handlers
        listEl.querySelectorAll('.workspace-item-delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteWorkspace(btn.dataset.name);
            });
        });
    } catch (error) {
        console.error('Failed to refresh workspace list:', error);
        listEl.innerHTML = '<div class="workspace-dropdown-empty">Failed to load</div>';
    }
}

function openDropdown() {
    const dropdown = document.getElementById('workspace-dropdown');
    if (dropdown) {
        dropdown.classList.add('open');
        refreshWorkspaceList();
    }
}

function closeDropdown() {
    const dropdown = document.getElementById('workspace-dropdown');
    if (dropdown) {
        dropdown.classList.remove('open');
    }
}

function toggleDropdown() {
    const dropdown = document.getElementById('workspace-dropdown');
    if (dropdown) {
        if (dropdown.classList.contains('open')) {
            closeDropdown();
        } else {
            openDropdown();
        }
    }
}

function showSaveDialog() {
    const overlay = document.getElementById('save-dialog-overlay');
    const input = document.getElementById('save-dialog-input');
    if (overlay && input) {
        input.value = currentWorkspaceName || '';
        overlay.style.display = 'flex';
        input.focus();
        if (!currentWorkspaceName) {
            input.select();
        }
    }
}

function hideSaveDialog() {
    const overlay = document.getElementById('save-dialog-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

async function saveWorkspace() {
    if (currentWorkspaceName) {
        // Save with existing name
        try {
            await saveWorkspaceAPI(currentWorkspaceName);
            hasUnsavedChanges = false;
            console.log('Workspace saved:', currentWorkspaceName);
            await refreshWorkspaceList();
        } catch (error) {
            console.error('Failed to save workspace:', error);
            alert('Failed to save workspace');
        }
    } else {
        // Show dialog for new workspace
        showSaveDialog();
    }
}

async function saveWorkspaceWithName(name) {
    if (!name || !name.trim()) {
        alert('Please enter a workspace name');
        return false;
    }
    
    const trimmedName = name.trim();
    
    try {
        await saveWorkspaceAPI(trimmedName);
        currentWorkspaceName = trimmedName;
        hasUnsavedChanges = false;
        updateCurrentWorkspaceDisplay(trimmedName);
        hideSaveDialog();
        await refreshWorkspaceList();
        console.log('Workspace saved:', trimmedName);
        return true;
    } catch (error) {
        console.error('Failed to save workspace:', error);
        alert('Failed to save workspace');
        return false;
    }
}

async function loadWorkspace(name) {
    try {
        const result = await loadWorkspaceAPI(name);
        console.log('Workspace loaded:', result);
        currentWorkspaceName = name;
        hasUnsavedChanges = false;
        updateCurrentWorkspaceDisplay(name);
        await loadConfig();
        // Apply the viewport transform immediately
        if (updateTransformCallback) {
            updateTransformCallback();
        }
    } catch (error) {
        console.error('Failed to load workspace:', error);
        alert('Failed to load workspace');
    }
}

async function deleteWorkspace(name) {
    if (!confirm(`Delete workspace "${name}"?`)) {
        return;
    }
    
    try {
        await deleteWorkspaceAPI(name);
        
        // If we deleted the current workspace, clear it
        if (name === currentWorkspaceName) {
            currentWorkspaceName = null;
            updateCurrentWorkspaceDisplay(null);
        }
        
        await refreshWorkspaceList();
        console.log('Workspace deleted:', name);
    } catch (error) {
        console.error('Failed to delete workspace:', error);
        alert('Failed to delete workspace');
    }
}

async function newWorkspace() {
    try {
        // Clear server state first
        const result = await clearWorkspaceAPI();
        
        currentWorkspaceName = null;
        hasUnsavedChanges = false;
        updateCurrentWorkspaceDisplay(null);
        closeDropdown();
        
        // Update client state from server response
        const { setNodes, setConnections, updateEditor } = await import('./state.js');
        const { renderNodes, renderConnections } = await import('./editor.js');
        
        if (result.snapshot) {
            setNodes(result.snapshot.nodes || []);
            setConnections(result.snapshot.connections || []);
            if (result.snapshot.viewport) {
                updateEditor({
                    pan: result.snapshot.viewport.pan || { x: 0, y: 0 },
                    zoom: result.snapshot.viewport.zoom || 1.0
                });
            }
        }
        
        renderNodes();
        renderConnections();
        
        // Apply the viewport transform
        if (updateTransformCallback) {
            updateTransformCallback();
        }
        
        console.log('New workspace created');
    } catch (error) {
        console.error('Failed to create new workspace:', error);
        alert('Failed to create new workspace');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

export async function initWorkspaces() {
    // Setup dropdown toggle
    const toggle = document.getElementById('workspace-dropdown-toggle');
    if (toggle) {
        toggle.addEventListener('click', toggleDropdown);
    }
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        const dropdown = document.getElementById('workspace-dropdown');
        if (dropdown && !dropdown.contains(e.target)) {
            closeDropdown();
        }
    });
    
    // Setup new workspace button
    const newBtn = document.getElementById('workspace-new-btn');
    if (newBtn) {
        newBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            newWorkspace();
        });
    }
    
    // Setup save dialog
    const saveBtn = document.getElementById('save-dialog-save');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            const input = document.getElementById('save-dialog-input');
            saveWorkspaceWithName(input.value);
        });
    }
    
    const cancelBtn = document.getElementById('save-dialog-cancel');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', hideSaveDialog);
    }
    
    const saveInput = document.getElementById('save-dialog-input');
    if (saveInput) {
        saveInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                saveWorkspaceWithName(saveInput.value);
            }
        });
    }
    
    // Close save dialog on overlay click
    const overlay = document.getElementById('save-dialog-overlay');
    if (overlay) {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                hideSaveDialog();
            }
        });
    }
    
    // Setup Ctrl+S shortcut
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            saveWorkspace();
        }
    });
    
    // Try to load current workspace from server
    try {
        const current = await getCurrentWorkspaceAPI();
        if (current && current.name) {
            await loadWorkspace(current.name);
        }
    } catch (error) {
        console.log('No current workspace to load');
    }
    
    // Initial refresh
    await refreshWorkspaceList();
}

// Export for use elsewhere
export { currentWorkspaceName, hasUnsavedChanges };
