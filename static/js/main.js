/**
 * MiniCortex - Main Entry Point
 * 
 * This is the main entry point for the MiniCortex node editor application.
 * It imports and initializes all modules.
 */

import { initApp } from './editor.js';

// ─────────────────────────────────────────────────────────────────────────────
// Application Entry Point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialize the application when the DOM is ready
 */
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await initApp();
        console.log('MiniCortex initialized successfully');
    } catch (error) {
        console.error('Failed to initialize MiniCortex:', error);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Global Error Handler
// ─────────────────────────────────────────────────────────────────────────────

window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});
