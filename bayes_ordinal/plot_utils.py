"""
Plot utilities for bayes_ordinal package.

This module provides utilities for saving plots consistently across the package.
"""

import os
import matplotlib.pyplot as plt
from typing import Optional

# Global settings for plot saving
_SAVE_PLOTS = False
_SAVE_DIR = "test_plots"
_SAVE_COUNTER = 0

# Store original show function
_ORIGINAL_SHOW = plt.show

def set_plot_saving(enabled: bool = True, save_dir: str = "test_plots"):
    """
    Enable or disable automatic plot saving.
    
    Parameters:
    -----------
    enabled : bool
        Whether to save plots automatically
    save_dir : str
        Directory to save plots in
    """
    global _SAVE_PLOTS, _SAVE_DIR, _ORIGINAL_SHOW
    _SAVE_PLOTS = enabled
    _SAVE_DIR = save_dir
    
    if enabled:
        os.makedirs(save_dir, exist_ok=True)
        # Monkey-patch plt.show to save plots
        plt.show = lambda: show_and_save()
    else:
        # Restore original show function
        plt.show = _ORIGINAL_SHOW

def save_current_plot(filename: Optional[str] = None, dpi: int = 300):
    """
    Save the current plot if plot saving is enabled.
    
    Parameters:
    -----------
    filename : str, optional
        Filename to save as. If None, auto-generates a name.
    dpi : int
        DPI for saved image
    """
    global _SAVE_PLOTS, _SAVE_DIR, _SAVE_COUNTER
    
    if not _SAVE_PLOTS:
        return
    
    if filename is None:
        global _SAVE_COUNTER
        _SAVE_COUNTER += 1
        filename = f"plot_{_SAVE_COUNTER:03d}.png"
    
    # Ensure filename has .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    
    filepath = os.path.join(_SAVE_DIR, filename)
    
    # Check if file already exists to avoid duplicates
    if os.path.exists(filepath):
        print(f"⚠️  Plot already exists: {filepath}")
        return
    
    try:
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"✓ Plot saved: {filepath}")
    except Exception as e:
        print(f"✗ Failed to save plot {filepath}: {e}")

def show_and_save(filename: Optional[str] = None, dpi: int = 300):
    """
    Show the plot and save it if saving is enabled.
    
    Parameters:
    -----------
    filename : str, optional
        Filename to save as. If None, auto-generates a name.
    dpi : int
        DPI for saved image
    """
    save_current_plot(filename, dpi)
    _ORIGINAL_SHOW()  # Use original show function to avoid recursion

def reset_save_counter():
    """Reset the automatic filename counter."""
    global _SAVE_COUNTER
    _SAVE_COUNTER = 0 