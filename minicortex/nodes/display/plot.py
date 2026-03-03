"""Plot node with history for MiniCortex."""

import numpy as np

from ...core.node import Node
from ...core.descriptors.ports import InputPort
from ...core.descriptors.store import Store
from ...core.descriptors.properties import Integer, Enum, Float
from ...core.descriptors.displays import Vector1D, BarChart, LineChart, Text


class PlotBar(Node):
    """Plot numeric values in a history array using BarChart display.
    
    This node receives numeric inputs and maintains a history of values.
    The history is displayed as a bar chart using BarChart with support for
    negative values. The history is limited by the max_history property.
    """

    # Input port for numeric values
    input_value = InputPort("Input", (int, float))
    
    # Property for maximum history length
    max_history = Integer("Max History", default=100)
    
    # Scale mode property - auto or manual
    scale_mode = Enum(
        "Scale Mode",
        options=["auto", "manual"],
        default="auto",
        on_change="_on_scale_changed"
    )
    
    # Manual scale range (used when scale_mode is "manual")
    scale_min = Float("Min", default=0.0, on_change="_on_scale_values_changed")
    scale_max = Float("Max", default=1.0, on_change="_on_scale_values_changed")
    
    # Store for history - persists across sessions
    history = Store("History", default=None)
    
    # Store for current count of valid elements
    history_count = Store("History Count", default=0)
    
    # Display for plotting the history as bar chart
    plot = BarChart("Plot", color="#e94560", show_negative=True)
    
    # Display for min/max info
    info = Text("Info", default="min: —  max: —")

    def _on_scale_changed(self, new_value, old_value):
        """Called when scale_mode property changes, updates the display config."""
        plot_descriptor = PlotBar.__dict__["plot"]
        if self.scale_mode == "manual":
            # Auto-set min/max to current history values when switching to manual
            if self.history_count > 0:
                max_len = int(self.max_history)
                if self.history_count < max_len:
                    display_array = self.history[-self.history_count:]
                else:
                    display_array = self.history.copy()
                dmin = float(np.nanmin(display_array))
                dmax = float(np.nanmax(display_array))
                self.scale_min = dmin
                self.scale_max = dmax
            plot_descriptor.change_scale("manual", float(self.scale_min), float(self.scale_max), obj=self)
        else:
            plot_descriptor.change_scale("auto", obj=self)

    def _on_scale_values_changed(self, new_value, old_value):
        """Called when scale_min or scale_max changes, updates the display config."""
        if self.scale_mode == "manual":
            plot_descriptor = PlotBar.__dict__["plot"]
            plot_descriptor.change_scale("manual", float(self.scale_min), float(self.scale_max), obj=self)

    def init(self):
        # Initialize history as numpy array if not already set
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
        else:
            # Resize if needed when loading from storage
            if len(self.history) != max_len:
                self._resize_history(max_len)
        
        if self.history_count is None:
            self.history_count = 0

    def _resize_history(self, new_size):
        """Resize history array, shifting existing data to the right."""
        old_size = len(self.history)
        new_history = np.full(new_size, np.nan, dtype=np.float32)
        
        # When increasing size: shift existing data to the END (right-aligned)
        # When decreasing size: keep most recent data at the end
        copy_count = min(old_size, new_size)
        if new_size > old_size:
            # Increase: put old data at the end
            new_history[-copy_count:] = self.history[-copy_count:]
        else:
            # Decrease: keep the most recent data
            new_history = self.history[-new_size:].copy()
        
        self.history = new_history
        
        # If increasing size and array was full, expand count to new size
        # If decreasing size, cap count at new size
        if new_size > old_size:
            if self.history_count >= old_size:
                self.history_count = new_size
        else:
            if self.history_count > new_size:
                self.history_count = new_size

    def process(self):
        max_len = int(self.max_history)
        
        # Check if max_history changed
        if len(self.history) != max_len:
            self._resize_history(max_len)
        
        if self.input_value is not None:
            # Convert input to float
            float_value = float(self.input_value)
            
            # Shift content left and add new value at the end
            if self.history_count > 0:
                # Shift existing values left
                self.history[:-1] = self.history[1:]
            
            # Add new value at the end
            self.history[-1] = float_value
            
            # Increment count, but cap at max_len
            if self.history_count < max_len:
                self.history_count += 1
        
        # Get the slice of valid data for display
        if self.history_count > 0:
            # Get the last history_count elements
            if self.history_count < max_len:
                display_array = self.history[-self.history_count:]
            else:
                display_array = self.history.copy()
            self.plot = display_array
            
            # Calculate and display min/max
            dmin = float(np.nanmin(display_array))
            dmax = float(np.nanmax(display_array))
            self.info = f"min: {dmin:.2f}  max: {dmax:.2f}"
        else:
            # Empty plot when no history
            self.plot = np.array([], dtype=np.float32)
            self.info = "min: —  max: —"


class PlotLine(Node):
    """Plot numeric values in a history array using LineChart display.
    
    This node receives numeric inputs and maintains a history of values.
    The history is displayed as a line chart. The history is limited by
    the max_history property.
    """

    # Input port for numeric values
    input_value = InputPort("Input", (int, float))
    
    # Property for maximum history length
    max_history = Integer("Max History", default=100)
    
    # Scale mode property - auto or manual
    scale_mode = Enum(
        "Scale Mode",
        options=["auto", "manual"],
        default="auto",
        on_change="_on_scale_changed"
    )
    
    # Manual scale range (used when scale_mode is "manual")
    scale_min = Float("Min", default=0.0, on_change="_on_scale_values_changed")
    scale_max = Float("Max", default=1.0, on_change="_on_scale_values_changed")
    
    # Store for history - persists across sessions
    history = Store("History", default=None)
    
    # Store for current count of valid elements
    history_count = Store("History Count", default=0)
    
    # Display for plotting the history as line chart
    plot = LineChart("Plot", color="#00f5ff", line_width=0.7)
    
    # Display for min/max info
    info = Text("Info", default="min: —  max: —")

    def _on_scale_changed(self, new_value, old_value):
        """Called when scale_mode property changes, updates the display config."""
        plot_descriptor = PlotLine.__dict__["plot"]
        if self.scale_mode == "manual":
            # Auto-set min/max to current history values when switching to manual
            if self.history_count > 0:
                max_len = int(self.max_history)
                if self.history_count < max_len:
                    display_array = self.history[-self.history_count:]
                else:
                    display_array = self.history.copy()
                dmin = float(np.nanmin(display_array))
                dmax = float(np.nanmax(display_array))
                self.scale_min = dmin
                self.scale_max = dmax
            plot_descriptor.change_scale("manual", float(self.scale_min), float(self.scale_max), obj=self)
        else:
            plot_descriptor.change_scale("auto", obj=self)

    def _on_scale_values_changed(self, new_value, old_value):
        """Called when scale_min or scale_max changes, updates the display config."""
        if self.scale_mode == "manual":
            plot_descriptor = PlotLine.__dict__["plot"]
            plot_descriptor.change_scale("manual", float(self.scale_min), float(self.scale_max), obj=self)

    def init(self):
        # Initialize history as numpy array if not already set
        max_len = int(self.max_history)
        if self.history is None:
            self.history = np.full(max_len, np.nan, dtype=np.float32)
        else:
            # Resize if needed when loading from storage
            if len(self.history) != max_len:
                self._resize_history(max_len)
        
        if self.history_count is None:
            self.history_count = 0

    def _resize_history(self, new_size):
        """Resize history array, shifting existing data to the right."""
        old_size = len(self.history)
        new_history = np.full(new_size, np.nan, dtype=np.float32)
        
        # When increasing size: shift existing data to the END (right-aligned)
        # When decreasing size: keep most recent data at the end
        copy_count = min(old_size, new_size)
        if new_size > old_size:
            # Increase: put old data at the end
            new_history[-copy_count:] = self.history[-copy_count:]
        else:
            # Decrease: keep the most recent data
            new_history = self.history[-new_size:].copy()
        
        self.history = new_history
        
        # If increasing size and array was full, expand count to new size
        # If decreasing size, cap count at new size
        if new_size > old_size:
            if self.history_count >= old_size:
                self.history_count = new_size
        else:
            if self.history_count > new_size:
                self.history_count = new_size

    def process(self):
        max_len = int(self.max_history)
        
        # Check if max_history changed
        if len(self.history) != max_len:
            self._resize_history(max_len)
        
        if self.input_value is not None:
            # Convert input to float
            float_value = float(self.input_value)
            
            # Shift content left and add new value at the end
            if self.history_count > 0:
                # Shift existing values left
                self.history[:-1] = self.history[1:]
            
            # Add new value at the end
            self.history[-1] = float_value
            
            # Increment count, but cap at max_len
            if self.history_count < max_len:
                self.history_count += 1
        
        # Get the slice of valid data for display
        if self.history_count > 0:
            # Get the last history_count elements
            if self.history_count < max_len:
                display_array = self.history[-self.history_count:]
            else:
                display_array = self.history.copy()
            self.plot = display_array
            
            # Calculate and display min/max
            dmin = float(np.nanmin(display_array))
            dmax = float(np.nanmax(display_array))
            self.info = f"min: {dmin:.2f}  max: {dmax:.2f}"
        else:
            # Empty plot when no history
            self.plot = np.array([], dtype=np.float32)
            self.info = "min: —  max: —"
