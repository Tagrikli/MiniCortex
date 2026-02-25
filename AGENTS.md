# AGENTS.md — LLM Agent Instructions for MiniCortex

This file tells you everything you need to operate in this codebase: how to run things, how the system works, and how to create new nodes.

---

## Environment & Tooling

- **Python ≥ 3.12** is required (see `.python-version`).
- **Use `uv`** for all dependency and execution operations:
  ```bash
  uv sync              # Install/update dependencies
  uv run python main.py  # Run the server
  ```
- Never use `pip install` or bare `python` — always use `uv run python` or `uv run` prefix.
- The server runs at `http://localhost:8000` and serves both the REST API and the browser-based node editor.

---

## Project Structure (What Matters)

```
minicortex/
├── core/
│   ├── node.py              # Node base class (inherit from this)
│   ├── registry.py          # Global node/connection registries
│   └── descriptors/         # Descriptor system (ports, properties, displays, etc.)
│       ├── ports.py          # InputPort, OutputPort
│       ├── properties.py     # Slider, Integer, CheckBox, RadioButtons
│       ├── displays.py       # Vector2D, Vector1D, Numeric, Text
│       ├── actions.py        # Button
│       ├── store.py          # Store (persistent state)
│       ├── dynamic.py        # @dynamic decorator (hot-reload)
│       └── node.py           # @node decorator (category registration)
├── network/
│   └── network.py           # Execution engine (topological sort, signal propagation)
├── nodes/                   # PUT NEW NODES HERE
│   ├── input.py             # Input nodes (data sources)
│   ├── area.py              # Processing nodes
│   └── utilities.py         # Utility/transform nodes
└── server/                  # FastAPI server (do not modify unless adding routes)
```

**Key rule:** To add new nodes, create or edit files only in `minicortex/nodes/`. All `.py` files in that directory are auto-discovered on startup.

---

## How the System Works

### Execution Flow

1. `main.py` calls `discover_nodes()` which scans `minicortex/nodes/*.py` and imports them.
2. Any class decorated with `@node.input`, `@node.processing`, `@node.utility`, etc. is registered in the palette.
3. The FastAPI server starts with a background **computation loop** and a **WebSocket broadcast loop**.
4. When the user presses Start (or the network is running), the computation loop calls `Network.execute_step()` each tick.
5. `execute_step()` topologically sorts all nodes (Kahn's algorithm), then for each node in order:
   - Sets input port values from upstream output signals.
   - Calls `node.process()`.
   - Reads output port values and stores them as signals.
6. The broadcast loop sends all display output values to connected browsers via WebSocket at ~40 FPS.

### Signal Propagation Rules

- **Feedforward**: processed in dependency order — downstream nodes see current-tick values.
- **Feedback/cycles**: nodes in a cycle get previous-tick values (t−1) since they can't be topologically resolved.
- **numpy arrays are cloned** between nodes to prevent aliased mutation.
- If any node raises an exception during `process()`, the network **stops** and the error (with full traceback) is broadcast to the browser.

---

## How to Create a New Node

### Step 1: Create or open a file in `minicortex/nodes/`

Any `.py` file (not starting with `_`) in `minicortex/nodes/` is auto-discovered.

### Step 2: Define the class

```python
import numpy as np
from minicortex.core.node import Node
from minicortex.core.descriptors.ports import InputPort, OutputPort
from minicortex.core.descriptors.properties import Slider, Integer, CheckBox, RadioButtons
from minicortex.core.descriptors.displays import Vector2D, Vector1D, Numeric, Text
from minicortex.core.descriptors.actions import Button
from minicortex.core.descriptors.store import Store
from minicortex.core.descriptors import node, dynamic


@node.processing  # Category decorator — REQUIRED
class MyNode(Node):
    """Docstring is optional but recommended."""

    # ── Ports ──────────────────────────────────────────────────────────
    input_data  = InputPort("Input", np.ndarray)     # Receives data
    output_data = OutputPort("Output", np.ndarray)   # Sends data

    # ── Properties (interactive UI controls) ───────────────────────────
    gain = Slider("Gain", 1.0, 0.0, 10.0)              # Float slider
    size = Integer("Size", default=28, min_val=1)       # Integer input
    flip = CheckBox("Flip", default=False)               # Boolean toggle
    mode = RadioButtons("Mode", ["A", "B"], default="A") # Enum select

    # ── Displays (live visualizations in the node) ─────────────────────
    preview  = Vector2D("Preview", color_mode="grayscale")  # 2D image
    bars     = Vector1D("Histogram")                         # 1D bar chart
    loss     = Numeric("Loss", format=".4f")                 # Scalar
    info     = Text("Info", default="Ready")                 # Text label

    # ── Actions (buttons) ──────────────────────────────────────────────
    reset = Button("Reset", callback="_on_reset")

    # ── Stores (persistent state) ──────────────────────────────────────
    step_count = Store(default=0)

    def init(self):
        """Called ONCE after the node is created and registered."""
        self.step_count = 0

    def process(self):
        """Called EVERY network tick. This is where computation happens."""
        if self.input_data is None:
            return

        result = self.input_data * float(self.gain)
        if self.flip:
            result = 1.0 - result

        self.output_data = result
        self.preview = result
        self.bars = result.mean(axis=0)
        self.loss = float(result.mean())
        self.step_count += 1
        self.info = f"Step {self.step_count}, mode={self.mode}"

    def _on_reset(self, params):
        """Called when the Reset button is pressed."""
        self.step_count = 0
        return {"status": "ok"}
```

### Step 3: Restart or rediscover

- **Restart**: `uv run python main.py`
- **Without restart**: If the server is running, the browser has a ↻ rediscover button in the drawer sidebar.

---

## Decorators Reference

### Category Decorators (REQUIRED — exactly one per node class)

These register the node in the palette so users can create instances.

| Decorator | Palette Category |
|---|---|
| `@node.input` | Input |
| `@node.processing` | Processing |
| `@node.utility` | Utilities |
| `@node.output` | Output |
| `@node.custom("CategoryName")` | Custom category |

**Where to import:** `from minicortex.core.descriptors import node`

### `@dynamic` Decorator (OPTIONAL)

Makes a node hot-reloadable. Users can edit the `.py` file and click ↻ in the browser to reload:
- The module is re-imported from disk.
- A new instance replaces the old one.
- **Store values and Property values are preserved.**
- **Connections to valid ports are preserved.** Connections to removed ports are pruned.
- `init()` is called on the new instance.

**Where to import:** `from minicortex.core.descriptors import dynamic`

**Stack order:** `@dynamic` goes ABOVE `@node.xxx`:
```python
@dynamic
@node.utility
class MyNode(Node):
    ...
```

---

## `init()` vs `process()` — Critical Distinction

### `init(self)`
- Called **once** after the node is created and registered.
- Called again after hot-reload (`@dynamic`).
- Called after workspace load (deserialization).
- Use it to: allocate arrays, load data, set up initial state.
- **Store and Property values are already set** when `init()` runs (restored from workspace or copied during reload).
- Can call `self.process()` if you want to produce initial output.

### `process(self)`
- Called **every network tick** when the network is running.
- Also called once during probing (when the user connects wires while the network is paused).
- **Always guard against `None` inputs**: input ports are `None` when not connected or when upstream hasn't produced a value yet.
- Read inputs: `self.input_data` (via InputPort descriptor)
- Set outputs: `self.output_data = result` (via OutputPort descriptor)
- Set displays: `self.preview = array` (via Display descriptor)
- Read properties: `self.gain` returns the current slider value
- Read/write stores: `self.step_count += 1`

---

## Descriptor Types — Complete Reference

### InputPort / OutputPort

```python
input_data = InputPort("Label", data_type)   # data_type: np.ndarray, int, float, str, or "any"
output_data = OutputPort("Label", data_type)
```

- Connections are type-checked. `"any"` accepts anything.
- Access: `self.input_data` returns the value or `None`. `self.output_data = value` sets it.

### Property Descriptors

| Class | Constructor | Access Type |
|---|---|---|
| `Slider` | `Slider("Label", default, min, max, scale="linear"\|"log")` | `float` |
| `Integer` | `Integer("Label", default=0, min_val=None, max_val=None)` | `int` |
| `CheckBox` | `CheckBox("Label", default=False)` | `bool` |
| `RadioButtons` | `RadioButtons("Label", options=["a","b"], default="a")` | `str` |

All accept an optional `on_change="method_name"` parameter that calls `self.method_name(new_value, old_value)` when the property changes.

### Display Descriptors

| Class | Constructor | Renders As |
|---|---|---|
| `Vector2D` | `Vector2D("Label", color_mode="grayscale"\|"bwr")` | 2D image canvas |
| `Vector1D` | `Vector1D("Label")` | Bar chart canvas |
| `Numeric` | `Numeric("Label", format=".4f")` | Formatted number |
| `Text` | `Text("Label", default="")` | Text string |

- Set via `self.display_name = value`. For `Vector2D`, assign a 2D numpy array. For `Vector1D`, assign a 1D array.
- Displays can be toggled on/off by the user in the browser.

### Store

```python
counter = Store(default=0)
matrix  = Store(default=None)
```

- Persisted to workspace JSON files. numpy arrays are serialized correctly.
- Preserved across `@dynamic` hot-reloads.
- Access: `self.counter` reads, `self.counter = 5` writes.

### Button (Action)

```python
reset = Button("Label", callback="_method_name")
```

- Calls `self._method_name(params)` when clicked. `params` is a dict (usually empty `{}`).
- Return a dict (e.g., `{"status": "ok"}`) or `None`.

---

## Common Patterns

### Node with no inputs (generator/source)

```python
@node.input
class Counter(Node):
    output_value = OutputPort("Value", int)
    step_size = Integer("Step", default=1, min_val=1)
    count = Store(default=0)
    info = Text("Count", default="0")

    def init(self):
        self.count = 0
        self.output_value = 0

    def process(self):
        self.count += int(self.step_size)
        self.output_value = self.count
        self.info = str(self.count)
```

### Node with multiple inputs

```python
@node.utility
class Blend(Node):
    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output_data = OutputPort("Result", np.ndarray)
    alpha = Slider("Alpha", 0.5, 0.0, 1.0)

    def process(self):
        if self.input_a is None or self.input_b is None:
            return
        a = float(self.alpha)
        self.output_data = (1.0 - a) * self.input_a + a * self.input_b
```

### Node that loads external data in `init()`

```python
@node.input
class DataLoader(Node):
    output_data = OutputPort("Sample", np.ndarray)
    idx = Store(default=0)

    def init(self):
        # Heavy loading happens once
        self._data = np.load("data/my_dataset.npy")
        self.idx = 0

    def process(self):
        self.output_data = self._data[self.idx]
        self.idx = (self.idx + 1) % len(self._data)
```

Note: `self._data` (prefixed with `_`) is NOT a descriptor — it's a plain instance variable. It won't be persisted or shown in the UI. Use plain attributes for transient/large data.

---

## Important Rules

1. **Always check for `None` inputs** in `process()`. Unconnected or not-yet-computed ports are `None`.
2. **Don't modify input arrays in-place.** They may be shared references. Use `.copy()` if you need to mutate.
3. **Category decorator is mandatory.** Without `@node.input` / `@node.processing` / `@node.utility` / `@node.output` / `@node.custom(...)`, the class won't appear in the palette.
4. **Prefix internal state with `_`** (e.g., `self._weights`) for things that shouldn't be descriptors. The `NodeMeta` metaclass only collects attributes that are descriptor instances.
5. **numpy arrays in Stores get serialized to JSON.** This works fine for small arrays but avoid storing very large arrays in Store descriptors — use plain `self._attr` for those.
6. **`process()` must be implemented.** It's the only required method (will raise `NotImplementedError` if missing).
7. **`init()` is optional** but strongly recommended for any node that needs setup.
8. **Use `uv run`** for all commands: `uv run python main.py`, `uv sync`, `uv run pytest`, etc.
