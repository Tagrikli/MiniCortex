"""
ComputationThread — QThread that runs the Network execution loop.

After each step it writes display outputs into the shared buffer on BridgeAPI
and sets the dirty flag.  A QTimer in the main thread polls at the UI refresh
rate and reads the buffer when dirty.
"""

import time

from PySide6.QtCore import QThread, Signal

from axonforge.network.network import NetworkError

from .bridge import BridgeAPI


class ComputationThread(QThread):
    """Background thread that continuously executes network steps."""

    # Emitted when a node processing error occurs
    network_error = Signal(str, str, str, str)  # node_id, node_name, error, traceback

    def __init__(self, bridge: BridgeAPI, parent=None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self._stop_flag = False

    def run(self) -> None:
        network = self.bridge.network
        while not self._stop_flag:
            if network.running:
                try:
                    network.execute_step()

                    # Update actual Hz
                    now = time.monotonic()
                    last = getattr(self, "_last_step_time", None)
                    if last is not None:
                        dt = now - last
                        if dt > 0:
                            network.actual_hz = 1.0 / dt
                    self._last_step_time = now

                    # Snapshot display outputs into shared buffer
                    self.bridge.snapshot_displays()

                    # Throttle based on speed setting
                    speed = float(network.speed)
                    if not bool(getattr(network, "max_speed", False)):
                        self.msleep(int(1000.0 / max(speed, 1.0)))

                except NetworkError as e:
                    self.network_error.emit(e.node_id, e.node_name, str(e.error), e.traceback)
                    self.msleep(100)
                except Exception:
                    self.msleep(100)
            else:
                self.msleep(50)

    def request_stop(self) -> None:
        self._stop_flag = True
