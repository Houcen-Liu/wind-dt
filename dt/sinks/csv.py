# dt/sinks/csv.py
import os, csv
from typing import List, Optional
from .base import Sink

class CSVSink(Sink):
    """
    Simple CSV logger.
    - Writes one row per pipeline output.
    - Creates parent folder and header automatically.
    - Windows-safe newline handling.
    """
    def __init__(self, path: str, fields: Optional[List[str]] = None, append: bool = True):
        self.path = path
        self.fields = fields  # if None, infer from first message
        self.append = append
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._initialized = False

    def _ensure_writer(self, msg: dict):
        mode = "a" if (self.append and os.path.exists(self.path)) else "w"
        self._fh = open(self.path, mode, newline="", encoding="utf-8")
        if self.fields is None:
            # Infer stable column order from keys of the first message
            self.fields = list(msg.keys())
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields, extrasaction="ignore")
        # Write header if we're overwriting or file is empty
        if mode == "w" or os.stat(self.path).st_size == 0:
            self._writer.writeheader()
        self._initialized = True

    def write(self, msg: dict):
        if not self._initialized:
            self._ensure_writer(msg)
        try:
            self._writer.writerow(msg)
            self._fh.flush()
        except Exception:
            # avoid crashing the pipeline due to I/O hiccups
            pass

    def __del__(self):
        try:
            if getattr(self, "_fh", None):
                self._fh.close()
        except Exception:
            pass
