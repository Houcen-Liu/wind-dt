import asyncio
import csv
import pandas as pd

from .base import StreamProvider
from ..types import MeteoFrame

class CSVReplayStream(StreamProvider):
    def __init__(
        self,
        path,
        timestamp_col,
        speedup=60,
        tz="UTC",
        sep=None,
        encoding="utf-8",
        on_bad_lines="warn",
        skiprows=None,
        comment=None,
        header_key=None,
        quotechar='"',
    ):
        self.path = path
        self.timestamp_col = timestamp_col
        self.speedup = speedup
        self.tz = tz
        self.sep = sep
        self.encoding = encoding
        self.on_bad_lines = on_bad_lines
        self.skiprows = skiprows
        self.comment = comment
        self.header_key = header_key or timestamp_col
        self.quotechar = quotechar

    def _find_header_row(self) -> int:
        """
        Return the 0-based line index of the real CSV header containing `header_key`.
        If skiprows is provided in config, it takes precedence.
        """
        if self.skiprows is not None:
            return int(self.skiprows)

        key = (self.header_key or "").strip().lower()
        delim = self.sep or ","

        with open(self.path, "r", encoding=self.encoding, errors="ignore", newline="") as f:
            for idx, line in enumerate(f):
                # Optionally ignore comment lines
                if self.comment and line.lstrip().startswith(self.comment):
                    continue
                if key and key in line.lower() and line.count(delim) >= 5:
                    return idx
        return 0  # fallback

    def _read_df(self) -> pd.DataFrame:
        header_row = self._find_header_row()

        df = pd.read_csv(
            self.path,
            sep=(self.sep or ","),
            engine="python",                 # safer with embedded commas/quotes
            encoding=self.encoding,
            on_bad_lines=self.on_bad_lines,  # "skip"/"warn"
            skiprows=header_row,             # skip description lines before header
            header=0,                        # next row is header
            comment=self.comment,            # ignore trailing comment lines
            #low_memory=False,
            quoting=csv.QUOTE_MINIMAL,
            quotechar=self.quotechar,
            doublequote=True,
        )

        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{self.timestamp_col}' not found. "
                f"First columns: {list(df.columns)[:8]}"
            )

        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True, errors="coerce")
        df = df.dropna(subset=[self.timestamp_col]).sort_values(self.timestamp_col)

        # Convert numeric columns (leave timestamp as-is)
        for c in df.columns:
            if c == self.timestamp_col:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    async def stream(self):
        df = self._read_df()

        # infer interval for replay speed
        if len(df) < 2:
            sleep = 0.5
        else:
            dt = (df[self.timestamp_col].iloc[1] - df[self.timestamp_col].iloc[0]).total_seconds()
            sleep = max(0.0, dt / max(1, self.speedup))

        # emit rows
        for _, row in df.iterrows():
            payload = {k: row[k] for k in df.columns if k != self.timestamp_col}
            yield MeteoFrame(ts=row[self.timestamp_col].to_pydatetime(), payload=payload)
            await asyncio.sleep(sleep)
    
    def close(self):
        return
