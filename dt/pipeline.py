import asyncio, yaml, numpy as np
from .streaming.csv_replay import CSVReplayStream
from .streaming.mqtt_stream import MQTTStream
from .processing.features import build_features
from .processing.transforms import apply_transforms
from .twin.performance import PerformanceTwin
from .models.lgbm import LGBMRegressorWrapper
from .models.svr import SVRWrapper
from .sinks.console import ConsoleSink
from .sinks.csv import CSVSink
from .sinks.mqtt_sink import MQTTSink

import math
try:
    from .sinks.websocket import WebSocketSink
except Exception:
    WebSocketSink = None

class Pipeline:
    def __init__(self, cfg_path="config/config.yaml"):
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.sinks = []
        for sk in self.cfg.get("sinks", []):
            if sk["kind"] == "console":
                self.sinks.append(ConsoleSink())
            elif sk["kind"] == "websocket" and WebSocketSink is not None:
                self.sinks.append(WebSocketSink(sk.get("options", {}).get("url")))
            elif sk["kind"] == "csv":
                opts = sk.get("options", {})
                self.sinks.append(CSVSink(
                    path=opts.get("path", "out/predictions.csv"),
                    fields=opts.get("fields"),           # optional explicit column order
                    append=opts.get("append", True),
                ))
            elif sk["kind"] == "mqtt":
                opts = sk.get("options", {})
                self.sinks.append(MQTTSink(opts.get("broker"),opts.get("port")))

        if not self.sinks:
            self.sinks = [ConsoleSink()]

    def _make_stream(self):
        scfg = self.cfg["stream"]
        if scfg["kind"] == "csv_replay":
            return CSVReplayStream(
                path=scfg["options"]["path"],
                timestamp_col=scfg["options"]["timestamp_col"],
                speedup=scfg["options"].get("speedup", 60),
                tz=scfg["options"].get("time_col_tz", "UTC"),
                sep=scfg["options"].get("sep"),
                encoding=scfg["options"].get("encoding", "utf-8"),
                on_bad_lines=scfg["options"].get("on_bad_lines", "warn"),
                skiprows=scfg["options"].get("skiprows"),
                comment=scfg["options"].get("comment"),
                header_key=scfg["options"].get("header_key"),
                quotechar=scfg["options"].get("quotechar", '"'),
            )
        elif scfg["kind"] == "mqtt":
                return MQTTStream(
                scfg["options"]["broker"],
                scfg["options"]["port"],
                scfg["options"]["topics"]
            )
        raise NotImplementedError("Unknown stream kind: %s" % scfg["kind"])

    def _make_model(self, feature_names):
        mcfg = self.cfg["model"]
        if mcfg["kind"] == "lgbm":
            return LGBMRegressorWrapper(
                monotone_speed_feature=mcfg["options"].get("monotone_speed_feature", True),
                feature_order=feature_names
            )
        elif mcfg["kind"] == "svr":
            opts = mcfg.get("options", {})
            return SVRWrapper(
            feature_order=feature_names,
            kernel=opts.get("kernel", "rbf"),
            C=float(opts.get("C", 10.0)),
            epsilon=float(opts.get("epsilon", 0.1)),
            gamma=opts.get("gamma", "scale"),
            #degree=int(opts.get("degree", 3)), # only for poly
            coef0=float(opts.get("coef0", 0.0)),
            )
        raise NotImplementedError("Unknown model kind: %s" % mcfg["kind"])

    async def run(self):
        stream = self._make_stream()
        pcfg = self.cfg["processing"]
        twin = None

        # Warmup fit on first N rows
        trainer_X, trainer_y, feature_names = [], [], None
        i = 0
        async for frame in stream.stream():
            # 1) Clean the raw row using config rules
            raw = apply_transforms(frame.payload, self.cfg.get("processing", {}).get("transforms", []))
            if not raw:
                continue  # row dropped by a transform rule

            # 2) Build features from the cleaned row
            x = build_features(raw, frame.ts, pcfg["features"])
            if not x:
                continue

            # 3) Cast & validate target and primary speed
            y = raw.get(self.cfg["twin"]["target_col"])
            try:
                y = float(y)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(y):
                continue

            v = x.get("v")
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            x["v"] = v  # ensure Python float

            if feature_names is None:
                feature_names = sorted(list(x.keys()))
            trainer_X.append([x.get(f,0.0) for f in feature_names])
            trainer_y.append(y)
            i += 1
            if i >= 5000:    # warmup window
                model = self._make_model(feature_names)
                model.fit(trainer_X, trainer_y)
                twin  = PerformanceTwin(model, pcfg["guardrails"])
                self._model_name = self.cfg["model"]["kind"]  # e.g., "lgbm"
                break
        
        stream.close()
        # Rewind: new stream for live replay
        stream_live = self._make_stream()
        async for frame in stream_live.stream():
            # 1) Clean the raw row using config rules
            raw = apply_transforms(frame.payload, self.cfg.get("processing", {}).get("transforms", []))
            if not raw:
                continue

            # 2) Build features from the cleaned row
            x = build_features(raw, frame.ts, pcfg["features"])
            if not x:
                continue

            # 3) Validate/cast primary speed
            v = x.get("v")
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            x["v"] = v

            # 4) Optional actual (for PI): cast if present and finite
            y = raw.get(self.cfg["twin"]["target_col"])
            if y is not None:
                try:
                    y = float(y)
                    if not math.isfinite(y):
                        y = None
                except (TypeError, ValueError):
                    y = None

            out = twin.step(x, y)
            out.update({
                "ts": frame.ts.isoformat(),
                "v": x.get("v"),
                "y": y,
                "model": getattr(self, "_model_name", "unknown"),
            })
            for s in self.sinks:
                if isinstance(s, MQTTSink):
                    s.write(out,"ml/svrPredictions1")
                else:
                    s.write(out)

        stream.close()
