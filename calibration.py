"""
Post-hoc recalibration of expected points.

The engine's raw output is systematically over-spread: backtesting gave
`actual = 0.63 * pred + 0.98` on 2025-26, meaning the model over-rates exactly
the players it likes most. That is the mechanism behind poor captaincy picks,
because captaincy only cares about the top of the ranking.

This module fits a per-position correction on a rolling window of completed
gameweeks and applies it inside `predict()`. Isotonic regression is used where
there is enough data (it is monotone, so it never reorders players within a
position), falling back to a linear fit and then to the identity.
"""
import logging
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_SAMPLES_ISOTONIC = 300
MIN_SAMPLES_LINEAR = 50
# Samples from a single gameweek are not a calibration set: they carry one
# realisation of fixture and rotation noise, and fitting on them mostly
# compresses the forecast toward that week's mean.
MIN_GAMEWEEKS = 3
# Weight of the raw prediction retained through an isotonic fit, purely to
# break ties inside a plateau without materially shifting the calibration.
TIE_BREAK = 0.02


class PointsCalibrator:
    """Maps raw expected points to calibrated expected points, per position."""

    def __init__(self, window_gws: int = 8, method: str = "linear"):
        # "linear"  - strictly monotone, preserves ordering exactly. Default,
        #             because the tail is sparse and isotonic collapses it.
        # "isotonic"- flexible shape, but non-decreasing only: it maps wide
        #             ranges of raw values onto one plateau, which is fatal at
        #             the top of the ranking where captaincy is decided.
        # "none"    - identity, for measuring whether calibration helps at all.
        self.method = method
        self.window_gws = window_gws
        self._models: Dict[int, Any] = {}
        self._kind: Dict[int, str] = {}
        self.is_fitted = False
        self.report: Dict[int, Dict[str, float]] = {}

    def fit(self, samples: List[Dict[str, Any]]) -> "PointsCalibrator":
        """
        `samples` are dicts with keys: element_type, pred, actual.
        Only gameweeks strictly before the forecast target may be supplied -
        the caller owns that constraint.
        """
        from sklearn.isotonic import IsotonicRegression

        distinct_gws = {s.get("gw") for s in samples if s.get("gw") is not None}
        if distinct_gws and len(distinct_gws) < MIN_GAMEWEEKS:
            logger.info(
                "Skipping calibration: only %d gameweek(s) of history available "
                "(need %d).", len(distinct_gws), MIN_GAMEWEEKS,
            )
            self._models, self._kind, self.report = {}, {}, {}
            self.is_fitted = False
            return self

        by_pos: Dict[int, List] = {}
        for s in samples:
            by_pos.setdefault(int(s["element_type"]), []).append((float(s["pred"]), float(s["actual"])))

        self._models, self._kind, self.report = {}, {}, {}
        for pos, rows in by_pos.items():
            x = np.array([r[0] for r in rows], dtype=float)
            y = np.array([r[1] for r in rows], dtype=float)
            if len(x) < MIN_SAMPLES_LINEAR or np.allclose(x.std(), 0):
                continue

            slope, intercept = np.polyfit(x, y, 1)
            self.report[pos] = {
                "n": float(len(x)),
                "raw_slope": float(slope),
                "raw_intercept": float(intercept),
            }

            if self.method == "none":
                continue
            if self.method == "isotonic" and len(x) >= MIN_SAMPLES_ISOTONIC:
                # y_min=0 because FPL points below zero are rare enough that
                # allowing a negative floor destabilises the low end.
                iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip", increasing=True)
                iso.fit(x, y)
                self._models[pos] = iso
                self._kind[pos] = "isotonic"
            else:
                self._models[pos] = (float(slope), float(intercept))
                self._kind[pos] = "linear"

        self.is_fitted = bool(self._models) and self.method != "none"
        if self.is_fitted:
            logger.info(
                "Calibrator fitted: %s",
                {p: f"{self._kind[p]} n={int(self.report[p]['n'])} "
                    f"slope={self.report[p]['raw_slope']:.2f}" for p in sorted(self._models)},
            )
        return self

    def apply(self, raw_pred: float, element_type: int) -> float:
        """Calibrate one prediction. Identity when unfitted for that position."""
        if not self.is_fitted:
            return raw_pred
        model = self._models.get(int(element_type))
        if model is None:
            return raw_pred
        if self._kind[int(element_type)] == "isotonic":
            # Isotonic is monotone *non-decreasing*, so it happily maps a whole
            # range of raw values onto one plateau. That is fatal at the top of
            # the ranking, where captaincy and precision@k need to tell near-
            # equal players apart. Mixing a sliver of the raw prediction back in
            # keeps the fitted shape while restoring a strict ordering.
            base = float(model.predict([raw_pred])[0])
            return float(max(0.0, (1.0 - TIE_BREAK) * base + TIE_BREAK * raw_pred))
        slope, intercept = model
        return float(max(0.0, slope * raw_pred + intercept))

    def apply_many(self, preds: np.ndarray, element_type: int) -> np.ndarray:
        if not self.is_fitted:
            return preds
        model = self._models.get(int(element_type))
        if model is None:
            return preds
        if self._kind[int(element_type)] == "isotonic":
            base = model.predict(preds)
            return np.maximum(0.0, (1.0 - TIE_BREAK) * base + TIE_BREAK * preds)
        slope, intercept = model
        return np.maximum(0.0, slope * preds + intercept)
