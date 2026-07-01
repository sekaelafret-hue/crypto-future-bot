"""ML Decision Engine using Random Forest for trade signal generation."""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.config import MODEL_PATH

logger = logging.getLogger(__name__)

SIGNAL_MAP = {0: "HOLD", 1: "LONG", 2: "SHORT"}
SIGNAL_MAP_INV = {"HOLD": 0, "LONG": 1, "SHORT": 2}


@dataclass
class Signal:
    action: str  # LONG, SHORT, HOLD
    confidence: float
    reasoning: str
    features: dict

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "macd_histogram",
    "macd_signal",
    "macd_cross",
    "ema_cross",
    "price_vs_ema50_pct",
    "atr",
    "price_change_pct",
    "volume_ratio",
    "bb_position",
]


class MLEngine:
    """Random Forest-based signal generator."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[RandomForestClassifier] = None
        self._load_or_build_model()

    def _load_or_build_model(self):
        """Load saved model or build a fresh one from synthetic data."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info("Loaded ML model from %s", self.model_path)
                return
            except Exception as e:
                logger.warning("Failed to load model: %s. Rebuilding.", e)

        logger.info("Building ML model from synthetic training data...")
        self._train_on_synthetic_data()

    def _generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate realistic synthetic TA feature data for training."""
        rng = np.random.RandomState(42)

        data = {
            "rsi": rng.uniform(15, 85, n_samples),
            "macd": rng.normal(0, 1, n_samples),
            "macd_histogram": rng.normal(0, 0.5, n_samples),
            "macd_signal": rng.normal(0, 0.8, n_samples),
            "macd_cross": rng.choice([0, 1], n_samples),
            "ema_cross": rng.choice([0, 1], n_samples),
            "price_vs_ema50_pct": rng.normal(0, 3, n_samples),
            "atr": rng.uniform(0.5, 5, n_samples),
            "price_change_pct": rng.normal(0, 2, n_samples),
            "volume_ratio": rng.uniform(0.3, 3.0, n_samples),
            "bb_position": rng.uniform(-0.1, 1.1, n_samples),
        }

        df = pd.DataFrame(data)

        # Generate labels based on indicator rules (with noise)
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            score = 0.0
            # RSI signal
            if df["rsi"].iloc[i] < 30:
                score += 2
            elif df["rsi"].iloc[i] > 70:
                score -= 2

            # MACD crossover
            if df["macd_cross"].iloc[i] == 1 and df["macd_histogram"].iloc[i] > 0:
                score += 1.5
            elif df["macd_cross"].iloc[i] == 0 and df["macd_histogram"].iloc[i] < 0:
                score -= 1.5

            # EMA alignment
            if df["ema_cross"].iloc[i] == 1 and df["price_vs_ema50_pct"].iloc[i] > 0:
                score += 1
            elif (
                df["ema_cross"].iloc[i] == 0
                and df["price_vs_ema50_pct"].iloc[i] < 0
            ):
                score -= 1

            # BB position
            if df["bb_position"].iloc[i] < 0.2:
                score += 0.5
            elif df["bb_position"].iloc[i] > 0.8:
                score -= 0.5

            # Volume confirmation
            if df["volume_ratio"].iloc[i] > 1.5:
                score *= 1.2

            # Add noise
            score += rng.normal(0, 0.5)

            if score > 1.5:
                labels[i] = SIGNAL_MAP_INV["LONG"]
            elif score < -1.5:
                labels[i] = SIGNAL_MAP_INV["SHORT"]
            else:
                labels[i] = SIGNAL_MAP_INV["HOLD"]

        df["label"] = labels
        return df

    def _train_on_synthetic_data(self):
        """Train the Random Forest on synthetic data."""
        df = self._generate_synthetic_data()
        x = df[FEATURE_COLUMNS]
        y = df["label"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(x_train, y_train)

        accuracy = self.model.score(x_test, y_test)
        logger.info("Model trained. Test accuracy: %.3f", accuracy)

        y_pred = self.model.predict(x_test)
        report = classification_report(
            y_test, y_pred, target_names=["HOLD", "LONG", "SHORT"]
        )
        logger.info("Classification report:\n%s", report)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info("Model saved to %s", self.model_path)

    def train_on_real_data(self, feature_df: pd.DataFrame):
        """Retrain the model on real feature data (from TechnicalAnalyzer)."""
        required = set(FEATURE_COLUMNS) | {"future_return"}
        available = set(feature_df.columns)
        if not required.issubset(available):
            missing = required - available
            logger.warning("Missing columns for training: %s", missing)
            return

        df = feature_df.dropna().copy()
        if len(df) < 50:
            logger.warning("Not enough data to retrain (%d rows)", len(df))
            return

        # Label based on future return
        labels = np.where(
            df["future_return"] > 0.002,
            SIGNAL_MAP_INV["LONG"],
            np.where(
                df["future_return"] < -0.002,
                SIGNAL_MAP_INV["SHORT"],
                SIGNAL_MAP_INV["HOLD"],
            ),
        )

        x = df[FEATURE_COLUMNS]
        y = labels

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(x, y)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info("Model retrained on %d real samples", len(df))

    def predict(self, features: dict) -> Signal:
        """Generate a trading signal from current indicator features."""
        if self.model is None:
            return Signal(
                action="HOLD",
                confidence=0.0,
                reasoning="Model not loaded",
                features=features,
            )

        feature_vector = []
        for col in FEATURE_COLUMNS:
            val = features.get(col, 0)
            if pd.isna(val):
                val = 0
            feature_vector.append(float(val))

        x = np.array([feature_vector])
        prediction = self.model.predict(x)[0]
        probabilities = self.model.predict_proba(x)[0]

        action = SIGNAL_MAP[prediction]
        confidence = float(probabilities[prediction])

        reasoning = self._build_reasoning(features, action, confidence, probabilities)

        return Signal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            features=features,
        )

    def _build_reasoning(
        self,
        features: dict,
        action: str,
        confidence: float,
        probabilities: np.ndarray,
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        parts = []

        rsi = features.get("rsi", 50)
        if rsi < 30:
            parts.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            parts.append(f"RSI overbought ({rsi:.1f})")
        else:
            parts.append(f"RSI neutral ({rsi:.1f})")

        macd_hist = features.get("macd_histogram", 0)
        if macd_hist > 0:
            parts.append("MACD bullish momentum")
        elif macd_hist < 0:
            parts.append("MACD bearish momentum")

        ema_cross = features.get("ema_cross", 0)
        if ema_cross == 1:
            parts.append("EMA9 > EMA21 (bullish)")
        else:
            parts.append("EMA9 < EMA21 (bearish)")

        bb_pos = features.get("bb_position", 0.5)
        if bb_pos < 0.2:
            parts.append("Near lower Bollinger Band")
        elif bb_pos > 0.8:
            parts.append("Near upper Bollinger Band")

        vol_ratio = features.get("volume_ratio", 1)
        if vol_ratio > 1.5:
            parts.append(f"High volume ({vol_ratio:.1f}x avg)")

        prob_str = (
            f"HOLD: {probabilities[0]:.0%}, "
            f"LONG: {probabilities[1]:.0%}, "
            f"SHORT: {probabilities[2]:.0%}"
        )
        parts.append(f"Model probabilities: [{prob_str}]")

        return f"{action} (conf: {confidence:.0%}) | " + " | ".join(parts)
