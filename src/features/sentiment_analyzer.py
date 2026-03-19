"""
Local sentiment analyzer using a Hugging Face sequence classification model.
Falls back to a deterministic heuristic if the model cannot be loaded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Local-first sentiment analyzer with a deterministic fallback."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or "ai4bharat/IndicBERTv2-alpha-SentimentClassification"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._mode = "lazy"

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(
                self.device
            )
            self._model.eval()
            self._mode = "local_transformers"
            logger.info("Sentiment model loaded: %s", self.model_name)
        except Exception as exc:
            self._mode = "heuristic_fallback"
            logger.warning("Falling back to heuristic sentiment mode: %s", exc)

    def analyze_sentiment(self, text: str, threshold: float = 0.7) -> Dict[str, Any]:
        self._ensure_model()

        if self._mode == "local_transformers" and self._model is not None and self._tokenizer is not None:
            try:
                inputs = self._tokenizer(
                    text,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    logits = self._model(**inputs).logits[0]
                    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

                predicted_index = int(torch.argmax(logits).item())
                label_map = self._model.config.id2label or {
                    0: "negative",
                    1: "neutral",
                    2: "positive"
                }
                label = str(label_map.get(predicted_index, "neutral")).lower()
                confidence = float(probabilities[predicted_index])

                return {
                    "sentiment": label,
                    "confidence": confidence,
                    "threshold": threshold,
                    "above_threshold": confidence >= threshold,
                    "language": "auto",
                    "model": self.model_name,
                    "mode": self._mode,
                    "distribution": {
                        str(label_map.get(index, index)).lower(): float(score)
                        for index, score in enumerate(probabilities)
                    }
                }
            except Exception as exc:
                logger.warning("Model inference failed, using heuristic sentiment: %s", exc)
                self._mode = "heuristic_fallback"

        return self._fallback_sentiment(text=text, threshold=threshold)

    def _fallback_sentiment(self, text: str, threshold: float) -> Dict[str, Any]:
        positive_markers = {"good", "great", "excellent", "resolved", "thanks", "helpful", "happy"}
        negative_markers = {"bad", "delay", "problem", "issue", "failed", "angry", "slow"}

        lower_text = text.lower()
        positive_hits = sum(marker in lower_text for marker in positive_markers)
        negative_hits = sum(marker in lower_text for marker in negative_markers)

        if positive_hits > negative_hits:
            sentiment = "positive"
            confidence = 0.74
        elif negative_hits > positive_hits:
            sentiment = "negative"
            confidence = 0.74
        else:
            sentiment = "neutral"
            confidence = 0.62

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "threshold": threshold,
            "above_threshold": confidence >= threshold,
            "language": "auto",
            "model": "heuristic_fallback",
            "mode": self._mode,
            "distribution": {
                "positive": 0.7 if sentiment == "positive" else 0.15,
                "neutral": 0.62 if sentiment == "neutral" else 0.15,
                "negative": 0.7 if sentiment == "negative" else 0.15
            }
        }
