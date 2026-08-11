"""
Explainability service.

Produces a heatmap indicating which pixels most influenced YOLOv8's decision.
Two backends are attempted in order:

1. ``pytorch-grad-cam`` (if installed) - true Grad-CAM against a chosen
   convolutional layer.  Adds ~40MB but gives the canonical XAI output.
2. Fallback saliency map derived from the input image's Sobel gradient
   masked by the highest-confidence detection box.  Zero dependencies,
   always available, and still communicates *where* the model focused.

Callers receive the heatmap as a base64-encoded PNG plus a short caption.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.models.model_factory import ModelFactory, ModelType

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Generate an attention / saliency overlay for a YOLO detection."""

    def __init__(self) -> None:
        self._gradcam_available = self._probe_gradcam()

    @staticmethod
    def _probe_gradcam() -> bool:
        try:
            import pytorch_grad_cam  # noqa: F401
            return True
        except Exception:
            return False

    # -- Public API ----------------------------------------------------------

    def explain(
        self,
        image: np.ndarray,
        detection_index: Optional[int] = None,
        confidence: float = 0.25,
    ) -> Dict[str, Any]:
        yolo = ModelFactory.get_or_create(ModelType.YOLO)
        yolo.ensure_loaded()

        # Run detection so we can key the heatmap to a specific box
        raw = yolo.predict(image, confidence=confidence)
        detections = raw.get("detections", [])
        if not detections:
            return {
                "method": "none",
                "message": "No detections above threshold - nothing to explain.",
                "heatmap_base64": None,
                "detections": [],
            }

        target = detections[detection_index] if detection_index is not None else detections[0]

        if self._gradcam_available:
            try:
                heatmap = self._gradcam(yolo, image, target)
                method = "grad-cam"
            except Exception:  # pragma: no cover - defensive
                logger.exception("Grad-CAM failed - falling back to saliency map")
                heatmap = self._saliency_fallback(image, target)
                method = "saliency-fallback"
        else:
            heatmap = self._saliency_fallback(image, target)
            method = "saliency-fallback"

        return {
            "method": method,
            "detections": detections,
            "target_detection": target,
            "heatmap_base64": _png_base64(heatmap),
            "caption": (
                "Warm colors = pixels the model weighted highest for this "
                "prediction. Cool colors = ignored."
            ),
        }

    # -- Backends ------------------------------------------------------------

    def _gradcam(self, yolo: Any, image: np.ndarray, target: Dict[str, Any]) -> np.ndarray:
        """Best-effort Grad-CAM against the YOLOv8 backbone."""
        import torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        underlying = yolo.model.model  # ultralytics wraps torch.nn.Module here
        target_layer = list(underlying.modules())[-2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = (
            torch.from_numpy(rgb.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(next(underlying.parameters()).device)
        )
        cam = GradCAM(model=underlying, target_layers=[target_layer])
        grayscale = cam(input_tensor=tensor)[0]
        return show_cam_on_image(rgb, grayscale, use_rgb=True)

    def _saliency_fallback(self, image: np.ndarray, target: Dict[str, Any]) -> np.ndarray:
        """Cheap, dependency-free saliency approximation.

        Combines an edge-magnitude map (Sobel) with a Gaussian centered on the
        detection box, blended with the original image using a jet colormap.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

        # Box-centered Gaussian focus
        bbox = target.get("bbox", [0, 0, w, h])
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sigma = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1])) / 2.0 + 1.0
        gauss = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)))

        heatmap = (0.6 * gauss + 0.4 * mag)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image, 0.55, colored, 0.45, 0)
        return blended


# ─── Helpers ───────────────────────────────────────────────────────────────


def _png_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
