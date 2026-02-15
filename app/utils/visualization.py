"""
Result visualization utilities.
Draws bounding boxes, masks, and labels on images.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import random


# Color palette for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255),
]


def get_color(idx: int) -> Tuple[int, int, int]:
    """Get a color from the palette, cycling if needed."""
    return COLORS[idx % len(COLORS)]


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_width: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.
    
    Args:
        image: Input image (BGR)
        detections: List of detection dicts with 'bbox', 'class_name'/'label', 'confidence'
        show_labels: Whether to draw class labels
        show_confidence: Whether to show confidence scores
        line_width: Bounding box line width
        
    Returns:
        Annotated image
    """
    annotated = image.copy()

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(c) for c in bbox]
        color = get_color(det.get("id", 0))

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)

        if show_labels:
            # Build label text
            label = det.get("class_name") or det.get("label", "object")
            if show_confidence:
                conf = det.get("confidence", 0)
                label = f"{label}: {conf:.2f}"

            # Draw label background
            font_scale = 0.5
            thickness = 1
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
            )

    return annotated


def draw_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.4,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Overlay segmentation masks on an image with transparency.
    
    Args:
        image: Input image (BGR)
        masks: List of binary masks (same H,W as image)
        alpha: Transparency of mask overlay (0=invisible, 1=opaque)
        labels: Optional labels for each mask
        
    Returns:
        Image with colored mask overlays
    """
    annotated = image.copy()
    overlay = image.copy()

    for i, mask in enumerate(masks):
        color = get_color(i)

        # Apply colored mask
        overlay[mask > 0] = color

        # Draw contours for mask boundary
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, 2)

    # Blend overlay with original
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

    return annotated


def create_comparison_image(
    original: np.ndarray,
    annotated: np.ndarray,
) -> np.ndarray:
    """Create side-by-side comparison of original and annotated images."""
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]

    # Resize to same height
    target_h = max(h1, h2)
    if h1 != target_h:
        scale = target_h / h1
        original = cv2.resize(original, (int(w1 * scale), target_h))
    if h2 != target_h:
        scale = target_h / h2
        annotated = cv2.resize(annotated, (int(w2 * scale), target_h))

    # Concatenate horizontally
    comparison = np.hstack([original, annotated])

    return comparison
