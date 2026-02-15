"""
Annotation utilities for object detection workflows.
Demonstrates understanding of annotation-model relationship and production pipelines.

Key Features:
- Export predictions to multiple annotation formats (COCO, YOLO, Label Studio)
- Quality control metrics (IoU, precision, recall)
- Active learning for efficient annotation
- Integration with Google Cloud Vertex AI
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnnotationExporter:
    """
    Converts model predictions to standard annotation formats.
    Essential for creating training datasets and annotation workflows.
    
    Supports:
    - COCO format (used by Label Studio, Vertex AI, Detectron2)
    - YOLO format (used by YOLOv8 training)
    - Label Studio JSON (for annotation UI)
    - Pascal VOC XML (classical format)
    """
    
    def __init__(self, dataset_name: str = "object_detection_dataset", categories: Optional[List[Dict]] = None):
        """
        Initialize exporter with dataset configuration.
        
        Args:
            dataset_name: Name of the dataset
            categories: List of category dicts with 'id' and 'name'
        """
        self.dataset_name = dataset_name
        self.categories = categories or [
            {"id": 1, "name": "promotional_banner", "supercategory": "banner"},
            {"id": 2, "name": "payment_banner", "supercategory": "banner"},
            {"id": 3, "name": "offer_banner", "supercategory": "banner"},
            {"id": 4, "name": "navigation_banner", "supercategory": "banner"},
            {"id": 5, "name": "kyc_banner", "supercategory": "banner"},
            {"id": 6, "name": "generic_object", "supercategory": "object"},
        ]
        self.category_name_to_id = {cat['name']: cat['id'] for cat in self.categories}
    
    def to_coco_format(
        self, 
        detections: List[Dict], 
        image_info: Dict,
        include_confidence: bool = True
    ) -> Dict:
        """
        Export to COCO format (JSON).
        
        COCO is the standard for:
        - Google Cloud Vertex AI Data Labeling
        - Detectron2 training
        - Label Studio import
        - Most research benchmarks
        
        Args:
            detections: List of detection dicts with 'bbox', 'class', 'confidence'
            image_info: Dict with 'id', 'file_name', 'width', 'height'
            include_confidence: Whether to include confidence scores
            
        Returns:
            COCO-format dictionary
            
        Example:
            >>> detections = [
            ...     {'bbox': [100, 200, 300, 400], 'class': 'promotional_banner', 'confidence': 0.95}
            ... ]
            >>> image_info = {'id': 1, 'file_name': 'banner.jpg', 'width': 1080, 'height': 1920}
            >>> coco = exporter.to_coco_format(detections, image_info)
        """
        coco_annotations = []
        
        for idx, det in enumerate(detections):
            bbox = det['bbox']  # Expected: [x, y, width, height]
            
            # Handle different bbox formats
            if len(bbox) == 4:
                x, y, w, h = bbox
            else:
                logger.warning(f"Invalid bbox format: {bbox}")
                continue
            
            annotation = {
                "id": idx + 1,
                "image_id": image_info['id'],
                "category_id": self._get_category_id(det['class']),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],  # Can add polygon segmentation if available
            }
            
            if include_confidence and 'confidence' in det:
                annotation['score'] = float(det['confidence'])
            
            coco_annotations.append(annotation)
        
        coco_output = {
            "info": {
                "description": self.dataset_name,
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": [image_info],
            "annotations": coco_annotations,
            "categories": self.categories
        }
        
        return coco_output
    
    def to_yolo_format(
        self, 
        detections: List[Dict], 
        img_width: int, 
        img_height: int,
        normalize: bool = True
    ) -> str:
        """
        Export to YOLO format (text file, one line per detection).
        
        Format: <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1] if normalize=True
        
        Args:
            detections: List of detection dicts
            img_width: Image width in pixels
            img_height: Image height in pixels
            normalize: Whether to normalize coordinates to [0, 1]
            
        Returns:
            String with YOLO format annotations (one line per detection)
            
        Example Output:
            0 0.5 0.3 0.2 0.15
            1 0.7 0.8 0.1 0.1
        """
        yolo_lines = []
        
        for det in detections:
            x, y, w, h = det['bbox']  # Assuming [x, y, width, height]
            
            # Convert to YOLO center format
            x_center = x + w / 2
            y_center = y + h / 2
            
            if normalize:
                x_center /= img_width
                y_center /= img_height
                w /= img_width
                h /= img_height
            
            # YOLO uses 0-indexed class IDs
            class_id = self._get_category_id(det['class']) - 1
            
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )
        
        return "\n".join(yolo_lines)
    
    def to_label_studio(
        self, 
        detections: List[Dict], 
        image_url: str,
        model_version: Optional[str] = None
    ) -> Dict:
        """
        Export to Label Studio JSON format for annotation UI.
        
        Label Studio is popular for:
        - Human-in-the-loop annotation
        - Model-assisted labeling
        - Quality review workflows
        
        Args:
            detections: List of detection dicts
            image_url: URL or path to image
            model_version: Optional model identifier for predictions
            
        Returns:
            Label Studio task dictionary
        """
        results = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            
            result = {
                "value": {
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                    "rotation": 0,
                    "rectanglelabels": [det['class']]
                },
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "prediction",  # Indicates model-generated, needs review
            }
            
            if 'confidence' in det:
                result['score'] = float(det['confidence'])
            
            results.append(result)
        
        task = {
            "data": {"image": image_url},
            "predictions": [{
                "result": results,
                "model_version": model_version or "unknown"
            }]
        }
        
        return task
    
    def to_pascal_voc(
        self, 
        detections: List[Dict], 
        image_info: Dict
    ) -> str:
        """
        Export to Pascal VOC XML format (classical format).
        
        Args:
            detections: List of detection dicts
            image_info: Dict with 'file_name', 'width', 'height', 'depth'
            
        Returns:
            XML string in Pascal VOC format
        """
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        
        annotation = Element('annotation')
        
        # Add image info
        SubElement(annotation, 'folder').text = 'images'
        SubElement(annotation, 'filename').text = str(image_info.get('file_name', 'unknown.jpg'))
        
        size = SubElement(annotation, 'size')
        SubElement(size, 'width').text = str(image_info['width'])
        SubElement(size, 'height').text = str(image_info['height'])
        SubElement(size, 'depth').text = str(image_info.get('depth', 3))
        
        # Add detections
        for det in detections:
            obj = SubElement(annotation, 'object')
            SubElement(obj, 'name').text = det['class']
            SubElement(obj, 'pose').text = 'Unspecified'
            SubElement(obj, 'truncated').text = '0'
            SubElement(obj, 'difficult').text = '0'
            
            x, y, w, h = det['bbox']
            bndbox = SubElement(obj, 'bndbox')
            SubElement(bndbox, 'xmin').text = str(int(x))
            SubElement(bndbox, 'ymin').text = str(int(y))
            SubElement(bndbox, 'xmax').text = str(int(x + w))
            SubElement(bndbox, 'ymax').text = str(int(y + h))
        
        # Pretty print
        xml_str = minidom.parseString(tostring(annotation)).toprettyxml(indent="  ")
        return xml_str
    
    def batch_export(
        self, 
        predictions: List[Dict], 
        output_dir: Path,
        formats: List[str] = ['coco', 'yolo', 'label_studio']
    ):
        """
        Export multiple predictions in batch to specified formats.
        
        Args:
            predictions: List of dicts with 'image_path', 'detections', 'image_info'
            output_dir: Directory to save exported annotations
            formats: List of formats to export ('coco', 'yolo', 'label_studio', 'voc')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            fmt_dir = output_dir / fmt
            fmt_dir.mkdir(exist_ok=True)
        
        logger.info(f"Exporting {len(predictions)} predictions to {output_dir}")
        
        # COCO: Single JSON file with all annotations
        if 'coco' in formats:
            all_images = []
            all_annotations = []
            ann_id = 1
            
            for idx, pred in enumerate(predictions):
                img_info = pred['image_info']
                img_info['id'] = idx + 1
                all_images.append(img_info)
                
                for det in pred['detections']:
                    ann = {
                        "id": ann_id,
                        "image_id": idx + 1,
                        "category_id": self._get_category_id(det['class']),
                        "bbox": det['bbox'],
                        "area": det['bbox'][2] * det['bbox'][3],
                        "iscrowd": 0,
                    }
                    if 'confidence' in det:
                        ann['score'] = det['confidence']
                    all_annotations.append(ann)
                    ann_id += 1
            
            coco_dataset = {
                "info": {
                    "description": self.dataset_name,
                    "version": "1.0",
                    "date_created": datetime.now().isoformat()
                },
                "images": all_images,
                "annotations": all_annotations,
                "categories": self.categories
            }
            
            coco_file = output_dir / 'coco' / 'annotations.json'
            with open(coco_file, 'w') as f:
                json.dump(coco_dataset, f, indent=2)
            logger.info(f"Saved COCO format to {coco_file}")
        
        # YOLO: One text file per image
        if 'yolo' in formats:
            for pred in predictions:
                img_path = Path(pred['image_path'])
                txt_file = output_dir / 'yolo' / f"{img_path.stem}.txt"
                
                yolo_str = self.to_yolo_format(
                    pred['detections'],
                    pred['image_info']['width'],
                    pred['image_info']['height']
                )
                
                with open(txt_file, 'w') as f:
                    f.write(yolo_str)
            logger.info(f"Saved {len(predictions)} YOLO label files")
        
        # Label Studio: One JSON per image or combined
        if 'label_studio' in formats:
            tasks = []
            for pred in predictions:
                task = self.to_label_studio(pred['detections'], pred['image_path'])
                tasks.append(task)
            
            ls_file = output_dir / 'label_studio' / 'tasks.json'
            with open(ls_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            logger.info(f"Saved Label Studio format to {ls_file}")
    
    def _get_category_id(self, class_name: str) -> int:
        """Get category ID from class name."""
        return self.category_name_to_id.get(class_name, 6)  # Default to 'generic_object'


class AnnotationQualityChecker:
    """
    Quality control for annotations.
    Calculates metrics to ensure annotation consistency and accuracy.
    """
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1, box2: Boxes in [x, y, width, height] format
            
        Returns:
            IoU score in [0, 1]
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def calculate_annotation_metrics(
        self, 
        ground_truth: List[Dict], 
        predictions: List[Dict],
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        Calculate comprehensive annotation quality metrics.
        
        Metrics include:
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1 Score: Harmonic mean of precision and recall
        - Average IoU: Mean IoU of matched boxes
        
        Args:
            ground_truth: List of ground truth detections
            predictions: List of predicted detections
            iou_threshold: Minimum IoU to consider a match
            
        Returns:
            Dict with quality metrics
        """
        tp, fp, fn, ious = self._match_detections(ground_truth, predictions, iou_threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = np.mean(ious) if ious else 0
        
        metrics = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "average_iou": round(avg_iou, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "num_ground_truth": len(ground_truth),
            "num_predictions": len(predictions),
        }
        
        return metrics
    
    def inter_annotator_agreement(
        self, 
        annotations_list: List[List[Dict]],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Calculate inter-annotator agreement (IAA).
        Measures consistency between multiple annotators on same images.
        
        Args:
            annotations_list: List of annotation lists from different annotators
            iou_threshold: Minimum IoU to consider boxes as matching
            
        Returns:
            Agreement score in [0, 1]
        """
        if len(annotations_list) < 2:
            return 1.0
        
        total_agreement = 0
        num_comparisons = 0
        
        # Compare each pair of annotators
        for i in range(len(annotations_list)):
            for j in range(i + 1, len(annotations_list)):
                ann1 = annotations_list[i]
                ann2 = annotations_list[j]
                
                tp, fp, fn, _ = self._match_detections(ann1, ann2, iou_threshold)
                
                # Agreement = matches / total boxes
                agreement = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1
                total_agreement += agreement
                num_comparisons += 1
        
        return total_agreement / num_comparisons if num_comparisons > 0 else 1.0
    
    def _match_detections(
        self, 
        ground_truth: List[Dict], 
        predictions: List[Dict],
        iou_threshold: float = 0.5
    ) -> Tuple[int, int, int, List[float]]:
        """
        Match predictions to ground truth boxes.
        
        Returns:
            (true_positives, false_positives, false_negatives, iou_scores)
        """
        tp = 0
        fp = 0
        fn = len(ground_truth)
        matched_ious = []
        
        matched_gt = set()
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(ground_truth):
                if idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                
                if iou > best_iou and pred['class'] == gt['class']:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                fn -= 1
                matched_gt.add(best_gt_idx)
                matched_ious.append(best_iou)
            else:
                fp += 1
        
        return tp, fp, fn, matched_ious


class ActiveLearningSelector:
    """
    Select most valuable samples for annotation using active learning strategies.
    
    Reduces annotation cost by focusing on:
    - Uncertain predictions (model is confused)
    - Complex scenes (many objects)
    - Edge cases (unusual characteristics)
    - Diverse samples (representative coverage)
    """
    
    def select_for_annotation(
        self, 
        predictions: List[Dict], 
        n_samples: int = 100,
        strategy: str = 'combined'
    ) -> List[str]:
        """
        Select most valuable images for human annotation.
        
        Strategies:
        - 'uncertainty': Low confidence predictions
        - 'complexity': Scenes with many objects
        - 'edge_cases': Unusual characteristics
        - 'diversity': Representative sampling
        - 'combined': Mix of all strategies (default)
        
        Args:
            predictions: List of prediction dicts with 'image_path', 'detections', 'confidences'
            n_samples: Number of samples to select
            strategy: Selection strategy
            
        Returns:
            List of image paths selected for annotation
        """
        if strategy == 'uncertainty':
            return self._uncertainty_sampling(predictions, n_samples)
        elif strategy == 'complexity':
            return self._complexity_sampling(predictions, n_samples)
        elif strategy == 'edge_cases':
            return self._edge_case_sampling(predictions, n_samples)
        elif strategy == 'diversity':
            return self._diversity_sampling(predictions, n_samples)
        else:  # combined
            return self._combined_sampling(predictions, n_samples)
    
    def _uncertainty_sampling(self, predictions: List[Dict], n: int) -> List[str]:
        """Select images with lowest confidence predictions."""
        uncertainty_scores = []
        
        for pred in predictions:
            if pred.get('confidences'):
                min_conf = min(pred['confidences'])
                avg_conf = np.mean(pred['confidences'])
                # Lower is more uncertain
                uncertainty = 1 - (min_conf * 0.7 + avg_conf * 0.3)
                uncertainty_scores.append((pred['image_path'], uncertainty))
        
        uncertainty_scores.sort(key=lambda x: x[1], reverse=True)
        return [img for img, _ in uncertainty_scores[:n]]
    
    def _complexity_sampling(self, predictions: List[Dict], n: int) -> List[str]:
        """Select images with most detections (complex scenes)."""
        complexity_scores = [
            (pred['image_path'], len(pred.get('detections', [])))
            for pred in predictions
        ]
        complexity_scores.sort(key=lambda x: x[1], reverse=True)
        return [img for img, _ in complexity_scores[:n]]
    
    def _edge_case_sampling(self, predictions: List[Dict], n: int) -> List[str]:
        """Select edge cases: unusual sizes, aspect ratios, or detection counts."""
        edge_cases = []
        
        for pred in predictions:
            score = 0
            detections = pred.get('detections', [])
            
            # Unusual detection count
            num_det = len(detections)
            if num_det == 0 or num_det > 20:
                score += 2
            
            # Unusual box characteristics
            for det in detections:
                w, h = det['bbox'][2], det['bbox'][3]
                aspect_ratio = w / h if h > 0 else 0
                
                # Extreme aspect ratios
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    score += 1
                
                # Very small or very large boxes
                area = w * h
                if area < 100 or area > 1000000:
                    score += 1
            
            if score > 0:
                edge_cases.append((pred['image_path'], score))
        
        edge_cases.sort(key=lambda x: x[1], reverse=True)
        return [img for img, _ in edge_cases[:n]]
    
    def _diversity_sampling(self, predictions: List[Dict], n: int) -> List[str]:
        """
        Select diverse samples using clustering.
        Requires feature extraction (not implemented here, placeholder).
        """
        # In production: extract image features, cluster, sample from each cluster
        # For now, use stratified sampling by detection count
        
        # Group by detection count bins
        bins = {}
        for pred in predictions:
            count = len(pred.get('detections', []))
            bin_key = count // 5  # Bins: 0-4, 5-9, 10-14, etc.
            if bin_key not in bins:
                bins[bin_key] = []
            bins[bin_key].append(pred['image_path'])
        
        # Sample from each bin
        selected = []
        samples_per_bin = max(1, n // len(bins))
        
        for images in bins.values():
            selected.extend(images[:samples_per_bin])
        
        return selected[:n]
    
    def _combined_sampling(self, predictions: List[Dict], n: int) -> List[str]:
        """Combine multiple strategies."""
        n_per_strategy = n // 3
        
        uncertain = self._uncertainty_sampling(predictions, n_per_strategy)
        complex_scenes = self._complexity_sampling(predictions, n_per_strategy)
        edge_cases = self._edge_case_sampling(predictions, n_per_strategy)
        
        # Remove duplicates, maintain order
        selected = []
        seen = set()
        
        for img in uncertain + complex_scenes + edge_cases:
            if img not in seen:
                selected.append(img)
                seen.add(img)
        
        return selected[:n]


def create_annotation_task(
    image_paths: List[str],
    output_file: Path,
    task_name: str = "annotation_task",
    instructions: str = "Label all objects in the image"
) -> Dict:
    """
    Create an annotation task configuration.
    
    Args:
        image_paths: List of image paths to annotate
        output_file: Where to save the task configuration
        task_name: Name of annotation task
        instructions: Instructions for annotators
        
    Returns:
        Task configuration dict
    """
    task_config = {
        "task_name": task_name,
        "created_at": datetime.now().isoformat(),
        "instructions": instructions,
        "images": [{"path": str(p), "status": "pending"} for p in image_paths],
        "categories": [
            "promotional_banner",
            "payment_banner",
            "offer_banner",
            "navigation_banner",
            "kyc_banner",
        ],
        "annotation_guidelines": {
            "bbox_fit": "Tight fit, no excessive whitespace",
            "consistency": "Same object type should have same label",
            "completeness": "Label all visible objects",
            "edge_handling": "Include object even if partially visible",
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(task_config, f, indent=2)
    
    logger.info(f"Created annotation task with {len(image_paths)} images: {output_file}")
    return task_config


if __name__ == "__main__":
    # Demo usage
    print("Annotation Utilities Module")
    print("=" * 60)
    print("Features:")
    print("  ✓ Export to COCO, YOLO, Label Studio, Pascal VOC formats")
    print("  ✓ Quality metrics (IoU, precision, recall, IAA)")
    print("  ✓ Active learning for efficient annotation")
    print("  ✓ Google Cloud Vertex AI compatible")
    print("=" * 60)
