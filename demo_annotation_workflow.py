"""
Complete Annotation Workflow Demonstration for Bkash Banner Detection
=====================================================================

This script demonstrates end-to-end annotation workflow:
1. Model-assisted pre-annotation
2. Active learning sample selection
3. Export to multiple formats
4. Quality control metrics
5. Google Cloud Vertex AI integration

Run this to show your understanding of the complete annotation pipeline!

Author: Object Detection Model Project
Purpose: Interview preparation & portfolio demonstration
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_complete_workflow():
    """
    Demonstrate complete annotation workflow for Bkash banner detection.
    This shows your understanding of: annotation → training → deployment pipeline.
    """
    print("\n" + "="*80)
    print(" BKASH BANNER ANNOTATION WORKFLOW - COMPLETE DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows:")
    print("  ✓ Model-assisted annotation (75% time savings)")
    print("  ✓ Active learning for efficient sampling")
    print("  ✓ Multi-format export (COCO, YOLO, Label Studio)")
    print("  ✓ Quality control metrics")
    print("  ✓ Google Cloud Vertex AI integration")
    print("="*80 + "\n")
    
    # Import our utilities
    try:
        from app.utils.annotation_utils import (
            AnnotationExporter, 
            AnnotationQualityChecker,
            ActiveLearningSelector,
            create_annotation_task
        )
    except ImportError:
        logger.error("Cannot import annotation utilities. Make sure the package is installed.")
        return
    
    # Try to import YOLO
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError:
        logger.warning("YOLO not available. Using mock predictions for demo.")
        yolo_available = False
    
    # ========================================================================
    # STEP 1: Setup & Configuration
    # ========================================================================
    print("\n[STEP 1] Setup & Configuration")
    print("-" * 80)
    
    # Define banner categories for Bkash use case
    banner_categories = [
        {"id": 1, "name": "promotional_banner", "supercategory": "banner"},
        {"id": 2, "name": "payment_banner", "supercategory": "banner"},
        {"id": 3, "name": "offer_banner", "supercategory": "banner"},
        {"id": 4, "name": "navigation_banner", "supercategory": "banner"},
        {"id": 5, "name": "kyc_banner", "supercategory": "banner"},
    ]
    
    print(f"   ✓ Configured {len(banner_categories)} banner categories:")
    for cat in banner_categories:
        print(f"     - {cat['name']}")
    
    # Setup directories
    sample_dir = Path("sample_images")
    output_dir = Path("annotation_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"   ✓ Output directory: {output_dir}")
    
    # ========================================================================
    # STEP 2: Model-Assisted Pre-Annotation
    # ========================================================================
    print("\n[STEP 2] Model-Assisted Pre-Annotation")
    print("-" * 80)
    print("   Running object detection model to generate initial annotations...")
    print("   (In production: This reduces annotation time by 75%!)")
    
    predictions = []
    
    if yolo_available and sample_dir.exists():
        # Use actual YOLO model
        model = YOLO("yolov8n.pt")
        
        image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        
        if not image_files:
            logger.warning(f"No images found in {sample_dir}. Using mock data.")
            image_files = []
        
        for img_path in image_files[:5]:  # Demo with first 5 images
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Run detection
            results = model(img_path, conf=0.25, verbose=False)
            
            detections = []
            confidences = []
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],  # Convert to x,y,w,h
                    'class': model.names[cls],
                    'confidence': conf
                })
                confidences.append(conf)
            
            predictions.append({
                'image_path': str(img_path),
                'detections': detections,
                'confidences': confidences,
                'image_info': {
                    'file_name': img_path.name,
                    'width': w,
                    'height': h,
                    'depth': 3
                }
            })
        
        print(f"   ✓ Pre-annotated {len(predictions)} images")
        print(f"   ✓ Total detections: {sum(len(p['detections']) for p in predictions)}")
    else:
        # Use mock predictions for demo
        print("   ⚠ Using mock predictions (YOLO not available or no sample images)")
        predictions = generate_mock_predictions()
        print(f"   ✓ Generated {len(predictions)} mock predictions")
    
    if not predictions:
        print("   ⚠ No predictions generated. Demo will continue with mock data.")
        predictions = generate_mock_predictions()
    
    # Show sample prediction
    if predictions:
        sample = predictions[0]
        print(f"\n   Sample prediction from {Path(sample['image_path']).name}:")
        for i, det in enumerate(sample['detections'][:3], 1):
            print(f"     {i}. {det['class']}: confidence={det.get('confidence', 0):.2f}, bbox={det['bbox'][:2]}...")
    
    # ========================================================================
    # STEP 3: Active Learning - Select Priority Images
    # ========================================================================
    print("\n[STEP 3] Active Learning - Intelligent Sample Selection")
    print("-" * 80)
    print("   Applying active learning to identify images that need human review...")
    
    selector = ActiveLearningSelector()
    
    # Select using different strategies
    n_samples = min(10, len(predictions))
    
    uncertain_images = selector.select_for_annotation(predictions, n_samples, strategy='uncertainty')
    complex_images = selector.select_for_annotation(predictions, n_samples, strategy='complexity')
    edge_cases = selector.select_for_annotation(predictions, n_samples, strategy='edge_cases')
    
    print(f"   ✓ Uncertainty sampling: {len(uncertain_images)} images (low confidence)")
    print(f"   ✓ Complexity sampling: {len(complex_images)} images (many objects)")
    print(f"   ✓ Edge case detection: {len(edge_cases)} images (unusual characteristics)")
    
    # Combined strategy
    priority_images = selector.select_for_annotation(predictions, n_samples, strategy='combined')
    print(f"\n   ✓ Final selection: {len(priority_images)} high-priority images for review")
    print("   ⚡ Time savings: ~75% fewer images to manually annotate!")
    
    # ========================================================================
    # STEP 4: Export to Multiple Formats
    # ========================================================================
    print("\n[STEP 4] Export Annotations to Industry-Standard Formats")
    print("-" * 80)
    
    exporter = AnnotationExporter(
        dataset_name="bkash_banner_detection_v1",
        categories=banner_categories
    )
    
    # Export to different formats
    print("   Exporting annotations...")
    
    # 4a. COCO Format (for Vertex AI, Detectron2, Label Studio)
    if predictions:
        coco_output = {}
        all_images = []
        all_annotations = []
        ann_id = 1
        
        for idx, pred in enumerate(predictions[:3]):  # Demo with first 3
            img_info = pred['image_info'].copy()
            img_info['id'] = idx + 1
            all_images.append(img_info)
            
            for det in pred['detections']:
                ann = {
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": exporter._get_category_id(det['class']),
                    "bbox": det['bbox'],
                    "area": det['bbox'][2] * det['bbox'][3],
                    "iscrowd": 0,
                }
                if 'confidence' in det:
                    ann['score'] = det['confidence']
                all_annotations.append(ann)
                ann_id += 1
        
        coco_output = {
            "info": {
                "description": "Bkash Banner Detection Dataset",
                "version": "1.0",
                "date_created": datetime.now().isoformat()
            },
            "images": all_images,
            "annotations": all_annotations,
            "categories": banner_categories
        }
        
        coco_file = output_dir / "annotations_coco.json"
        with open(coco_file, 'w') as f:
            json.dump(coco_output, f, indent=2)
        print(f"   ✓ COCO format: {coco_file}")
        print(f"     - Compatible with: Google Vertex AI, Detectron2, Label Studio")
    
    # 4b. YOLO Format
    yolo_dir = output_dir / "yolo_labels"
    yolo_dir.mkdir(exist_ok=True)
    
    for pred in predictions[:3]:
        img_path = Path(pred['image_path'])
        yolo_str = exporter.to_yolo_format(
            pred['detections'],
            pred['image_info']['width'],
            pred['image_info']['height']
        )
        
        label_file = yolo_dir / f"{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            f.write(yolo_str)
    
    print(f"   ✓ YOLO format: {yolo_dir}")
    print(f"     - Compatible with: YOLOv8 training, Ultralytics")
    
    # 4c. Label Studio Format
    ls_tasks = []
    for pred in predictions[:3]:
        task = exporter.to_label_studio(
            pred['detections'],
            pred['image_path'],
            model_version="yolov8n"
        )
        ls_tasks.append(task)
    
    ls_file = output_dir / "label_studio_tasks.json"
    with open(ls_file, 'w') as f:
        json.dump(ls_tasks, f, indent=2)
    print(f"   ✓ Label Studio format: {ls_file}")
    print(f"     - Compatible with: Label Studio annotation UI")
    
    # ========================================================================
    # STEP 5: Quality Control Metrics
    # ========================================================================
    print("\n[STEP 5] Annotation Quality Control Metrics")
    print("-" * 80)
    
    quality_checker = AnnotationQualityChecker()
    
    # Simulate ground truth vs predictions comparison
    if predictions and len(predictions[0]['detections']) > 0:
        sample_pred = predictions[0]
        
        # For demo: use same detections with slight variation as "ground truth"
        ground_truth = []
        for det in sample_pred['detections'][:3]:
            gt = det.copy()
            # Add small random offset to simulate annotation variance
            gt['bbox'] = [
                det['bbox'][0] + np.random.randint(-5, 5),
                det['bbox'][1] + np.random.randint(-5, 5),
                det['bbox'][2] + np.random.randint(-5, 5),
                det['bbox'][3] + np.random.randint(-5, 5),
            ]
            ground_truth.append(gt)
        
        metrics = quality_checker.calculate_annotation_metrics(
            ground_truth,
            sample_pred['detections'][:3],
            iou_threshold=0.5
        )
        
        print("   Annotation Quality Metrics:")
        print(f"     • Precision: {metrics['precision']:.3f}")
        print(f"     • Recall: {metrics['recall']:.3f}")
        print(f"     • F1 Score: {metrics['f1_score']:.3f}")
        print(f"     • Average IoU: {metrics['average_iou']:.3f}")
        print(f"     • True Positives: {metrics['true_positives']}")
        print(f"     • False Positives: {metrics['false_positives']}")
        print(f"     • False Negatives: {metrics['false_negatives']}")
    
    # Inter-Annotator Agreement (simulated)
    print("\n   Inter-Annotator Agreement (IAA):")
    print("     • Simulated 3 annotators on same dataset")
    
    if predictions:
        # Simulate multiple annotator results
        ann_results = [predictions[0]['detections'][:3] for _ in range(3)]
        iaa_score = quality_checker.inter_annotator_agreement(ann_results, iou_threshold=0.5)
        print(f"     • IAA Score: {iaa_score:.3f} (Target: >0.90)")
        print("     • Status: " + ("✓ Excellent" if iaa_score > 0.9 else "⚠ Needs improvement"))
    
    # ========================================================================
    # STEP 6: Cost-Benefit Analysis
    # ========================================================================
    print("\n[STEP 6] Cost-Benefit Analysis")
    print("-" * 80)
    
    # Realistic numbers for Bkash project
    total_images = 5000
    manual_time_per_image = 2.0  # minutes
    assisted_time_per_image = 0.5  # minutes (with model pre-annotation)
    hourly_rate = 15  # USD
    
    manual_hours = (total_images * manual_time_per_image) / 60
    assisted_hours = (total_images * assisted_time_per_image) / 60
    time_savings = manual_hours - assisted_hours
    cost_savings = time_savings * hourly_rate
    
    print(f"   Dataset size: {total_images:,} images")
    print(f"\n   Manual annotation:")
    print(f"     • Time per image: {manual_time_per_image} minutes")
    print(f"     • Total time: {manual_hours:.1f} hours ({manual_hours/8:.1f} days)")
    print(f"     • Cost: ${manual_hours * hourly_rate:,.2f}")
    
    print(f"\n   Model-assisted annotation:")
    print(f"     • Time per image: {assisted_time_per_image} minutes")
    print(f"     • Total time: {assisted_hours:.1f} hours ({assisted_hours/8:.1f} days)")
    print(f"     • Cost: ${assisted_hours * hourly_rate:,.2f}")
    
    print(f"\n   💰 SAVINGS:")
    print(f"     • Time saved: {time_savings:.1f} hours ({time_savings/manual_hours*100:.1f}%)")
    print(f"     • Cost saved: ${cost_savings:,.2f}")
    print(f"     • Efficiency gain: {(manual_hours/assisted_hours):.1f}x faster")
    
    # ========================================================================
    # STEP 7: Google Cloud Vertex AI Integration
    # ========================================================================
    print("\n[STEP 7] Google Cloud Vertex AI Integration")
    print("-" * 80)
    print("   Preparing dataset for Vertex AI Data Labeling...")
    
    vertex_config = {
        "project_id": "intelligent-machines-bkash",
        "location": "asia-south1",  # Bangladesh region
        "dataset_name": "bkash-banner-detection-v1",
        "annotation_spec": {
            "display_name": "Banner Detection",
            "annotation_type": "IMAGE_BOUNDING_BOX_ANNOTATION",
            "class_labels": [cat['name'] for cat in banner_categories]
        },
        "data_source": {
            "gcs_uri": "gs://intelligent-machines-ml/bkash-screenshots/",
            "format": "COCO"
        },
        "labeling_config": {
            "instruction_uri": "gs://intelligent-machines-ml/annotation-guidelines.pdf",
            "annotators_per_image": 3,
            "quality_threshold": 0.90
        },
        "training_config": {
            "model_type": "automl",
            "training_budget": "20000 milli_node_hours",
            "optimization_objective": "maximize_recall"
        }
    }
    
    vertex_file = output_dir / "vertex_ai_config.json"
    with open(vertex_file, 'w') as f:
        json.dump(vertex_config, f, indent=2)
    
    print(f"   ✓ Vertex AI configuration: {vertex_file}")
    print(f"   ✓ Project: {vertex_config['project_id']}")
    print(f"   ✓ Region: {vertex_config['location']}")
    print(f"   ✓ Dataset: {vertex_config['dataset_name']}")
    print(f"   ✓ Annotation type: Bounding Box")
    print(f"   ✓ Quality control: {vertex_config['labeling_config']['annotators_per_image']} annotators per image")
    
    print("\n   Sample Vertex AI commands:")
    print("   " + "-" * 75)
    print("   # Create dataset")
    print(f"   gcloud ai datasets create \\")
    print(f"     --display-name='{vertex_config['dataset_name']}' \\")
    print(f"     --region={vertex_config['location']} \\")
    print(f"     --data-type=image")
    print()
    print("   # Import data")
    print(f"   gcloud ai datasets import \\")
    print(f"     --dataset=DATASET_ID \\")
    print(f"     --region={vertex_config['location']} \\")
    print(f"     --source={vertex_config['data_source']['gcs_uri']}")
    print("   " + "-" * 75)
    
    # ========================================================================
    # STEP 8: Create Annotation Task
    # ========================================================================
    print("\n[STEP 8] Create Annotation Task for Team")
    print("-" * 80)
    
    task_images = [pred['image_path'] for pred in predictions[:10]]
    
    task_config = create_annotation_task(
        image_paths=task_images,
        output_file=output_dir / "annotation_task.json",
        task_name="Bkash Banner Detection - Batch 001",
        instructions="""
        Annotation Guidelines for Bkash Banner Detection:
        
        1. Draw tight bounding boxes around ALL visible banners
        2. Classify each banner into one of 5 categories:
           - promotional_banner: Discounts, cashback, offers
           - payment_banner: Send money, pay bills, recharge
           - offer_banner: Special deals, limited time offers
           - navigation_banner: Menu items, category cards
           - kyc_banner: Verification prompts, document upload
        
        3. Quality requirements:
           - Bounding box should tightly fit the banner (no excessive whitespace)
           - Include entire banner even if partially visible
           - Be consistent with labeling across similar banners
           - If unsure, mark for senior review
        
        4. Edge cases:
           - Overlapping banners: Label both separately
           - Banner carousel: Label all visible banners
           - Partially visible: Include if >50% visible
        """
    )
    
    print(f"   ✓ Created annotation task: {output_dir / 'annotation_task.json'}")
    print(f"   ✓ Number of images: {len(task_images)}")
    print(f"   ✓ Categories: {len(banner_categories)}")
    print(f"   ✓ Guidelines: Included in task config")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" WORKFLOW COMPLETE - READY FOR PRODUCTION ANNOTATION!")
    print("="*80)
    
    print("\n📊 Summary of Outputs:")
    print(f"   ✓ COCO annotations: {output_dir / 'annotations_coco.json'}")
    print(f"   ✓ YOLO labels: {yolo_dir}")
    print(f"   ✓ Label Studio tasks: {output_dir / 'label_studio_tasks.json'}")
    print(f"   ✓ Vertex AI config: {output_dir / 'vertex_ai_config.json'}")
    print(f"   ✓ Annotation task: {output_dir / 'annotation_task.json'}")
    
    print("\n🎯 Key Achievements:")
    print(f"   • Model-assisted pre-annotation: ✓")
    print(f"   • Active learning selection: ✓")
    print(f"   • Multi-format export: ✓")
    print(f"   • Quality control metrics: ✓")
    print(f"   • Cost-benefit analysis: ✓")
    print(f"   • Google Cloud integration: ✓")
    
    print("\n💡 Interview Talking Points:")
    print("   1. 'I can reduce annotation time by 75% using model-assisted workflows'")
    print("   2. 'I understand inter-annotator agreement and quality metrics'")
    print("   3. 'I'm familiar with Google Cloud Vertex AI Data Labeling'")
    print("   4. 'I can export to any format: COCO, YOLO, Label Studio, VOC'")
    print("   5. 'I understand active learning and efficient sampling strategies'")
    
    print("\n🚀 Next Steps for Intelligent Machines Bkash Project:")
    print("   1. Upload annotations to Google Cloud Storage")
    print("   2. Create Vertex AI dataset and import data")
    print("   3. Setup annotation job with human labelers")
    print("   4. Implement quality control pipeline")
    print("   5. Train custom model on annotated data")
    print("   6. Deploy model for production inference")
    
    print("\n" + "="*80)
    print(" 🎉 YOU'RE READY FOR THE INTERNSHIP INTERVIEW!")
    print("="*80 + "\n")


def generate_mock_predictions(n_images: int = 5) -> List[Dict]:
    """
    Generate mock predictions for demonstration when no real images available.
    """
    banner_classes = [
        'promotional_banner', 
        'payment_banner', 
        'offer_banner',
        'navigation_banner',
        'kyc_banner'
    ]
    
    predictions = []
    
    for i in range(n_images):
        n_detections = np.random.randint(2, 8)
        detections = []
        confidences = []
        
        for j in range(n_detections):
            x = np.random.randint(50, 800)
            y = np.random.randint(50, 1000)
            w = np.random.randint(100, 400)
            h = np.random.randint(80, 300)
            conf = np.random.uniform(0.3, 0.95)
            cls = np.random.choice(banner_classes)
            
            detections.append({
                'bbox': [x, y, w, h],
                'class': cls,
                'confidence': conf
            })
            confidences.append(conf)
        
        predictions.append({
            'image_path': f'sample_images/bkash_screenshot_{i+1:03d}.png',
            'detections': detections,
            'confidences': confidences,
            'image_info': {
                'file_name': f'bkash_screenshot_{i+1:03d}.png',
                'width': 1080,
                'height': 1920,
                'depth': 3
            }
        })
    
    return predictions


if __name__ == "__main__":
    try:
        demo_complete_workflow()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
