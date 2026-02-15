"""
Batch Testing Script for Object Detection Models
Tests YOLO, Grounding DINO, SAM, and Pipeline with multiple images.
"""

import requests
import json
import base64
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class ModelTester:
    """Automated testing for object detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        self.output_dir = Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
    def test_health(self) -> Dict:
        """Test health endpoint."""
        print("\n" + "="*60)
        print("🏥 HEALTH CHECK")
        print("="*60)
        
        response = requests.get(f"{self.base_url}/health")
        data = response.json()
        
        print(f"Status: {data['status']}")
        print("Models:")
        for model, status in data['models'].items():
            print(f"  • {model}: {status}")
        
        self.results['health'] = data
        return data
    
    def test_yolo(self, images: List[Path]) -> Dict:
        """Test YOLOv8 detection on multiple images."""
        print("\n" + "="*60)
        print("🎯 TESTING YOLO v8")
        print("="*60)
        
        yolo_dir = self.run_dir / "yolo"
        yolo_dir.mkdir(exist_ok=True)
        
        results = []
        
        for img_path in images:
            print(f"\n📷 Processing: {img_path.name}")
            
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                data = {'confidence': 0.25}
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/yolo/detect",
                    files=files,
                    data=data
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    detections = result.get('detections', [])
                    
                    # Print detections
                    print(f"  ⏱️  Time: {elapsed:.2f}s")
                    print(f"  🔍 Detections: {len(detections)}")
                    for det in detections[:3]:  # Show first 3
                        print(f"     • {det['class_name']} ({det['confidence']:.2%})")
                    if len(detections) > 3:
                        print(f"     ... and {len(detections) - 3} more")
                    
                    results.append({
                        'image': img_path.name,
                        'success': True,
                        'time': elapsed,
                        'detections': len(detections)
                    })
                else:
                    print(f"  ❌ Error: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    results.append({
                        'image': img_path.name,
                        'success': False,
                        'error': response.text
                    })
        
        self.results['yolo'] = results
        return results
    
    def test_grounding_dino(self, images: List[Path]) -> Dict:
        """Test Grounding DINO with text prompts."""
        print("\n" + "="*60)
        print("🔤 TESTING GROUNDING DINO (Text-Based Detection)")
        print("="*60)
        
        dino_dir = self.run_dir / "grounding_dino"
        dino_dir.mkdir(exist_ok=True)
        
        # Different prompts for different images
        test_cases = [
            {'image': 'bus.jpg', 'prompt': 'bus. person. car.'},
            {'image': 'zidane.jpg', 'prompt': 'person. face. shirt.'},
            {'image': 'people.jpg', 'prompt': 'person. crowd. people.'},
            {'image': 'car.jpg', 'prompt': 'car. vehicle. automobile.'},
            {'image': 'dog.jpg', 'prompt': 'dog. animal. pet.'},
        ]
        
        results = []
        
        for test in test_cases:
            img_name = test['image']
            img_path = next((p for p in images if p.name == img_name), None)
            
            if not img_path or not img_path.exists():
                print(f"\n⏭️  Skipping {img_name} (not found)")
                continue
            
            print(f"\n📷 {img_path.name}")
            print(f"   Prompt: \"{test['prompt']}\"")
            
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                data = {
                    'text_prompt': test['prompt'],
                    'box_threshold': 0.35,
                    'text_threshold': 0.25
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/grounding-dino/detect",
                    files=files,
                    data=data
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    detections = result.get('detections', [])
                    
                    print(f"  ⏱️  Time: {elapsed:.2f}s")
                    print(f"  🔍 Detections: {len(detections)}")
                    for det in detections[:3]:
                        print(f"     • {det.get('label', 'unknown')} ({det['confidence']:.2%})")
                    
                    results.append({
                        'image': img_path.name,
                        'prompt': test['prompt'],
                        'success': True,
                        'time': elapsed,
                        'detections': len(detections)
                    })
                else:
                    print(f"  ❌ Error: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    results.append({
                        'image': img_path.name,
                        'prompt': test['prompt'],
                        'success': False,
                        'error': response.text
                    })
        
        self.results['grounding_dino'] = results
        return results
    
    def test_sam_auto(self, images: List[Path]) -> Dict:
        """Test SAM auto-segmentation."""
        print("\n" + "="*60)
        print("✂️  TESTING SAM (Segment Anything - Auto Mode)")
        print("="*60)
        
        sam_dir = self.run_dir / "sam"
        sam_dir.mkdir(exist_ok=True)
        
        results = []
        
        # Test on first 3 images (SAM is slower)
        for img_path in images[:3]:
            print(f"\n📷 Processing: {img_path.name}")
            
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/sam/segment-auto",
                    files=files
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    masks = result.get('masks', []) 
                    
                    print(f"  ⏱️  Time: {elapsed:.2f}s")
                    print(f"  🎭 Masks: {len(masks)}")
                    
                    results.append({
                        'image': img_path.name,
                        'success': True,
                        'time': elapsed,
                        'masks': len(masks)
                    })
                else:
                    print(f"  ❌ Error: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    results.append({
                        'image': img_path.name,
                        'success': False,
                        'error': response.text
                    })
        
        self.results['sam'] = results
        return results
    
    def test_pipeline(self, images: List[Path]) -> Dict:
        """Test combined DINO + SAM pipeline."""
        print("\n" + "="*60)
        print("🔥 TESTING PIPELINE (Grounding DINO + SAM)")
        print("="*60)
        
        pipeline_dir = self.run_dir / "pipeline"
        pipeline_dir.mkdir(exist_ok=True)
        
        test_cases = [
            {'image': 'bus.jpg', 'prompt': 'bus'},
            {'image': 'dog.jpg', 'prompt': 'dog'},
        ]
        
        results = []
        
        for test in test_cases:
            img_name = test['image']
            img_path = next((p for p in images if p.name == img_name), None)
            
            if not img_path or not img_path.exists():
                print(f"\n⏭️  Skipping {img_name} (not found)")
                continue
            
            print(f"\n📷 {img_path.name}")
            print(f"   Prompt: \"{test['prompt']}\"")
            
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                data = {
                    'text_prompt': test['prompt'],
                    'box_threshold': 0.35
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/pipeline/detect-and-segment",
                    files=files,
                    data=data,
                    timeout=600  # 10 minutes for slow models
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    segments = result.get('segments', [])
                    
                    print(f"  ⏱️  Time: {elapsed:.2f}s")
                    print(f"  🎯 Objects found & segmented: {len(segments)}")
                    
                    results.append({
                        'image': img_path.name,
                        'prompt': test['prompt'],
                        'success': True,
                        'time': elapsed,
                        'segmentations': len(segments)
                    })
                else:
                    print(f"  ❌ Error: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    results.append({
                        'image': img_path.name,
                        'prompt': test['prompt'],
                        'success': False,
                        'error': response.text
                    })
        
        self.results['pipeline'] = results
        return results
    
    def generate_report(self):
        """Generate summary report."""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY REPORT")
        print("="*60)
        
        report_path = self.run_dir / "summary.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📁 Results saved to: {self.run_dir}")
        print(f"📄 JSON report: {report_path.name}")
        
        # Print summary statistics
        for model, results in self.results.items():
            if model == 'health':
                continue
            
            if isinstance(results, list):
                success_count = sum(1 for r in results if r.get('success', False))
                total = len(results)
                avg_time = sum(r['time'] for r in results if 'time' in r) / max(success_count, 1)
                
                print(f"\n{model.upper()}:")
                print(f"  • Success: {success_count}/{total}")
                print(f"  • Avg time: {avg_time:.2f}s")
        
        print(f"\n✅ All done! Check {self.run_dir.name} for annotated images.")


def main():
    """Run batch tests on all models."""
    print("🚀 Starting Batch Model Testing")
    print("Make sure the server is running at http://localhost:8000\n")
    
    # Get test images
    image_dir = Path("sample_images")
    images = sorted(image_dir.glob("*.jpg"))
    
    if not images:
        print("❌ No test images found in sample_images/")
        return
    
    print(f"📷 Found {len(images)} test images:")
    for img in images:
        print(f"   • {img.name}")
    
    # Initialize tester
    tester = ModelTester()
    
    try:
        # Run tests
        tester.test_health()
        tester.test_yolo(images)
        tester.test_grounding_dino(images)
        tester.test_sam_auto(images)
        tester.test_pipeline(images)
        
        # Generate report
        tester.generate_report()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server at http://localhost:8000")
        print("Make sure the server is running with:")
        print("  uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
