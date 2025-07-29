import requests
import json
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
IMAGE_FILE = "cat.jpg"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return False

def test_image_upload(image_path):
    """Test the image upload endpoint with cat.jpg"""
    print(f"\nTesting image upload with {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return False
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'img': (os.path.basename(image_path), image_file, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/detect_img", files=files)
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if result.get('status_code') == 200:
            print("Detection successful!")
            data = result.get('data', {})
            print(f"Image Name: {data.get('image_name')}")
            print(f"Image Size: {data.get('image_size')}")
            
            inference_results = data.get('inference_results', [])
            if inference_results:
                print(f"Number of detections: {len(inference_results)}")
                for i, detection in enumerate(inference_results):
                    conf = detection.get('confidence', 0)
                    class_name = detection.get('name', 'Unknown')
                    bbox = detection.get('box', {})
                    print(f"  Detection {i+1}: {class_name} (confidence: {conf:.2f})")
                    if bbox:
                        print(f"    Bounding box: x1={bbox.get('x1', 0):.1f}, y1={bbox.get('y1', 0):.1f}, "
                              f"x2={bbox.get('x2', 0):.1f}, y2={bbox.get('y2', 0):.1f}")
            else:
                print("No objects detected in the image.")
        else:
            print(f"Detection failed: {result}")
        
        return result.get('status_code') == 200
        
    except requests.exceptions.RequestException as e:
        print(f"Error uploading image: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_detect_by_path():
    """Test detection by providing a path to the current directory"""
    print(f"\nTesting detection by path...")
    current_dir = os.getcwd().replace('\\', '$')  # Replace backslashes with $ as expected by the API
    
    try:
        response = requests.get(f"{BASE_URL}/detect", params={'path': current_dir})
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if result.get('status_code') == 200:
            data = result.get('data', {})
            inference_results = data.get('inference_results', [])
            print(f"Found {len(inference_results)} images processed")
            
            for i, img_result in enumerate(inference_results):
                path = img_result.get('path', 'Unknown')
                detections = img_result.get('detections', [])
                print(f"  Image {i+1}: {os.path.basename(path)} - {len(detections)} detections")
        else:
            print(f"Path detection failed: {result}")
        
        return result.get('status_code') == 200
        
    except requests.exceptions.RequestException as e:
        print(f"Error testing path detection: {e}")
        return False

def test_get_images():
    """Test getting available images in the current directory"""
    print(f"\nTesting get available images...")
    current_dir = os.getcwd().replace('\\', '$')  # Replace backslashes with $ as expected by the API
    
    try:
        response = requests.get(f"{BASE_URL}/images", params={'path': current_dir})
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if result.get('status_code') == 200:
            data = result.get('data', {})
            available_images = data.get('available_images', [])
            size = data.get('size', 0)
            print(f"Found {size} images: {available_images}")
        else:
            print(f"Get images failed: {result}")
        
        return result.get('status_code') == 200
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting images: {e}")
        return False

def main():
    """Main function to run all tests"""
    print("=== FastAPI YOLO Model Testing Client ===")
    print(f"Server URL: {BASE_URL}")
    print(f"Test Image: {IMAGE_FILE}")
    print("=" * 50)
    
    # Check if the image file exists
    if not os.path.exists(IMAGE_FILE):
        print(f"Warning: {IMAGE_FILE} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("Available files:", [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Get Available Images", test_get_images),
        ("Image Upload Detection", lambda: test_image_upload(IMAGE_FILE)),
        ("Path Detection", test_detect_by_path),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print("="*50)
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

if __name__ == "__main__":
    main()