import cv2
import json
import numpy as np
from pathlib import Path

def visualize_detections(image_path, data_path, output_path, confidence_threshold=0.2):
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    img_h, img_w = image.shape[:2]
    
    # Load detection data
    with open(data_path, 'r') as f:
        try:
            # The file contains a JSON-like 2D array
            raw_data = json.load(f)
        except json.JSONDecodeError:
            # Fallback if it's not strictly valid JSON
            content = f.read().replace("'", '"')
            raw_data = json.loads(content)
            
    detections = np.array(raw_data)
    print(f"Loaded {len(detections)} raw detections.")
    
    # Filter by confidence (5th column)
    valid_detections = detections[detections[:, 4] >= confidence_threshold]
    print(f"Found {len(valid_detections)} detections above threshold {confidence_threshold}.")
    
    for i, det in enumerate(valid_detections):
        cx, cy, w, h, conf = det
        
        # Convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label = f"Plate: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"Detection {i+1}: Conf={conf:.4f}, Box=[{x1}, {y1}, {x2}, {y2}]")

    # Save output image
    cv2.imwrite(str(output_path), image)
    print(f"Saved marked image to {output_path}")

if __name__ == "__main__":
    # Define paths
    base_path = Path("/Users/abhishek/PycharmProjects/MultiModal-Detection/PLATE_AND_OBJECT_DETECTION/triton_server/app")
    img_in = base_path / "frame_0000.jpg"
    data_in = base_path / "xyxrc.txt"
    img_out = base_path / "frame_0000_marked.jpg"
    
    visualize_detections(img_in, data_in, img_out, confidence_threshold=0.2)
