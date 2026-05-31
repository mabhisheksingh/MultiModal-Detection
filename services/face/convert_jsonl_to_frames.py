import json

def convert_jsonl_to_frames_json(input_file, output_file):
    """Convert JSONL file with Ceph URLs to frames JSON format expected by video_client.py"""
    
    frames_data = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse each JSON line
                data = json.loads(line.strip())
                
                # Extract frame number from file_name (e.g., "frame_000001.jpg" -> 1)
                frame_name = data.get('file_name', '')
                if 'frame_' in frame_name:
                    frame_number_str = frame_name.split('frame_')[1].split('.')[0]
                    frame_id = int(frame_number_str.lstrip('0')) if frame_number_str != '000000' else 0
                else:
                    frame_id = line_num
                
                # Create frame entry in expected format
                frame_entry = {
                    "frame_id": frame_id,
                    "frame_path": data.get('url')  # Map 'url' to 'frame_path'
                }
                
                frames_data.append(frame_entry)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Save as JSON array
    with open(output_file, 'w') as f:
        json.dump(frames_data, f, indent=2)
    
    print(f"Converted {len(frames_data)} frames")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    return len(frames_data)

if __name__ == "__main__":
    input_file = "input/uploaded_ceph_links_off_duty_10fps.jsonl"
    output_file = "input/uploaded_ceph_links_off_duty_10fps.json"
    
    convert_jsonl_to_frames_json(input_file, output_file)
