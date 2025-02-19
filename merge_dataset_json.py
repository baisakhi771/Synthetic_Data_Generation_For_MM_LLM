import json
import argparse
import os

def merge_json_files(input_files, output_file):
    
    merged_data = []
    current_id = 1  # Start conversation IDs from 1

    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for entry in data:
            entry["conversation_id"] = current_id  # Assign a new continuous ID
            merged_data.append(entry)
            current_id += 1  # Increment ID for the next entry
    
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged {len(input_files)} files into '{output_file}' with {len(merged_data)} continuous conversation IDs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSON dataset files into one with continuous IDs.")
    parser.add_argument("--input_files", nargs="+", required=True, help="List of JSON files to merge.")
    parser.add_argument("--output_file", required=True, help="Output JSON file path.")

    args = parser.parse_args()
    merge_json_files(args.input_files, args.output_file)
