import os
import random
from PIL import Image
import base64
from io import BytesIO
import json


# Function to convert image to base64
def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def save_to_json_file(data, json_file="output.json"):
    """
    Save the conversation data to a JSON file, appending new runs.
    :param data: The new conversation data to save.
    :param json_file: The path to the JSON file.
    """
    # Load existing data if the file exists
    if os.path.exists(json_file):
        with open(json_file, "r") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:  # Handle empty or corrupted files
                print(
                    f"Warning: {json_file} is empty or corrupted. Initializing new file."
                )
                existing_data = []
    else:
        existing_data = []
    last_id = existing_data[-1]["conversation_id"] if existing_data else 0
    data["conversation_id"] = last_id + 1

    # Append the new data
    existing_data.append(data)

    # Save back to the JSON file
    with open(json_file, "w") as file:
        json.dump(existing_data, file, indent=4)
