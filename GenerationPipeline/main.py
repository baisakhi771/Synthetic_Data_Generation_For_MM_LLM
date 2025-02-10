import argparse
import time
import random
import google.api_core
from llm_prompt import generate_prompt
from image_sampling import sample_from_folder, sample_from_cluster
from create_dataset import image_to_base64
from create_dataset import save_to_json_file

parser = argparse.ArgumentParser(
    description="CLI for image sampling and prompt generation"
)

parser.add_argument(
    "--num_datapoints", type=int, default=200, help="Number of datapoints to generate"
)
parser.add_argument(
    "--images_file_path", type=str, help="Path to the images folder or dataset"
)
parser.add_argument(
    "--save_to", type=str, help="Name of output JSON file", default="output.json"
)
parser.add_argument(
    "--sampling",
    type=str,
    choices=["sample_from_folder", "sample_from_cluster"],
    default="sample_from_folder",
    help="Sampling method to use",
)
parser.add_argument(
    "--num_clusters",
    type=int,
    nargs="?",
    default=None,
    help="Number of clusters (required if sampling from cluster)",
)

args = parser.parse_args()
sampled_images = []


def generate_with_retries(model_name, sampled_images, max_retries=7):
    for attempt in range(max_retries):
        try:
            return generate_prompt(model_name, sampled_images)  # Attempt API call
        except google.api_core.exceptions.ResourceExhausted:
            wait_time = (2**attempt) + random.uniform(0, 1)  # Exponential Backoff
            print(f"Quota exceeded. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("Max retries reached. Unable to complete request.")


for i in range(args.num_datapoints):
    print(f"Generating datapoint {i + 1} of {args.num_datapoints}")
    if args.sampling == "sample_from_folder":
        sampled_images = sample_from_folder(args.images_file_path)
    elif args.sampling == "sample_from_cluster":
        if args.num_clusters is None:
            raise ValueError(
                "num_clusters must be specified when using sample_from_cluster"
            )
        sampled_images = sample_from_cluster(args.images_file_path, args.num_clusters)

    output_text = generate_with_retries("gemini-1.5-flash", sampled_images)
    images_for_model = [img[0] for img in sampled_images]
    image_names = [img[1] for img in sampled_images]
    # Convert images to base64 for inline embedding
    base64_images = [image_to_base64(img) for img in images_for_model]
    # Generate the image tag-to-name mapping
    image_tag_mapping = {f"<img_{i+1}>": name for i, name in enumerate(image_names)}
    conversation_entry = {
        "images": [
            {"name": name, "base64": b64}
            for name, b64 in zip(image_names, base64_images)
        ],
        "conversation": output_text,
        "image_tag_mapping": image_tag_mapping,  # Add the image-to-tag mapping here
    }

    # Save to JSON file
    save_to_json_file(conversation_entry, json_file=args.save_to)
    print(f"Conversation {i} saved to human_bot_conversation.json.")
    time.sleep(4)
