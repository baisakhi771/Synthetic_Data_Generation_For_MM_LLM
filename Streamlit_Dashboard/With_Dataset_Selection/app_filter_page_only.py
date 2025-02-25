import streamlit as st
import pandas as pd
import json
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Streamlit Header and Introduction
st.set_page_config(page_title="Synthetic Data Generation for Multi-modal LLMs", layout="wide")

st.title("🧠 Synthetic Data Generation for Multi-modal LLMs")
st.markdown("""
    Welcome to the **Synthetic Data Generation Dashboard** for multi-modal Large Language Models (LLMs).  
    This dashboard allows users to **explore, analyze, and filter** human-bot conversations generated from **Gemini**, and evaluate their quality based on 3H (Honesty, Helpfulness and Harmlessness).  
""")
st.divider()  # 🔹 Adds a horizontal line for better UI separation

# Function to load and cache datasets
@st.cache_data
def load_conversation_data(json_file):
    """Loads conversation JSON data."""
    with open(json_file, "r") as file:
        return pd.json_normalize(json.load(file), sep="_")

@st.cache_data
def load_evaluation_data(json_file):
    """Loads evaluation JSON data (scores only, no explanations)."""
    with open(json_file, "r") as file:
        data = json.load(file)
    
    # Extract only numeric scores
    for entry in data:
        for key, value in entry["evaluation_scores"].items():
            entry["evaluation_scores"][key] = value["score"]  # Keep only scores
    
    return pd.json_normalize(data, sep="_")


# Convert filtered DataFrame to JSON
@st.cache_data
def convert_df_to_json(df):
    return df.to_json(orient="records", indent=4)

# Function to decode base64 image
def decode_base64_image(encoded_string):
    """Decodes a base64 image and returns an HTML image tag."""
    return f'<img src="data:image/png;base64,{encoded_string}" style="width:50px;height:50px;" />' 


# Load datasets separately
# conversation_data = load_conversation_data("../TestCode/human_bot_conversation.json")
# evaluation_data = load_evaluation_data("../TestCode/conversation_evaluation_results_gemini_anime2.json")

conversation_data = load_conversation_data("../TestCode/human_bot_conversation_part_0.json")
evaluation_data = load_evaluation_data("../TestCode/conversation_evaluation_results_gemini.json")

# Merge evaluation scores into conversation data for better filtering
merged_data = conversation_data.merge(evaluation_data, on="conversation_id", how="left")


# Sidebar Filters
st.sidebar.header("🔍 Filter Options")

# Filter by Number of Images
if "images" in merged_data.columns:
    image_counts = merged_data['images'].apply(len).unique()
    selected_image_count = st.sidebar.multiselect("Select Number of Images", image_counts, default=image_counts)
else:
    st.sidebar.warning("⚠️ 'images' column not found. Skipping filter.")
    selected_image_count = []

# Extract Score Columns
score_columns = [col for col in evaluation_data.columns if "_score" in col]

# Select Score Metric
selected_score = None
if score_columns:
    selected_score = st.sidebar.selectbox("Filter by Score Metric", score_columns)
    min_score, max_score = st.sidebar.slider("Select Score Range", 0, 10, (5, 10))
else:
    st.sidebar.error("⚠️ No evaluation score columns found!")

# Filter by keyword in conversation
search_text = st.sidebar.text_input("Search in Conversation")


# Apply Filters
filtered_conversations = merged_data.copy()

if "images" in merged_data.columns and selected_image_count:
    filtered_conversations = filtered_conversations[
        filtered_conversations['images'].apply(len).isin(selected_image_count)
    ]

if selected_score and selected_score in merged_data.columns:
    filtered_conversations = filtered_conversations[
        filtered_conversations[selected_score].between(min_score, max_score)
    ]
elif selected_score:
    st.warning(f"⚠️ Selected score column '{selected_score}' not found.")

if search_text and "conversation" in merged_data.columns:
    filtered_conversations = filtered_conversations[
        filtered_conversations["conversation"].str.contains(search_text, case=False, na=False)
    ]
elif search_text:
    st.warning("⚠️ 'conversation' column not found. Skipping search filter.")

# Visualization: Dynamic Average Scores by Metric
if not filtered_conversations.empty and score_columns:
    filtered_avg_scores = filtered_conversations[score_columns].mean().reset_index()
    filtered_avg_scores.columns = ["Metric", "Average Score"]
    
    st.subheader("📊 Average Scores by Metric (Filtered Data)")
    st.bar_chart(filtered_avg_scores.set_index("Metric"))
# Show Filtered Dataset with Image Thumbnails and Image-to-Tag Mapping
st.subheader("📊 Filtered Conversations")

if not filtered_conversations.empty:

    # Add Download Button for Filtered Dataset (JSON)
    json_data = convert_df_to_json(filtered_conversations)
    st.download_button(
        label="📥 Download Filtered Data (JSON)",
        data=json_data,
        file_name="filtered_dataset.json",
        mime="application/json",
    )

    for index, row in filtered_conversations.iterrows():
        st.markdown(f"### **Conversation ID: {row['conversation_id']}**")

        # Image-to-Tag Mapping
        st.markdown("**📷 Image-to-Tag Mapping:**")
        image_mappings = {}
        for idx, img_data in enumerate(row["images"]):
            img_name = img_data["name"]
            img_tag = f"<img_{idx+1}>"
            image_mappings[img_tag] = img_name
        
        st.json(image_mappings)  # Display mapping

        # Show Images as Thumbnails
        st.markdown("**🖼️ Images Used:**")
        image_html = ""
        for img in row["images"]:
            image_html += decode_base64_image(img["base64"]) + " "
        
        st.markdown(image_html, unsafe_allow_html=True)  # Render images inline
        
        # Show Conversation
        st.markdown(f"**💬 Conversation:** {row['conversation']}")

        # Show Scores
        st.markdown("**📊 Evaluation Scores:**")
        scores = {key: row[key] for key in score_columns if key in row}
        st.json(scores)

        st.divider()  # Add a separator between conversations

else:
    st.warning("⚠️ No data matches your filters.")


# # Visualization: Dynamic Average Scores by Metric
# if not filtered_conversations.empty and score_columns:
#     filtered_avg_scores = filtered_conversations[score_columns].mean().reset_index()
#     filtered_avg_scores.columns = ["Metric", "Average Score"]
    
#     st.subheader("📊 Average Scores by Metric (Filtered Data)")
#     st.bar_chart(filtered_avg_scores.set_index("Metric"))


