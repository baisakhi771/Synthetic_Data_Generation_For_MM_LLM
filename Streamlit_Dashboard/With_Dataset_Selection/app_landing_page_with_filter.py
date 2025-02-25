import streamlit as st
import pandas as pd
import json
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px

# ----------------- SETUP MULTI-PAGE NAVIGATION -----------------
st.set_page_config(page_title="Synthetic Data Generation for Multi-modal LLMs", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Dataset Explorer"])

# ----------------- HOME PAGE -----------------
if page == "🏠 Home":
    st.title("🧠 Synthetic Data Generation for Multi-modal LLMs")
    st.markdown("""
    ## Welcome to the Synthetic Data Generation Dashboard!  
    This project focuses on generating high-quality **multi-modal datasets** using **Gemini AI** and evaluating chatbot responses with **Gemini AI** based on 3H (Honesty, Helpfulness and Harmlessness) parameters.  

    ### 📌 **Project Objectives**
    - Generate synthetic **human-bot conversations** based on **text and images**.
    - Ensure **ethical AI** by preventing biased, toxic, or identifiable personal information.
    - **Evaluate** chatbot responses using **multiple LLM models** to assess quality.

    ### 🔍 **Methodology**
    1. **Synthetic Data Generation**:  
       - Uses **Gemini AI** to generate human-bot conversations.
       - Includes **multi-turn dialogues** with references to images.
    2. **Dataset Evaluation**:  
       - Uses **Gemini** to provide **7 evaluation scores** per conversation:
         - **Relevance, Coherence, Factual Accuracy, Bias, Fluency, Image Alignment, Creativity**.
    3. **Dataset Explorer & Visualization**:  
       - Interactive filtering and visualization of scores.
       - Image thumbnail previews for conversations.

    ### 🚀 **Key Features**
    - 📊 **Dataset Filtering & Score Visualization**
    - 🖼️ **Image Previews & Mapping**
    - 📥 **Download Filtered Dataset**
    """)

    st.info("🔄 Use the sidebar to navigate to the **Dataset Explorer**!")

# ----------------- DATASET EXPLORER PAGE -----------------
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")

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

    # Load datasets
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

    # # Visualization: Dynamic Average Scores by Metric
    # if not filtered_conversations.empty and score_columns:
    #     filtered_avg_scores = filtered_conversations[score_columns].mean().reset_index()
    #     filtered_avg_scores.columns = ["Metric", "Average Score"]
        
    #     st.subheader("📊 Average Scores by Metric (Filtered Data)")
    #     st.bar_chart(filtered_avg_scores.set_index("Metric"))

    # ✅ Define evaluation score columns
    score_columns = [
        "evaluation_scores_Relevance",
        "evaluation_scores_Coherence",
        "evaluation_scores_Factual Accuracy",
        "evaluation_scores_Bias & Toxicity",
        "evaluation_scores_Fluency",
        "evaluation_scores_Image Alignment",
        "evaluation_scores_Creativity"
    ]

    # ✅ Ensure filtered_data is not empty before calculations
    if not filtered_conversations.empty:
        # ✅ Compute average scores
        avg_scores = filtered_conversations[score_columns].mean().reset_index()
        avg_scores.columns = ["Metric", "Average Score"]  # Rename columns

        # ✅ Rename metrics for better readability
        clean_labels = {
            "evaluation_scores_Relevance": "Relevance",
            "evaluation_scores_Coherence": "Coherence",
            "evaluation_scores_Factual Accuracy": "Factual Accuracy",
            "evaluation_scores_Bias & Toxicity": "Bias & Toxicity",
            "evaluation_scores_Fluency": "Fluency",
            "evaluation_scores_Image Alignment": "Image Alignment",
            "evaluation_scores_Creativity": "Creativity"
        }
        avg_scores["Metric"] = avg_scores["Metric"].replace(clean_labels)

        # ✅ Re-plot bar chart with updated labels
        st.subheader("📊 Average Scores by Metric (Filtered Data)")
        fig = px.bar(avg_scores, x="Metric", y="Average Score", color="Metric", text="Average Score")
        fig.update_layout(xaxis_title="Evaluation Metric", yaxis_title="Average Score")
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ No data available after filtering. Adjust filters to see results.")


    # Show Filtered Dataset with Image Thumbnails and Image-to-Tag Mapping
    st.subheader("📊 Filtered Conversations")




    if not filtered_conversations.empty:
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

