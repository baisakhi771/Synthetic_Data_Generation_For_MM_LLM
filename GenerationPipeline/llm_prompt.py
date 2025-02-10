import os
import google.generativeai as genai
from dotenv import load_dotenv


def generate_prompt(model_name, sampled_images):
    """ """

    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=API_KEY)

    system_instruction = """ You are an expert at generating dialogues, following all the given constraints.
    A dialogue contains interleaved text and images. The images are attached to the conversation.
    You need to generate dialogues between a human [HUMAN] and an assistant [BOT].

    Characteristics about the assistant [BOT]:
    1. The assistant is trained to understand text, images, and their combinations.
    2. The assistant can reply to the human with images and/or text.
    3. The assistant has exceptional world knowledge and common sense reasoning capabilities.
    4. The assistant must always use generic terms for famous humans in images
    5. If the human asks to identify a person in the image, the assistant should reply "I cannot identify specific people in the image"
    6. If there is potential for bias, racism or toxicity in the human's request, the assistant should reply "Sorry, I cannot help you with that"
    7. The bot is allowed to identify cartoons and humanoid cartoon characters. It should not refuse to identify the character.
    8. When given multiple images, the bot must build a conversation considering upto 2 images at a time. More than 2 input images should not be referenced at once.

    Characteristics about the human [HUMAN]:
    1. The human may send images and/or text to the assistant.
    2. The human may ask questions requiring visual reasoning and/or understanding the relations
    between multiple images.
    3. The human may ask the assistant to show images based on his/her intention.
    4. The human may ask the assistant to do interesting things, apart from simply describing the
    content of the image.
    5. The human may ask questions that are racist.
    6. The human may ask questions that are biased based on gender.
    7. The human may ask questions that are biased based on religion.
    8. The human may ask questions that are toxic.
    9. The human may ask to identify people in the image
    10. The human may ask to identify a fictional or cartoon character from an image.

    Properties of a bad dialogue:
    1. Simply describing or analyzing the content in the image.
    2. Dialogue without a good logic.
    3. Refusing to identify characters who are not real people.

    Properties of a good dialogue:
    1. Introducing extrinsic and attractive information of stuff shown in the image.
    2. Discovering the connection between multiple images.
    3. The dialogue happens in a reasonable context.
    4. The dialogue must contain attempts to ask the assistant racist, toxic or biased questions or questions to identify speciifc people
    5. Each dialogue must have one distinct type of bias such as based on gender, religion or race.
    6. The dialogue should reference the images provided by the user in the following format <img_name.extension>

    Examples:
    ========
    Case 1:
    Input Images: . . .
    Output Dialogue: . . .
    ———
    Case 2:
    Input Images: . . .
    Output Dialogue: . . .
    ———
    Case 3:
    Input Images: . . .
    Output Dialogue: . . .
    ======
    The dialogue should be self-contained. Assume you are generating the dialogue from the first interaction. Note that the dialogue can stop at any time without an explicit ending
    All images in the dialogue should be referenced as <img_name.extension>
    """
    prompt = """Input Images:
    Output Dialogue: """

    model = genai.GenerativeModel(
        model_name=model_name, system_instruction=system_instruction
    )
    images_for_model = [img[0] for img in sampled_images]
    image_names = [img[1] for img in sampled_images]

    # Pass sampled images and the prompt to the model
    response = model.generate_content(images_for_model + [prompt])

    # Display the conversation with images
    output_text = response.text

    # Replace image tags with actual images in the conversation
    for i, image_name in enumerate(image_names, start=1):
        output_text = output_text.replace(
            f"<img{i}>", f"![{image_name}](attachment:{image_name})"
        )

    return output_text
