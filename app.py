import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import streamlit as st
import os
from huggingface_hub import login
import torchvision
from torchvision import transforms, models
import timm
import torch.nn as nn
from dotenv import load_dotenv
import warnings
from PIL import Image
import re


device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

load_dotenv()
api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
login(token = api_key)

#Loading up the LLM and the torch model for predictions and caption generation
# Correct way to define a path
model_path = r'D:\LLama trained' 
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
tokenizer.pad_token = tokenizer.eos_token
llm_model = AutoModelForCausalLM.from_pretrained(model_path)
llm_model.to(device)

checkpont_path = r"C:\Users\ghoul\Chest-X-Ray-Captioning\model_epoch_10.pth"
label_columns = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
            'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
]
xray_model = timm.create_model("resnet50", pretrained = True, num_classes=len(label_columns))
xray_model.fc = nn.Linear(xray_model.num_features, len(label_columns))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    checkpoint = torch.load(checkpont_path, map_location=torch.device("cpu"))

xray_model.load_state_dict(checkpoint)
xray_model.to(device)

# Transform function to preprocess the image after upload
def transform_images(img):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed_img = preprocess(img)
    transformed_img = transformed_img.unsqueeze(0)  
    print("image has transformed")
    return transformed_img


# Inference function for caption generation
def inference(llm_model, input_text, max_input_tokens = 1000, max_output_tokens = 150):
    encodings = tokenizer(
        input_text, 
        padding= True,
        truncation = True,
        return_tensors = "pt",
        max_length = max_input_tokens
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"]

    generated_text = llm_model.generate(
        input_ids,
        attention_mask = attention_mask,
        max_new_tokens = max_output_tokens,
        eos_token_id= tokenizer.eos_token_id,
        #temperature=0.7, # Adjust temperature for more deterministic responses 
        # top_k=50, 
        # top_p= 0.9,
        # repetition_penalty=1.2
    )

    answer = tokenizer.batch_decode(generated_text, skip_special_tokens=True)
    print("caption has been generated")
    return answer


# Diagnosis prediction function
def get_prediction(transformed_image, labels, model):
    image = transformed_image.to(torch.device('cpu'))
    threshold = 0.5
    with torch.no_grad():  
        prediction = model(image)
        logits = torch.tensor(prediction)
        probabilities = torch.sigmoid(logits)
        active_indices = (probabilities > threshold).nonzero(as_tuple=True)[1]
        print("prediction has been made")
        return [labels[i] for i in active_indices]


def post_process_caption(caption, instruction):
    # If the input is a list, join it into a single string
    if isinstance(caption, list):
        caption = " ".join(caption)

    # Remove everything before and including '### Response'
    caption = caption.replace(instruction.strip(), '').strip()

    caption = re.sub(r'.*### Response\s*', '', caption)

    # Remove the instruction part
    caption = caption.replace(instruction, '').strip()

    # Ensure the input is a string
    if not isinstance(caption, str):
        raise TypeError("Expected a string, but got: {}".format(type(caption)))

    # Remove repeated sentences or phrases
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', caption)  # Split into sentences
    unique_sentences = []
    for sentence in sentences:
        if sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())
    caption = " ".join(unique_sentences)
    
    # Handle incomplete phrases and cleanup
    caption = caption.replace("the are intact", "")  # Fix specific phrases
    caption = re.sub(r'\.{2,}', '.', caption)  # Replace multiple periods with a single period
    caption = re.sub(r'\s+', ' ', caption)  # Replace multiple spaces with a single space

    # Ensure proper capitalization and punctuation
    caption = caption.strip()
    if caption and not caption.endswith('.'):
        caption += '.'  # Add a period if it's missing
    caption = caption[0].upper() + caption[1:]  # Capitalize the first letter
    print(caption)

    return caption




# App Layout is here
def main():
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Chest X-Ray Radiologist Bot")

    with st.sidebar:
        st.subheader("Your X-Ray Images")
        image = st.file_uploader(
            "Upload your X-Ray image here", accept_multiple_files=False
        )

        if image is not None:
            pil_image = Image.open(image)
            #st.image(pil_image, caption="Uploaded X-Ray", use_column_width=True)
            caption = ""

            if st.button("Process"):
                with st.spinner("Processing..."):
                    # Transform the image
                    transformed_image = transform_images(pil_image)
                    
                    # Get the diagnosis prediction
                    prediction = get_prediction(transformed_image, label_columns, xray_model)
                    keyword_str = ", ".join(prediction)
                    
                    # Instruction for the LLM
                    if keyword_str == "":
                        instruction = """### Instruction:
                                        Summarize the following findings into a concise medical caption.

                                         ### Findings:
                                        No problems found in lungs.

                                         ### Response:
                                        """
                    else:
                        instruction = f"""### Instruction:
                                            Please generate a concise caption for the result {keyword_str}
                                            
                                            ### Response:
                                            """
                    print(keyword_str)
                    print(instruction)
                    
                    # Generate and post-process caption
                    uncleaned_caption = inference(llm_model, instruction)
                    caption = post_process_caption(uncleaned_caption, instruction)
                    # Generate and post-process caption
                    uncleaned_caption = inference(llm_model, instruction)
                    print(f"Uncleaned Caption: {uncleaned_caption}")  # Debugging line
                    caption = post_process_caption(uncleaned_caption, instruction)
                    print(f"Cleaned Caption: {caption}")  # Debugging line


# Layout for Results
# st.markdown("## Chest X-Ray Radiologist Bot")
    st.markdown("---")

    if image is not None:
        # Two-column layout for results
        col1, col2 = st.columns(2)

        # Display image in the first column
        with col1:
            st.image(pil_image, caption="Processed X-Ray", use_container_width=True)

        # Display caption in the second column
        with col2:
            st.subheader("Generated Caption:")
            if caption:
                st.markdown(f"{caption}")
            else:
                st.markdown("_No caption generated yet._")
    else:
        # Placeholder for illustration when no image is uploaded
        st.image(r"C:\Users\ghoul\Chest-X-Ray-Captioning\xray image_placeholder.jpg", use_container_width=False, caption="Upload an X-Ray to get started!")

    st.markdown("---")
    st.markdown(
        """
        <style>
            .css-18e3th9 {  # Reduces padding for the Streamlit app
                padding-top: 1rem;
            }
            .css-1d391kg {  # Removes extra margin above the title
                margin-top: 0rem;
            }
        </style>
        <div class="footer">
            Built with Streamlit, Pytorch and Huggingface 🤗
        </div>
    """,
    unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
