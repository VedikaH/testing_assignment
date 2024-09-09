import os
import streamlit as st
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,BitsAndBytesConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
@st.cache_resource
def load_model_and_tokenizer(path):
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

@st.cache_resource
def load_gemma_model_and_tokenizer():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    token = os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        raise ValueError("Hugging Face token not found. Please set the 'HUGGINGFACE_TOKEN' environment variable.")

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map="auto",
        token=token,
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=token)
    return model, tokenizer

def refine_instructions(gemma_model, gemma_tokenizer, original_instructions):
    refining_prompt = f"""
    <start_of_turn>user\n
    Refine the following test instructions to improve structure and remove redundancies.Dont give any extra commentry. 
    The Original test instructions are:
    {original_instructions}
    
    Ensure each test case follows this format:

    Feature: [Feature Name] 
    Description: [Brief description] 
    Pre-conditions: [List of pre-conditions] 
    Testing Steps: 
    1. [Step 1]
    2. [Step 2]
    ...
    Expected Result: [Expected outcome]
    
    <end_of_turn>\n <start_of_turn>model\n

    Testing instructions:
    """

    inputs = gemma_tokenizer(refining_prompt, return_tensors="pt").to(gemma_model.device)
    outputs = gemma_model.generate(**inputs, max_new_tokens=1024)
    refined_instructions = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the refined instructions part
    refined_instructions = refined_instructions.split("Testing instructions:")[-1].strip()

    return refined_instructions

st.title("Testing Instructions Generator")
context = st.text_area("Optional Context:", placeholder="Enter any specific context for the testing instructions.")
sys_prompt=f'''
You are tasked with creating concise test cases for mobile app features shown in a image. The image may contain multiple features that need separate test cases. {context} Focus on testing basic and general functionalities without going into excessive detail. Each image many contain more than one feature.
For each feature, generate a test case using the following structure:

Description: Briefly describe the feature being tested.
Pre-conditions: List what must be set up before testing.
Testing Steps: Provide step-by-step instructions for testing the core functionality.
Expected Result: Describe the expected outcome.
'''
# A image is attached. Thoroughly inspect the image to generate a complete set of test cases for each feature. Provide only the required information as per the format, avoiding any additional commentary.

uploaded_files = st.file_uploader("Upload screenshots:", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Describe Testing Instructions"):
    if uploaded_files:
        model, tokenizer = load_model_and_tokenizer('OpenGVLab/InternVL2-2B')
        gemma_model, gemma_tokenizer = load_gemma_model_and_tokenizer()

        pixel_values_list = []
        num_patches_list= []
        for file in uploaded_files:
            pixel_values = load_image(file, max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))

        pixel_values = torch.cat((pixel_values_list), dim=0)

        # Check if you need to pass the image through an image encoder
        # If the model uses a different method for encoding images, modify accordingly
#         image_embeddings = model.encode_images(pixel_values)

#         # Combine image embeddings with text inputs
#         inputs = tokenizer(context, return_tensors="pt").to("cuda")
#         # Adjust this line based on how the model handles text and image inputs together
#         inputs['pixel_values'] = image_embeddings

        # Generate the testing instructions using both text and image inputs
        generation_config = dict(max_new_tokens=1024, do_sample=True)
#         output = model.generate(**inputs, **generation_config)
        questions = [f'<image>\n{sys_prompt}.'] * len(num_patches_list)
        responses = model.batch_chat(tokenizer, pixel_values,
                                     num_patches_list=num_patches_list,
                                     questions=questions,
                                     generation_config=generation_config)
#         instructions = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("**Generated Testing Instructions:**")
        for i, response in enumerate(responses):
            #st.write(f"Original instructions for image {i+1}:")
            #st.write(response)
            
            refined_instructions = refine_instructions(gemma_model, gemma_tokenizer, response)

            
            st.markdown(refined_instructions, unsafe_allow_html=True)
    else:
        st.error("Please upload at least one image.")
