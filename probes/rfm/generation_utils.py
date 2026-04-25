import requests
import os
import io
from PIL import Image


def extract_image(image_input):
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # Handle URL
                response = requests.get(image_input, stream=True)
                image = Image.open(response.raw)
            elif os.path.exists(image_input):
                # Handle local path
                image = Image.open(image_input)
        else:
            image = Image.open(io.BytesIO(image_input))
    return image



def format_image_prompt(model, processor, input_text, image_input, **kwargs):

    """
    Generate text from multimodal input (text + image) or text-only input.
    
    Args:
        model: The model to use for generation
        processor: The processor/tokenizer for the model
        input_: Dict containing:
            - 'text': str, the text prompt
            - 'image': str (path/url) or PIL.Image or bytes or None
        **kwargs: Additional generation parameters
    
    Returns:
        str: Generated text response
    """

    image = None

    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # Handle URL
                response = requests.get(image_input, stream=True)
                image = Image.open(response.raw)
            elif os.path.exists(image_input):
                # Handle local path
                image = Image.open(image_input)
        else:
            image = Image.open(io.BytesIO(image_input))

    elif isinstance(image_input, Image.Image):
        # Already a PIL Image
        image = image_input
    
    if image is None:
        raise ValueError("Invalid image input")
    
    # Format messages for the model with image
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": input_text}
        ]}
    ]
    
    # Prepare inputs
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=input_text,
        images=image,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    return inputs

def generate_on_image_and_text_deepseek(llm, processor, plaintext_prompt, image_path, **kwargs):

    image = extract_image(image_path)
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{plaintext_prompt}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = [image]
    prepare_inputs = processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(llm.language_model.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = llm.language_model.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = llm.language_model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=llm.tokenizer.eos_token_id,
        bos_token_id=llm.tokenizer.bos_token_id,
        eos_token_id=llm.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    generated_text = llm.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    return generated_text



def generate_on_image_and_text(model, processor, plaintext_prompt, image, **kwargs):
    tensor_inputs = format_image_prompt(model, processor, plaintext_prompt, image, **kwargs)
    stop_token_id = processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # print(tensor_inputs)

    # print(tensor_inputs.shape)
    outputs = model.generate(**tensor_inputs,
                            #  eos_token_id=stop_token_id,
                            **kwargs)
    generated_text = processor.decode(outputs[0])    
    return generated_text

def generate_on_text(model, tokenizer, input_text, **kwargs):
    # print("Generating on text: ", input_text)
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    # stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # Generate output
    outputs = model.generate(
        **inputs,
        # eos_token_id=stop_token_id,     
        pad_token_id=tokenizer.eos_token_id,   
        **kwargs,
        # stopping_criteria=[StoppingCriteriaList([stop_token_id])])        
    )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0])
    return generated_text
    
def hook_model(model, directions, layers_to_control, control_coef):
    hooks = {}
    
    # For multimodal models, hook only the language model layers
    if hasattr(model, 'language_model'):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
        
    for layer_idx in layers_to_control:
            
        control_vec = directions[layer_idx]
        if len(control_vec.shape)==1:
            control_vec = control_vec.reshape(1,1,-1)

        block = layers[layer_idx]

        def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef):
            new_output = output[0] if isinstance(output, tuple) else output
            new_output = new_output + control_coef*control_vec.to(dtype=new_output.dtype, device=new_output.device)
            
            if isinstance(output, tuple):
                new_output = (new_output,) + output[1:] 
            
            return new_output
        
        hook_handle = block.register_forward_hook(block_hook)
        hooks[layer_idx] = hook_handle
    
    return hooks


def clear_hooks(hooks) -> None:
    for hook_handle in hooks.values():
        hook_handle.remove()