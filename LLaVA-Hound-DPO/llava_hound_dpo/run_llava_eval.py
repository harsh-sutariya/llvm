from tqdm import tqdm
import json
import argparse
import torch

from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_token, process_images

from PIL import Image

import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map="balanced") # Add device_map = "balanced"
    
    model = model.to(torch.float32)
    print( "Model Processor dtype: ", model.dtype )

    # qs = args.query
    # if model.config.mm_use_im_start_end:
    #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    # else:
    #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()

    with open(args.data_path, "r") as ofile:
        test_data = json.load( ofile )

    res = []

    for example in tqdm( test_data ):
        qs = example['prompt']

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # image = load_image(args.image_file)
        images = [
                load_image(args.data_image + i) for i in example['image']
        ]

        processed_images = []
        keys = []
        for image in images:
            # image = image_processor['image'].preprocess(image, return_tensors='pt')['pixel_values'][0].half()
            # image = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
            processed_images.append(image)
            keys.append("image")

        print("Total images: ", len(processed_images))

        image_tensor = process_images(processed_images, image_processor['image'], model.config)
        print("image_tensor before unsqueeze: ", image_tensor.shape )
        image_tensor = image_tensor.unsqueeze(0).to(model.device).to(dtype=model.dtype)

        print( "Image tensor device:", image_tensor.device )
        print("Image tensor shape:", image_tensor.shape )
        print("Image tensor: ", image_tensor)

        # image_tensor = torch.cat(processed_images).unsqueeze(0).to(model.device) # probably mention the model.device
        # image_tensor = [image_tensor, keys]
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['IMAGE'], return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[image_tensor, keys],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        example['model_output'] = outputs
        res.append( example )
    return res
        
        
if __name__ == "__main__":
    # --data_path : json data
    # --data_image : image folder

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_image", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--output_json_path", type=str, default="/scratch/spp9399/mia_output.json")

    args = parser.parse_args()

    res = eval_model(args)
    with open(args.output_json_path, "w") as ofile:
        json.dump(res, ofile, indent=4)
