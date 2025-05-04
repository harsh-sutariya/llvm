import os
import copy
import random
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Tuple, Union
import torch

import transformers
import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

from llava.constants import DEFAULT_X_TOKEN, IGNORE_INDEX
from torch.utils.data import Dataset
from utils import load_jsonl, load_json

from llava import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
from llava.model import *
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from trl.trainer.utils import DPODataCollatorWithPadding

from tqdm import tqdm
from PIL import Image

from llava.conversation import conv_templates, SeparatorStyle
from peft import PeftModel

from torch.utils.data import DataLoader
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    X: Optional[List[str]] = field(default=None)
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_x_start_end: bool = field(default=False)
    mm_use_x_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    bf16: bool = field(default=False)
    model_max_length: int = field(default=4096)
    base_model_name_or_path: Optional[str] = field(default="")
    device: str = field(default="cuda")

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    training_modal: Optional[str] = field(default='video')
    num_sample: Optional[int] = field(default=None)
    conv_mode: str = field(default="llava_v1")

@dataclass
class DecoderArguments:
    temperature:  float = field(default=0)
    top_p: Optional[float] = field(default=1.0)
    num_beams: int = field(default=1)
    max_new_tokens: int = field(default=128)
    answers_file: str = field(default="./answer.json")
    # AdaptVis arguments
    adaptive_attention: bool = field(default=False)
    confidence_threshold_low: float = field(default=0.3)
    confidence_threshold_high: float = field(default=0.7)
    alpha_low: float = field(default=0.5)
    alpha_high: float = field(default=2.0)

def load_data(data_args):
    if 'jsonl' in data_args.data_path:
        data_list = load_jsonl(data_args.data_path)
    else: 
        data_list = load_json(data_args.data_path)
    return data_list


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, X : str = None) -> Dict:
    conv = conv_templates["llava_v1"].copy() # hard coding rn!
    conv.append_message(conv.roles[0], sources)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize conversations

    # input_ids = torch.stack([tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
    
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
    return input_ids


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_X: str = None
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    # conv = conv_templates[args.conv_mode].copy()
    # conv.append_message(conv.roles[0], qs) # role=human
    # conv.append_message(conv.roles[1], None) # role=assistant

    X = has_X if has_X is None else has_X.upper()
    return preprocess_v1(sources, tokenizer, X=X)

class DPODataset(Dataset):
    """Dataset for inference"""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):

        super(Dataset, self).__init__()
        list_data_dict = load_data(data_args)
        if data_args.num_sample is not None:
            list_data_dict = list_data_dict[:data_args.num_sample]

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.training_modal = data_args.training_modal

    def __len__(self):
        # return 20
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        '''
        {
            'id': '....'
            'prompt': '<image>\n In Image1 Is there a snowman wearing a green scarf and hat in the background?',
            'image_path': '/mnt/bn/liangkeg/data/ruohongz/dpo_data/dpo_images/LRVInstruction-000000009569.jpg',
        }
        '''
        try:
            has_X = None
            data_dict = copy.deepcopy(self.list_data_dict[i]) # inplace modification following
            if type(data_dict['image'])==str:
                image_file = data_dict['image']
                image_folder = self.data_args.image_folder
                processor = self.data_args.image_processor
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                prompt = data_dict['prompt']
                prompt = prompt.replace("<image>", "").strip()
                prompt = "<image>\n" + prompt
                data_dict['prompt'] = prompt
                has_X = 'image'

            elif type(data_dict['image'])==list:
                processed_images = []
                for image_file in data_dict['image']:
                    image_folder = self.data_args.image_folder
                    processor = self.data_args.image_processor
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        image = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
                        processed_images.append(image)
                    else:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        image = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
                        processed_images.append(image)
                image = torch.cat(processed_images)

                prompt = data_dict['prompt']
                data_dict['prompt'] = prompt
                has_X = 'image'

            data_dict['has_X'] = has_X
            if has_X == 'image':
                data_dict['image'] = image

            return data_dict
        except Exception as e:
            print(f'Error with {e}, {self.list_data_dict[i]}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    def collate(self, batch):
        # first, pad everything to the same length
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX)
        # input_ids = input_ids[:, :self.tokenizer.model_max_length]
        # labels = labels[:, :self.tokenizer.model_max_length]
        # batch = dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue

                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        
        return padded_batch


    def tokenize_batch_element(
        self,
        prompt: str,
        has_X: str = None
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # import pdb; pdb.set_trace()
        batch = {}
        
        prompt_data_dict = preprocess(
            prompt,
            self.tokenizer,
            has_X=has_X
        )

        batch['prompt_ids'] = prompt_data_dict

        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        Xs, keys = [], []
        ids = []
        for feature in features:
            prompt = feature["prompt"]
            has_X = feature['has_X']
            Xs.append(feature[has_X])
            keys.append(has_X)
            ids.append( feature['id'] )
            
            batch_element = self.tokenize_batch_element(prompt, has_X=has_X)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch =  self.collate(tokenized_batch)
        padded_batch['images'] = Xs  # we do not change the key's name.
        padded_batch['keys'] = keys
        padded_batch['id'] = ids

        return padded_batch


# AdaptVis attention scaling implementation for model.generate
class AdaptVisAttentionScaling:
    def __init__(self, model, confidence_threshold_low=0.3, confidence_threshold_high=0.7, 
                 alpha_low=0.5, alpha_high=2.0):
        self.model = model
        self.confidence_threshold_low = confidence_threshold_low
        self.confidence_threshold_high = confidence_threshold_high
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        
        # Store the original attention function
        self.original_attention_forward = None
        self.hooked = False
        self.image_token_start = None
        self.image_token_end = None
        
    def get_attention_scaling_hook(self):
        """Create a hook function to modify cross-attention based on token confidence"""
        def attention_scaling_hook(module, inputs, outputs):
            # This hook will be called during the forward pass of the attention module
            # We modify the cross-attention between the *last* text token and image tokens
            
            # Get attention weights (query x key) before softmax
            attention_weights = outputs[1]  # [batch_size, num_heads, seq_len, seq_len]
            
            # Use the fixed alpha determined after the first generation step
            if hasattr(self, 'fixed_alpha_for_generation'):
                alpha = self.fixed_alpha_for_generation
                
                # Apply the scaling only to the last token's attention to image tokens
                if alpha != 1.0 and self.image_token_start is not None and self.image_token_end is not None:
                    # Check if we are in the generation phase (seq_len > initial_len)
                    # This hook runs on every forward pass, including the first one where alpha should be 1.0
                    # We rely on fixed_alpha_for_generation being set *after* the first pass.
                    # A potentially cleaner way might involve checking past_key_values length.
                    
                    scaled_attention = attention_weights.clone()
                    scaled_attention[:, :, -1, self.image_token_start:self.image_token_end] *= alpha
                    return (outputs[0], scaled_attention) + outputs[2:]
            
            # If fixed_alpha_for_generation is not set (e.g., first step), or alpha is 1.0, return original
            return outputs
        
        return attention_scaling_hook
    
    def find_image_token_positions(self, input_ids):
        """Find the start and end positions of image tokens in the sequence"""
        # In LLaVA, image tokens are marked with IMAGE_TOKEN_INDEX (-200)
        # The image tokens are inserted where the <image> token was in the prompt
        image_token_id = -200  # LLaVA's IMAGE_TOKEN_INDEX
        
        # Find where the image tokens are
        image_token_pos = (input_ids == image_token_id).nonzero()
        if len(image_token_pos) > 0:
            # The image token position marks where the image patches start
            self.image_token_start = image_token_pos[0].item()
            # Each image has 256 patch tokens following the image token
            self.image_token_end = self.image_token_start + 257  # Include the image token itself
        else:
            print("No image tokens found in input")
            self.image_token_start = None
            self.image_token_end = None
    
    def hook_model(self):
        """Add hooks to the model's attention modules"""
        if self.hooked:
            return
            
        # Find all cross-attention modules in the model
        # This is model-specific and would need to be adapted
        for name, module in self.model.named_modules():
            # Look for cross-attention modules that attend to image tokens
            if "crossattention" in name.lower() or "cross_attention" in name.lower():
                # Store original forward method
                if self.original_attention_forward is None:
                    self.original_attention_forward = module.forward
                    
                # Register hook
                module.register_forward_hook(self.get_attention_scaling_hook())
        
        self.hooked = True
    
    def unhook_model(self):
        """Remove hooks and restore original forward methods"""
        if not self.hooked:
            return
            
        # Restore original forward methods
        for name, module in self.model.named_modules():
            if "crossattention" in name.lower() or "cross_attention" in name.lower():
                if hasattr(module, "_forward_hooks"):
                    module._forward_hooks.clear()
                    
        self.hooked = False
        
    def generate_with_adaptive_attention(self, *args, **kwargs):
        """Wrapper around the model's generate method that implements adaptive attention scaling
           following the original AdaptVis strategy (fixed alpha after first token).
        """
        self.hook_model()
        
        input_ids = kwargs.get("input_ids")
        images = kwargs.get("images", None)
        max_new_tokens = kwargs.get("max_new_tokens", 20)
        eos_token_id = kwargs.get("eos_token_id")
        
        # Find image token positions at the start
        self.find_image_token_positions(input_ids[0])
        
        # --- Step 1: Generate the first token without scaling --- 
        self.current_confidence = None # Ensure no scaling for the first token
        alpha = 1.0 # Explicitly set alpha to 1 for the first step
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                images=images,
                use_cache=True, # Use cache for subsequent steps
                return_dict=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Sample the first token (greedy for simplicity here, adapt if using sampling)
            first_next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([input_ids, first_next_token], dim=-1)

            # --- Step 2: Calculate confidence based on the first token --- 
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            first_token_confidence = torch.max(probs, dim=-1)[0].item()
            print(f"First token confidence: {first_token_confidence:.3f}")

            # --- Step 3: Determine the fixed scaling factor for the rest --- 
            if first_token_confidence < self.confidence_threshold_low:
                fixed_alpha = self.alpha_low
                print(f"Using fixed alpha_low: {fixed_alpha}")
            elif first_token_confidence > self.confidence_threshold_high:
                fixed_alpha = self.alpha_high
                print(f"Using fixed alpha_high: {fixed_alpha}")
            else:
                fixed_alpha = 1.0
                print(f"Using fixed alpha: {fixed_alpha} (moderate confidence)")
            
            # Set the fixed alpha for the hook to use in subsequent steps
            self.current_confidence = first_token_confidence # Store confidence for hook
            # The hook will use the fixed_alpha derived from this confidence
            # Need to modify hook slightly or pass fixed_alpha explicitly
            self.fixed_alpha_for_generation = fixed_alpha

            # --- Step 4: Generate remaining tokens with the fixed scaling factor --- 
            current_token_ids = first_next_token
            
            for i in range(1, max_new_tokens): # Start from 1 since we already generated the first token
                # Prepare inputs for the next step using cache
                model_inputs = self.model.prepare_inputs_for_generation(current_token_ids, past_key_values=past_key_values)
                
                # Forward pass with the fixed alpha applied by the hook
                outputs = self.model(
                    **model_inputs,
                    images=images, # Must pass images even with cache if hook needs them?
                    return_dict=True,
                    use_cache=True,
                    output_attentions=False, # Avoid storing attentions unless needed
                    output_hidden_states=False
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values # Update cache
                
                # Sample next token (greedy)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                current_token_ids = next_token # Next input is just the new token
                
                # Check for EOS
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        self.unhook_model()
        delattr(self, "fixed_alpha_for_generation") # Clean up
        
        return generated_ids


def evaluate(attn_implementation):

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DecoderArguments))
    model_args, data_args, decoder_args = parser.parse_args_into_dataclasses()
    
    """
    print("Loading base model from: ", model_args.base_model_name_or_path )
    
    model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.base_model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if model_args.bf16 else torch.float16),
        )

    # Loading Peft
    if model_args.model_name_or_path:
        print("Loading Peft Model from: ", model_args.model_name_or_path )

        model = PeftModel.from_pretrained( model, model_args.model_name_or_path )

        print("Mering LoRa with Base model")
        model = model.merge_and_unload()
    else:
        print("No PEFT Model")

    model = model.to(torch.bfloat16 if model_args.bf16 else torch.float16)
    model = model.to("cuda")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    image_tower = model.get_image_tower()
    if image_tower is None:
        model.get_model().initialize_image_modules(
            model_args=model_args
        )
        image_tower = model.get_image_tower()
    if not image_tower.is_loaded:
        # print('load image tower')
        image_tower.load_model()


    image_tower.to(dtype=torch.bfloat16 if model_args.bf16 else torch.float16, device="cuda")

    model.image_tower = image_tower 

    data_args.image_processor = image_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

    model.initialize_X_tokenizer(model_args, tokenizer=tokenizer)


    """

    disable_torch_init()
    model_path = os.path.expanduser( model_args.model_name_or_path )
    model_name = get_model_name_from_path(model_path)

    # torch_dtype=(torch.bfloat16 if model_args.bf16 else torch.float16)
    tokenizer, model, processor, context_len = load_pretrained_model( model_path,  model_args.base_model_name_or_path, model_name,  bf16 = model_args.bf16)
    data_args.image_processor = processor['image']

    eval_dataset = DPODataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    collator = DPODataCollator(
            tokenizer,
            label_pad_token_id=IGNORE_INDEX,
            pad_token_id=tokenizer.pad_token_id,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,  # or more, but make sure generate() handles padding correctly
        collate_fn=collator
    )

    # Initialize AdaptVis if adaptive attention is enabled
    adaptvis = None
    if decoder_args.adaptive_attention:
        adaptvis = AdaptVisAttentionScaling(
            model,
            confidence_threshold_low=decoder_args.confidence_threshold_low,
            confidence_threshold_high=decoder_args.confidence_threshold_high,
            alpha_low=decoder_args.alpha_low,
            alpha_high=decoder_args.alpha_high
        )
        print(f"AdaptVis enabled with confidence thresholds: {decoder_args.confidence_threshold_low}-{decoder_args.confidence_threshold_high}, " 
              f"alpha values: {decoder_args.alpha_low}-{decoder_args.alpha_high}")

    answers_file = os.path.expanduser(decoder_args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        
        prompt_ids = batch["prompt_ids"][0].unsqueeze(0).to(model.device)
        image = batch["images"][0].to(model.device)
        keys = batch['keys']

        idx = batch["id"][0]
        
        stop_str = conv_templates[data_args.conv_mode].sep if conv_templates[data_args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[data_args.conv_mode].sep2

        # Decoder args!
        with torch.inference_mode():
            if decoder_args.adaptive_attention:
                # Use AdaptVis for generation with adaptive attention scaling
                output_ids = adaptvis.generate_with_adaptive_attention(
                    input_ids=prompt_ids,
                    images=[image.to(dtype=torch.bfloat16 if model_args.bf16 else torch.float16).unsqueeze(0), keys],
                    do_sample=True if decoder_args.temperature > 0 else False,
                    temperature=decoder_args.temperature,
                    top_p=decoder_args.top_p,
                    max_new_tokens=decoder_args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                # Standard generation without AdaptVis
                output_ids = model.generate(
                    prompt_ids,
                    images=[image.to(dtype=torch.bfloat16 if model_args.bf16 else torch.float16).unsqueeze(0), keys],
                    do_sample=True if decoder_args.temperature > 0 else False,
                    temperature=decoder_args.temperature,
                    top_p=decoder_args.top_p,
                    num_beams=decoder_args.num_beams,
                    max_new_tokens=decoder_args.max_new_tokens,
                    use_cache=True)
            
        input_token_len = prompt_ids.shape[1]
        n_diff_input_output = (prompt_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   }
                        ) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    evaluate(attn_implementation="flash_attention_2")
