import argparse
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO

# Hook utilities for attention
class AttentionHook:
    """Class to capture and modify attention computations via PyTorch hooks."""
    
    def __init__(self, 
                 confidence_low_threshold: float = 0.3, 
                 confidence_high_threshold: float = 0.7,
                 alpha_low: float = 0.5, 
                 alpha_high: float = 2.0):
        """
        Initialize the hook parameters.
        
        Args:
            confidence_low_threshold: Below this confidence, smooth attention (alpha_low)
            confidence_high_threshold: Above this confidence, sharpen attention (alpha_high)
            alpha_low: Temperature scaling for low confidence (> 1 for smoothing)
            alpha_high: Temperature scaling for high confidence (< 1 for sharpening)
        """
        self.confidence_low_threshold = confidence_low_threshold
        self.confidence_high_threshold = confidence_high_threshold
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.hooks = []
        self.current_confidence = 1.0  # Default confidence (no adjustment)
        self.active = True  # Whether hooks are active
        
    def update_confidence(self, confidence: float):
        """Update the current confidence score."""
        self.current_confidence = confidence
        
    def get_alpha(self) -> float:
        """Determine temperature scaling factor based on current confidence."""
        if self.current_confidence < self.confidence_low_threshold:
            return self.alpha_low
        elif self.current_confidence > self.confidence_high_threshold:
            return self.alpha_high
        else:
            return 1.0
        
    def attn_hook(self, module, inputs, outputs):
        """PyTorch forward hook to modify attention before softmax is applied."""
        if not self.active:
            return outputs
            
        # Extract the query-key attention scores before softmax
        # Typically, attention output is (batch_size, num_heads, seq_len, key_len)
        attn_weights = outputs[0]  # Get attention weights
        
        # Get scaling factor based on confidence
        alpha = self.get_alpha()
        
        # In MIA-DPO, we need to identify which tokens are image tokens
        # This is a simplification and may need to be adjusted based on actual model architecture
        # For LLaVA, image tokens are typically at the beginning of the sequence
        # We identify the image token count from the module configuration
        
        # Scale the attention weights for image tokens
        # This assumes the attention matrix has image tokens in a known location
        # May need adjustment based on the specific model architecture
        
        # Apply the scaling (temperature adjustment)
        scaled_weights = attn_weights * alpha
        
        # Replace the original attention weights
        outputs = (scaled_weights,) + outputs[1:]
        return outputs
        
    def register_hooks(self, model):
        """Register hooks on all relevant attention modules."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Find all attention modules in the model
        # This is specific to LLaVA/LLaMA architecture and may need adjustment
        for name, module in model.named_modules():
            # Target cross-attention modules
            if "self_attn" in name and hasattr(module, "forward"):
                # Register a hook on the attention computation
                hook = module.register_forward_hook(self.attn_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def deactivate(self):
        """Temporarily deactivate hooks without removing them."""
        self.active = False
        
    def activate(self):
        """Activate hooks."""
        self.active = True


def calculate_confidence(logits: torch.Tensor) -> float:
    """
    Calculate a confidence score from next-token prediction logits.
    
    Args:
        logits: The logits output from the model for next token prediction
        
    Returns:
        float: A confidence score between 0 and 1
    """
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Take the highest probability as confidence
    confidence = torch.max(probs).item()
    
    return confidence


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


    answers_file = os.path.expanduser(decoder_args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for batch in tqdm( eval_dataloader, desc="Evaluating"):
        
        prompt_ids = batch["prompt_ids"][0].unsqueeze(0).to( model.device )
        image = batch["images"][0].to( model.device )
        keys = batch['keys']

        idx = batch["id"][0]
        
        stop_str = conv_templates[data_args.conv_mode].sep if conv_templates[data_args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[data_args.conv_mode].sep2

        # Decoder args!
        with torch.inference_mode():
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
    parser = argparse.ArgumentParser(description="Run inference with AdaptVis")
    
    # Model parameters
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model weights")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path")
    
    # Data parameters
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to input JSON data file")
    parser.add_argument("--image-dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save output JSON results")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument("--top-p", type=float, default=0.95,
                       help="Top-p sampling threshold")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"],
                       help="Model precision")
    parser.add_argument("--conv-mode", type=str, default=None,
                       help="Conversation mode, inferred from model if not specified")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad",
                       help="Image aspect ratio treatment: 'pad' or 'crop'")
    
    # AdaptVis parameters
    parser.add_argument("--use-adaptvis", action="store_true",
                       help="Whether to use AdaptVis adaptive attention")
    parser.add_argument("--confidence-low-threshold", type=float, default=0.3,
                       help="Below this confidence, smooth attention")
    parser.add_argument("--confidence-high-threshold", type=float, default=0.7,
                       help="Above this confidence, sharpen attention")
    parser.add_argument("--alpha-low", type=float, default=0.5,
                       help="Temperature scaling for low confidence (< 1 for smoothing)")
    parser.add_argument("--alpha-high", type=float, default=2.0,
                       help="Temperature scaling for high confidence (> 1 for sharpening)")
    
    # Test mode parameters
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with a small subset of data")
    parser.add_argument("--test-size", type=int, default=3,
                       help="Number of examples to process in test mode")
    parser.add_argument("--test-max-tokens", type=int, default=100,
                       help="Maximum new tokens to generate in test mode")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output for debugging")
    
    args = parser.parse_args()
    
    # Run inference
    results = evaluate(attn_implementation="flash_attention_2", args=args)
    
    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {args.output_path}")