export image_dir=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/
export data_path=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/sequence_test_images.json
export output_file=/scratch/hs5580/mia2/MIA-DPO/answer_base_model.json

# --model_name_or_path "/scratch/spp9399/llava_lora_our_loss" \

python3 ./my_eval.py \
    --base_model_name_or_path "liuhaotian/llava-v1.5-7b" \
    --model_name_or_path "/scratch/spp9399/mia/llava_lora_our_loss" \
    --version v1 \
    --data_path ${data_path} \
    --image_folder ${image_dir} \
    --X "Image" --training_modal 'image' \
    --image_tower "/scratch/spp9399/models/clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --top_p 0.7 \
    --temperature 0.2 \
    --answers_file ${output_file} \
    --adaptive_attention False \
    --confidence_threshold_low 0.3 \
    --confidence_threshold_high 0.7 \
    --alpha_low 0.5 \
    --alpha_high 2.0
