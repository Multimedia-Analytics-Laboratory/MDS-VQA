from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
import random
import re
import os
from tqdm import tqdm
import json
import argparse


def score_video(image_path, model, processor, quality_assessment=False):
    if not quality_assessment:
        PROMPT = (
            "You are doing the video quality assessment task. Here is the question: "
            "Assess how difficult it is to evaluate this video's quality for video quality assessment. "
            "The difficulty rating should be a float between 1 and 5, rounded to two decimal places, "
            "with 1 representing very easy to evaluate and 5 representing very difficult to evaluate."
        )
        QUESTION_TEMPLATE = "{Question} Please only output the final answer with only one score in <answer> </answer> tags."
    else:        
        PROMPT = (
            "You are doing the video quality assessment task. Here is the question: "
            "What is your overall rating on the quality of this video? The rating should be a float between 1 and 5, "
            "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
        )
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags."

    message = [
        {
            "role": "user",
            "content": [
                {'type': 'video', 'video': image_path},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=PROMPT)}
            ],
        }
    ]

    # Preparation for inference
    text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, add_vision_id=True)]
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=True, top_k=50, top_p=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        if not quality_assessment:
            reasoning = "N/A"
        else:
            reasoning = re.findall(r'<think>(.*?)</think>', batch_output_text[0], re.DOTALL)
            reasoning = reasoning[-1].strip()
        
        model_output_matches = re.findall(r'<answer>(.*?)</answer>', batch_output_text[0], re.DOTALL)
        model_answer = model_output_matches[-1].strip() if model_output_matches else batch_output_text[0].strip()
        score = float(re.search(r'\d+(\.\d+)?', model_answer).group())
    except:
        print(f"================= Meet error with {image_path}, please generate again. =================")
        score = batch_output_text

    return reasoning, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for video quality assessment and failure prediction.")
    parser.add_argument("--ckpt-path", 
                        type=str,
                        default="output/MDS-VQA-Failure-Predictor",
                        # default="output/MDS-VQA-Active-Finetuning",
                        help="Path to the model checkpoint.")
    parser.add_argument("--quality-assessment", 
                        action='store_true', 
                        help="Whether to only output the score without reasoning.")
    parser.add_argument("--video-roots",
                        nargs='+',
                        help="List of root directories containing videos for inference.")
    parser.add_argument("--save-paths",
                        nargs='+',
                        help="List of paths to save the inference results.")
    parser.add_argument("--filter-paths",
                        nargs='+',
                        default=["datasets/sdr/sdr_train_set.txt"],
                        help="List of paths to the inference set files containing video names for filtering.")
    args = parser.parse_args()

    MODEL_PATH = args.ckpt_path
    video_roots = args.video_roots
    save_paths = args.save_paths
    filter_paths = args.filter_paths
    quality_assessment = args.quality_assessment
    
    random.seed(42)
    device = torch.device("cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"

    for video_root, save_path, filter_path in zip(video_roots, save_paths, filter_paths): 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pred_dict = {}
        video_paths = []

        for foldername, subfolders, filenames in os.walk(video_root):
            for filename in filenames:
                if filename.endswith('.mp4'):
                    file_path = os.path.join(foldername, filename)
                    video_paths.append(file_path)

        filter_list = []
        with open(filter_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n').split(" ")[0]
                filter_list.append(os.path.join(video_root, line))

        video_paths = [path for path in video_paths if path in filter_list]
        for file in tqdm(video_paths):
            try:
                reasoning, score = score_video(
                    file, model, processor, quality_assessment=quality_assessment
                )

                preds = {}
                id = os.path.basename(file)

                preds['reasoning'] = reasoning
                preds['score'] = score

                pred_dict[id] = preds
            except Exception as e:
                print(f"Error {file} ============================")
                print(e)

                with open(save_path, "w", encoding='utf-8') as f:
                    f.write(json.dumps(pred_dict, indent=4))

        with open(save_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(pred_dict, indent=4))