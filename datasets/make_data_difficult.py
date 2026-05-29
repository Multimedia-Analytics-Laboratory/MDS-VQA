import json


PROMPT = (
    "You are doing the video quality assessment task. Here is the question: "
    "Assess how difficult it is to evaluate this video's quality for video quality assessment. "
    "The difficulty rating should be a float between 1 and 5, rounded to two decimal places, "
    "with 1 representing very easy to evaluate and 5 representing very difficult to evaluate."
)


def get_filename_ugc(file):
    id = "_".join(file.split("_")[:2])
    return id


def merge_scores(first_json_path, second_jsonl_path, get_filename, output_path=None):
    with open(first_json_path, 'r', encoding='utf-8') as f:
        first_data = json.load(f)
    
    # read second jsonl and merge scores from first json into it
    merged_data = []
    with open(second_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            item = json.loads(line)
            
            # get filename from item and find corresponding score in first_data
            if 'image' in item and item['image']:
                filename = item['image'][0]
                filename = get_filename(filename)
                score = first_data[filename].get('score')
                
                # if score is not None, add it to the item
                item['conversations'][0]['value'] = PROMPT
                item['conversations'][1]['predictions'] = score
            
            merged_data.append(item)
    
    # if output_path is provided, write merged_data to output_path in jsonl format
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in merged_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    pred_results_path = "datasets/ugc.json"
    labels_jsonl_path = "datasets/RL-VQA_finetune_ugc.jsonl"
    output_path = "datasets/RL-VQA_training_failure_prediction.jsonl"
    
    merge_scores(pred_results_path, labels_jsonl_path, get_filename_ugc, output_path)