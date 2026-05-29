import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
import argparse
import torch
import cv2
import clip
from transformers import AutoModel, AutoProcessor
from utils import fit_curve, contains_nan


def extract_diversity_features_from_video(video_path, model, preprocess, device, model_name):
    """
    extract diversity features from a video file by sampling frames at regular intervals and processing them through the CLIP model.
    
    Parameters:
        video_path: path to the video file
        model: diversity model
        preprocess: image preprocessing function
        frame_interval: frame interval, extract features every few frames
        model_name: name of the model

    Returns:
        features: extracted feature array
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    features = []
    
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    
    with tqdm(total=total_frames, desc="Extracting frame features") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            if model_name == 'CLIP_RN101':
                image_input = preprocess(frame_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            elif model_name == 'SigLip2':
                inputs = preprocess(images=[frame_pil], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
            
            features.append(image_features.cpu().numpy())

            pbar.update(1)
    
    cap.release()
    
    if len(features) == 0:
        print(f"Failed to extract any features from video: {video_path}")
        return None, None
    
    features = np.vstack(features)
    return features


def extract_diversity_features(input_dir, labels_path, output_dir, model_name='CLIP_RN101'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model_name == 'CLIP_RN101':
        model, preprocess = clip.load("RN101", device=device)
    elif model_name == 'SigLip2':
        ckpt = "google/siglip2-giant-opt-patch16-384"
        model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
        preprocess = AutoProcessor.from_pretrained(ckpt)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    video_ids = set()
    with open(labels_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(" ")
            video_id = parts[0]
            video_ids.add(video_id)
    
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    print(f"Found {len(video_files)} MP4 files")
    video_files = [f for f in video_files if f in video_ids]
    print(f"{len(video_files)} video files match the labels and will be processed")
    
    for video_file in video_files:
        output_filename = os.path.splitext(video_file)[0] + '_features.npz'
        if os.path.exists(os.path.join(output_dir, output_filename)):
            print(f"Feature file already exists for {video_file}, skipping extraction.")
            continue
        output_path = os.path.join(output_dir, output_filename)
        video_path = os.path.join(input_dir, video_file)
        features = extract_diversity_features_from_video(video_path, model, preprocess, device, model_name=model_name)
        
        if features is not None:
            np.savez_compressed(output_path, features=features, video_name=video_file)
            print(f"Feature saved: {output_path}")


def get_pairwise_distance(mat1, mat2):
    """Calculate the bidirectional Chamfer distance between two matrices"""
    dists_12 = cdist(mat1, mat2, 'euclidean')
    dists_21 = cdist(mat2, mat1, 'euclidean')
    min_dists_12 = np.min(dists_12, axis=1)
    min_dists_21 = np.min(dists_21, axis=1)
    return (np.mean(min_dists_12) + np.mean(min_dists_21)) / 2


def select_data(logits, g_values, K, scaling):
    """
    Select K samples using the specified algorithm
    
    Returns:
        selected_indices (list): Indices of selected samples in order
    """
    N = len(logits)
    selected_mask = np.zeros(N, dtype=bool)
    selected_indices = []
    
    distance_cache = np.full((N, N), np.nan)  # cache for pairwise distances
    dist_sum = np.zeros(N)  # the sum of distances to selected samples for each candidate
    
    # initialize by selecting the sample with the highest g(x) value
    first_idx = np.argmax(g_values)
    selected_mask[first_idx] = True
    selected_indices.append(first_idx)
    
    # calculate initial distances from the first selected sample to all others
    for j in range(N):
        if j != first_idx:
            dist = get_pairwise_distance(logits[first_idx], logits[j])
            distance_cache[first_idx, j] = dist
            distance_cache[j, first_idx] = dist
            dist_sum[j] = dist
    
    # interatively select K-1 samples
    for k in tqdm(range(1, K)):
        candidate_mask = ~selected_mask
        candidate_indices = np.where(candidate_mask)[0]
        
        if len(candidate_indices) == 0:
            break
        
        avg_dists = dist_sum[candidate_indices] / k
        
        # calculate total scores for candidates
        scores = g_values[candidate_indices] + scaling * avg_dists
        
        # select the candidate with the highest score
        best_idx_in_candidates = np.argmax(scores)
        best_idx = candidate_indices[best_idx_in_candidates]
        
        # update selected mask and indices
        selected_mask[best_idx] = True
        selected_indices.append(best_idx)
        
        # update distance sums for remaining candidates
        for j in candidate_indices:
            if j != best_idx:
                # chck cache first
                if np.isnan(distance_cache[best_idx, j]):
                    dist = get_pairwise_distance(logits[best_idx], logits[j])
                    distance_cache[best_idx, j] = dist
                    distance_cache[j, best_idx] = dist
                else:
                    dist = distance_cache[best_idx, j]
                
                # update distance sum
                dist_sum[j] += dist
    
    return selected_indices
        

def get_npz_sfv(file):
    id = "_".join(file.split("_")[:-1]) + '.mp4'

    return id


def get_item_name_sfv(file):
    id = os.path.basename(file)
    
    return id


def main_select_data(
    clip_feature_path,
    failure_score_path,
    get_npz_name,
    get_item_name,
    save_path_candidate,
    K,
    scaling
):
    with open(failure_score_path, "r") as f:
        data = json.load(f)
    
    clip_data = {}
    for npz in os.listdir(clip_feature_path):
        if not npz.endswith('.npz'):
            continue
        video_name = get_npz_name(npz)
        clip_feat = np.load(os.path.join(clip_feature_path, npz))
        clip_data[video_name] = clip_feat["features"]

    logits_list = []
    g_values_list = []
    name_list = []
    
    for item in data:
        clip_feat = clip_data[get_item_name(item)]
        g_value = data[item]["score"]
        
        logits_list.append(clip_feat)
        g_values_list.append(g_value)
        name_list.append(get_item_name(item))
        
    logits = logits_list
    g_values = np.array(g_values_list)
    
    selected = select_data(logits, g_values, K=K, scaling=scaling)  # clip: 2.5e-1, siglip2: 7.5e-2
    
    selected_data = []
    for i in selected:
        selected_data.append(name_list[i])
    
    with open(save_path_candidate, "w", encoding='utf-8') as f:
        f.write(json.dumps(selected_data, indent=4))


def gMAD_srcc(gt_path, pred_path, filter_list_path=None):
    mos = {}

    with open(gt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(" ")
            video_id = parts[0]
            mos_score = float(parts[1])
            mos[video_id] = mos_score

    with open(pred_path) as f:
        pred_ori = json.load(f)
    
    if filter_list_path:
        filter_list = []
        with open(filter_list_path, "r") as f:
            filter_list = json.load(f)

    pred_list = []
    mos_list = []
    for item in pred_ori:
        if item not in mos:
            continue
        if filter_list_path and item not in filter_list:
            continue

        pred_list.append(pred_ori[item]["score"])
        mos_list.append(mos[item])

    fitted_x = fit_curve(pred_list, mos_list)

    if contains_nan(fitted_x) or all(x == fitted_x[0] for x in fitted_x):
        fitted_x = pred_list

    srcc = spearmanr(fitted_x, mos_list)[0]
    plcc = pearsonr(fitted_x, mos_list)[0]

    return srcc, plcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run greedy selection for selecting samples based on CLIP features and failure scores")
    parser.add_argument("--video-dir", 
                        type=str,
                        help="Path to the directory containing video files for feature extraction")
    parser.add_argument("--labels-path", 
                        type=str,
                        default="datasets/sdr/sdr_train_mos.txt",
                        help="Path to the file containing labels for the videos")
    parser.add_argument("--failure-score-path", 
                        type=str,
                        default="test_results/training_failure_prediction/sdr_130.json",
                        help="Path to the JSON file containing failure scores")
    parser.add_argument("--diversity-feature-path", 
                        type=str,
                        default="ckpt/MDS-VQA_sdr_diversity_features",
                        help="Path to the directory containing diversity features")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to select")
    parser.add_argument("--save-path-candidate", 
                        type=str, 
                        default="test_results/training_failure_prediction/candidates.json",
                        help="Path to save the selected candidate samples")
    parser.add_argument("--baseline-pred-path",
                        type=str,
                        default="test_results/baseline_sdr.json",
                        help="Path to the JSON file containing baseline predictions on the target dataset")
    parser.add_argument("--dataset",
                        type=str,
                        default="YouTube-SFV SDR",
                        help="Name of the dataset for evaluation")
    parser.add_argument("--model-name",
                        choices=['CLIP_RN101', 'SigLip2'],
                        default="CLIP_RN101",
                        help="Name of the model for feature extraction, options: CLIP_RN101, SigLip2")
    args = parser.parse_args()

    diversity_feature_path = args.diversity_feature_path
    failure_score_path = args.failure_score_path
    save_path_candidate = args.save_path_candidate

    if args.model_name == 'CLIP_RN101':
        scaling = 0.25
    elif args.model_name == 'SigLip2':
        scaling = 7.5e-2
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    os.makedirs(os.path.dirname(save_path_candidate), exist_ok=True)
    print("========== Starting feature extraction for videos in:", args.video_dir)
    extract_diversity_features(
        input_dir=args.video_dir,
        labels_path=args.labels_path,
        output_dir=diversity_feature_path,
        model_name=args.model_name
    )  
    
    main_select_data(
        clip_feature_path=diversity_feature_path,
        failure_score_path=failure_score_path,
        get_npz_name=get_npz_sfv,
        get_item_name=get_item_name_sfv,
        save_path_candidate=save_path_candidate,
        K=args.num_samples,
        scaling=scaling
    )

    srcc1, plcc1 = gMAD_srcc(
        args.labels_path,
        args.baseline_pred_path,
    )
 
    srcc2, plcc2 = gMAD_srcc(
        args.labels_path,
        args.baseline_pred_path,
        filter_list_path=save_path_candidate
    )
    
    print(f"========== Selected samples saved to: {save_path_candidate}")
    print(f"{args.dataset}: Performance on all training samples SRCC: {srcc1}, PLCC: {plcc1}")
    print(f"{args.dataset}: Performance on the selected samples SRCC: {srcc2}, PLCC: {plcc2}")
