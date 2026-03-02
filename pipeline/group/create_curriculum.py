import json
import os
from collections import defaultdict

def create_dual_curriculum_data(all_data):
    """
    Applies a dual-layer curriculum learning strategy to the input data.
    The data is bucketed based on group length (complexity) and reward
    difference (difficulty), and then assembled into three training phases.

    Args:
        all_data (list): A list of dictionaries, where each dictionary
                         represents a preference pair and contains 'group_length'
                         and 'difficulty' keys.

    Returns:
        dict: A dictionary containing the data for each curriculum phase
              and a detailed breakdown of the buckets.
    """
    # --- 1. Define curriculum learning threshold ---
    # Absolute Task Length
    # L1: Short (<=3), L2: Medium (4-6), L3: Long (>6)
    LENGTH_BUCKETS = {1: 3, 2: 6}

    # Reward Gap
    # D1: Easy (>=0.7), D2: Medium (0.4-0.7), D3: Hard (<0.4)
    DIFFICULTY_THRESHOLDS = {1: 0.7, 2: 0.4}

    # --- 2. Data preparation and bucketing ---
    buckets = defaultdict(lambda: defaultdict(list))

    for item in all_data:
        length = item.get("group_length", 0)
        difficulty_score = item.get("difficulty", 0.0)

        if length <= LENGTH_BUCKETS[1]:
            len_bucket_idx = 1
        elif length <= LENGTH_BUCKETS[2]:
            len_bucket_idx = 2
        else:
            len_bucket_idx = 3

        if difficulty_score >= DIFFICULTY_THRESHOLDS[1]:
            diff_bucket_idx = 1
        elif difficulty_score >= DIFFICULTY_THRESHOLDS[2]:
            diff_bucket_idx = 2
        else:
            diff_bucket_idx = 3
        
        buckets[f'L{len_bucket_idx}'][f'D{diff_bucket_idx}'].append(item)

    # --- 3. Construct curriculum phases ---

    # Phase 1: Master the Basics
    phase_1_data = buckets['L1']['D1']

    # Phase 2: Expand Abilities
    phase_2_data = (
        buckets['L1']['D1'] + 
        buckets['L1']['D2'] + 
        buckets['L2']['D1']
    )

    # Phase 3: Full Complexity
    phase_3_data = all_data

    # --- 4. Format output ---
    final_buckets = {k: dict(v) for k, v in buckets.items()}
    
    curriculum_datasets = {
        "phase_1": phase_1_data,
        "phase_2": phase_2_data,
        "phase_3": phase_3_data,
        "details": {
            "total_samples": len(all_data),
            "phase_1_samples": len(phase_1_data),
            "phase_2_samples": len(phase_2_data),
            "phase_3_samples": len(phase_3_data),
            "buckets": {
                f"{len_key}_{diff_key}": len(data)
                for len_key, diff_buckets in final_buckets.items()
                for diff_key, data in diff_buckets.items()
            }
        }
    }
    
    return curriculum_datasets


if __name__ == "__main__":
    
    with open("traj_preference_1_group_7b_entropy.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)

    curriculum = create_dual_curriculum_data(all_data)

    print("--- Dual-Layer Curriculum Data Breakdown ---")
    print(json.dumps(curriculum['details'], indent=4))

    output_dir = "curriculum_data/7b_entropy"
    os.makedirs(output_dir, exist_ok=True)

    for phase_name, data in curriculum.items():
        if phase_name != "details":
            file_path = os.path.join(output_dir, f"{phase_name}_data.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Saved {len(data)} samples to {file_path}")