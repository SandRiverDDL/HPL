import json
import math

def process_conversations_by_group_count(input_file_path, output_file_path, num_groups=3):
    if num_groups <= 0:
        print("Error: num_groups should be a positive integer.")
        return

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: no file '{input_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: cannot parse JSON file '{input_file_path}'")
        return

    new_data = []
    for item in original_data:
        gpt_indices = [
            i for i, conv in enumerate(item['conversations'])
            if conv.get('from') == 'gpt'
        ]

        actions_original_indices = gpt_indices[1:]
        num_actions = len(actions_original_indices)

        action_groups_idx = []
        if num_actions > 0:
            group_size = (num_actions + num_groups - 1) // num_groups

            action_labels = list(range(num_actions))

            grouped_labels = [
                action_labels[i:i + group_size]
                for i in range(0, num_actions, group_size)
            ]

            action_groups_idx = [
                [group[0], group[-1]] for group in grouped_labels
            ]

        new_item = {
            "id": item["id"],
            "original": item,
            "action_groups_idx": action_groups_idx
        }
        new_data.append(new_item)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Success. Output to file '{output_file_path}'")


if __name__ == "__main__":
    input_filename = "expert_trajectories/intercode_sql_sft_10.json"
    output_filename = "grouped_action_output/intercode_sql_action_groups_fixed_groups_3.json"
    total_groups = 3

    process_conversations_by_group_count(input_filename, output_filename, total_groups)
