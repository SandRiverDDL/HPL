import json

def process_conversations_by_entropy(input_file_path, output_file_path, entropy_threshold=1.2):
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
        all_gpt_conversations = [
            conv for conv in item['conversations']
            if conv.get('from') == 'gpt'
        ]

        actions_to_process = all_gpt_conversations[1:]
        num_actions = len(actions_to_process)
        
        action_groups_idx = []
        if num_actions > 0:
            group_start_idx = 0
            for i in range(num_actions):
                current_action = actions_to_process[i]
                entropy = current_action.get('entropy', 0)

                if entropy > entropy_threshold:
                    action_groups_idx.append([group_start_idx, i])
                    group_start_idx = i + 1
            
            if group_start_idx < num_actions:
                action_groups_idx.append([group_start_idx, num_actions - 1])

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
    input_filename = "expert_trajectories/intercode_sql_sft_7b_with_entropy.json"
    output_filename_entropy = "grouped_action_output/intercode_sql_action_groups_entropy_7b.json"
    entropy_cut_threshold = 0.1

    process_conversations_by_entropy(input_filename, output_filename_entropy, entropy_cut_threshold)
