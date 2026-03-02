import os
import json
import logging
import argparse
import glob
from typing import Any, Dict, List
import numpy as np

import tasks
import agents
import envs
import time

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('construct_preference_group.log')
    ]
)
logger = logging.getLogger(__name__)


def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item['role'] == "assistant":
            message['from'] = "gpt"
            message['value'] = item['content'].strip()
        else:
            message['from'] = "human"
            message['value'] = item['content'].strip()
        messages.append(message)
    return messages

def is_empty_conversations(conversation):
    for item in conversation:
        if item['value'].strip() == "":
            return True
    return False

def cal_monte_carlo_reward(args):
    traj_path = args.sample_path
    results = {}
    sample_num = len(glob.glob(traj_path + "/*"))
    for i in range(sample_num):
        paths = glob.glob(f"mc_samples_group_{i}_iter_{args.iter}/{args.task}/1.5b/semantic/*.jsonl")
        cur_results = []
        for path in paths:
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            cur_results.extend(data)
        for item in cur_results:
            group_id = item['group_id']
            if group_id not in results:
                results[group_id] = [item['agent_final_reward']]
            else:
                results[group_id].append(item['agent_final_reward'])
    final_results = {}
    for key, value in results.items():
        final_results[key] = {
            "monte_carlo_reward": np.mean(value),
        }
        
    output_data = []
    for file in glob.glob(f"{args.traj_path}/*.jsonl"):
        data = []
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        for item in data:
            group_id = item['group_id']
            
            if group_id in final_results:
                item['monte_carlo_step_reward'] = final_results[group_id]['monte_carlo_reward']
            else:
                item['monte_carlo_step_reward'] = item['agent_final_reward']
            output_data.append(item)
                
    return output_data


def build_preference(args):
    win = 0
    tie = 0
    lose = 0
    global_traj = 0
    local_step_traj = 0
    local_entire_traj = 0
    group_step_traj = 0

    sft_data = json.load(open(f"grouped_action_output/{args.task}_action_groups_semantic.json"))
    temp = {}
    for item in sft_data:
        temp[item['id']] = item
    sft_data = temp
    if args.task == "alfworld":
        new_sft_data = {}
        for value in sft_data.values():
            new_sft_data[value['original']['game_file']] = value
    else:
        new_sft_data = sft_data

    
    golden_raw = json.load(open(f"expert_trajectories/{args.task}_sft_iter.json"))
    pm_data = []
    
    explored_traj = cal_monte_carlo_reward(args)
    
    step_monte_carlo_threshold = args.step_threshold
    traj_threshold = args.traj_threshold

    expert_group_data = json.load(open(f"grouped_action_output/{args.task}_action_groups_semantic.json"))
    new_expert_group_data = {}
    for item in expert_group_data:
        new_expert_group_data[item["id"]] = item
    expert_group_data = new_expert_group_data
    
    for item in explored_traj:
        id = item['id']
        group_idx = item['group_idx']
        group_id = item['group_id']
        if args.task == "alfworld":
            game_file = item['game_file']
            cur_sft_data = new_sft_data[game_file]
        else:
            cur_sft_data = new_sft_data[id]

        # positive_group: List[str] = []
        # # get positive group actions
        # for i in range(action_groups_idx[group_idx][0], action_groups_idx[group_idx][1]+1):
        #     positive_group.append(env.parse_action(conversations[2*i+3]['value']))

        
        # agent_step_conversations = item['agent_step_conversations']
        agent_group_conversations = item['generated_group']
        agent_step_reward = item['monte_carlo_step_reward']
        agent_final_reward = item['agent_final_reward']

        iteration = expert_group_data[id]['action_groups_idx'][group_idx][1]
        group_start_iteration = expert_group_data[id]['action_groups_idx'][group_idx][0]

        if f"{id}_{iteration}" not in golden_raw:
            continue
        
        gpt_step_conversations = golden_raw[f"{id}_{iteration}"]['gpt_step_conversations']
        gpt_step_reward = golden_raw[f"{id}_{iteration}"]['monte_carlo_step_reward']
        gpt_final_reward = golden_raw[f"{id}_{iteration}"]['gpt_reward']
        
        gpt_step_conversation_length = len(gpt_step_conversations)
        agent_group_length = len(agent_group_conversations)
        
        # if is_empty_conversations(agent_group_conversations) or is_empty_conversations(gpt_step_conversations):
        #     continue

        # if item["actions"] == positive_group:
        #     continue
        
        if agent_final_reward >= gpt_final_reward and agent_step_reward > gpt_step_reward + 0.3 and agent_step_reward >= 0.8:
            win += 1
            # group_step_traj += 1
            # pm_data.append({
            #     "id": group_id,
            #     "prompt": gpt_step_conversations[:3+2*group_start_iteration],
            #     "chosen": template_change(agent_group_conversations),
            #     "rejected": gpt_step_conversations[3+2*group_start_iteration:],
            # })
        elif gpt_final_reward >= agent_final_reward and gpt_step_reward > agent_step_reward + 0.3 and gpt_step_reward >= 0.8:
            lose += 1
            group_step_traj += 1
            pm_data.append({
                "id": group_id,
                "prompt": gpt_step_conversations[:3+2*group_start_iteration],
                "chosen": gpt_step_conversations[3+2*group_start_iteration:],
                "rejected": template_change(agent_group_conversations),
                "group_length": agent_group_length // 2 + 1,
                "difficulty": gpt_step_reward - agent_step_reward,
            })
        else:
            tie += 1
            
    json.dump(pm_data, open(args.output_path, "w"), indent=4)  
    print(f"win: {win}, tie: {tie}, lose: {lose}")
    print(f"global_traj: {global_traj}, local_step_traj: {local_step_traj}, local_entire_traj: {local_entire_traj}, global_step_traj: {group_step_traj}")          


def main():
    parser = argparse.ArgumentParser("Construct trajectory preference dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="webshop",
        help="task name",
    )
    parser.add_argument(
        "--golden_traj_path",
        type=str,
        default="expert_trajectories/webshop_sft_iter.json",
        help="path of the golden trajectory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='webshop_pref_data/1.5b/traj_preference_1_group_1.5b_semantic.json',
        help="output path of the trajectory preference dataset",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default="mc_samples_group_0_iter_1/webshop/1.5b/semantic",
        help="task name",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="mc_samples_iter_1"
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
    )
    
    args = parser.parse_args()

    build_preference(args)


if __name__ == "__main__":
    main()
