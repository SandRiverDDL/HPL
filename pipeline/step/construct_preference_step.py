import os
import json
import logging
import argparse
import glob
import numpy as np


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('construct_preference_step.log')
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
    results_original_reward = {}
    sample_num = len(glob.glob(traj_path + "/*"))
    for i in range(sample_num):
        paths = glob.glob(f"mc_samples_iter_{args.iter}/mc_samples_{i}_iter_{args.iter}/7b/{args.task}/*.jsonl")
        cur_results = []
        for path in paths:
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            cur_results.extend(data)
        for item in cur_results:
            id = item['id']
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"
            if id_iteration not in results:
                results[id_iteration] = [item['agent_final_reward']]
            else:
                results[id_iteration].append(item['agent_final_reward'])
            if id_iteration not in results_original_reward:
                results_original_reward[id_iteration] = item['agent_step_reward']
            else:
                assert results_original_reward[id_iteration] == item['agent_step_reward']
    final_results = {}
    for key, value in results.items():
        final_results[key] = {
            "monte_carlo_reward": np.mean(value),
            "env_reward": results_original_reward[key]
        }
        
    output_data = []
    for file in glob.glob(f"{args.traj_path}/*.jsonl"):
        data = []
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        for item in data:
            id = item['id']
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"
            
            if id_iteration in final_results:
                item['monte_carlo_step_reward'] = final_results[id_iteration]['monte_carlo_reward']
            else:
                item['monte_carlo_step_reward'] = item['agent_step_reward']
            output_data.append(item)
                
    return output_data


def build_preference(args):
    win = 0
    tie = 0
    lose = 0
    global_traj = 0
    local_step_traj = 0
    local_entire_traj = 0
    
    golden_raw = json.load(open(f"expert_trajectories/{args.task}_sft_iter.json"))
    pm_data = []
    
    explored_traj = cal_monte_carlo_reward(args)
    
    step_monte_carlo_threshold = args.step_threshold
    
    for item in explored_traj:
        id = item['id']
        iteration = item['iteration']
        if iteration == 0:
            continue
        
        agent_conversations = template_change(item['agent_conversations'])
        agent_step_reward = 0 if item['monte_carlo_step_reward'] is None else item['monte_carlo_step_reward']
        agent_final_reward = item['agent_final_reward']

        if f"{id}_{iteration}" not in golden_raw:
            continue
        
        gpt_conversations = golden_raw[f"{id}_{iteration}"]['gpt_conversations']
        gpt_step_reward = golden_raw[f"{id}_{iteration}"]['monte_carlo_step_reward']
        gpt_final_reward = golden_raw[f"{id}_{iteration}"]['gpt_reward']
        gpt_step_conversations = golden_raw[f"{id}_{iteration}"]['gpt_step_conversations']
        
        gpt_length = len(gpt_step_conversations)
        agent_length = len(gpt_step_conversations)
        
        flag = False
        if args.task == "webshop" or args.task =="intercode_sql":
            flag = False
        elif args.task == "alfworld":
            if len(agent_conversations[agent_length-1: -1]) < len(gpt_conversations[gpt_length-1:]):
                flag = True
        
        if is_empty_conversations(agent_conversations) or is_empty_conversations(gpt_conversations):
            continue
        
        if (agent_final_reward > gpt_final_reward + 0.01 or flag) and agent_step_reward > gpt_step_reward + step_monte_carlo_threshold:
            win += 1
            local_entire_traj += 1
            pm_data.append({
                "id": f"{id}_{iteration}",
                "prompt": gpt_conversations[:gpt_length-1],
                "chosen": agent_conversations[agent_length-1: -1],
                "rejected": gpt_conversations[gpt_length-1:],
            })
        elif gpt_final_reward > agent_final_reward + 0.01 and gpt_step_reward > agent_step_reward + step_monte_carlo_threshold:
            lose += 1
            local_entire_traj += 1
            pm_data.append({
                "id": f"{id}_{iteration}",
                "prompt": gpt_conversations[:gpt_length-1],
                "chosen": gpt_conversations[gpt_length-1:],
                "rejected": agent_conversations[agent_length-1: -1],
            })
        else:
            tie += 1
    json.dump(pm_data, open(args.output_path, "w"), indent=4)  
    print(f"win: {win}, tie: {tie}, lose: {lose}")
    print(f"global_traj: {global_traj}, local_step_traj: {local_step_traj}, local_entire_traj: {local_entire_traj}")          

def main():
    parser = argparse.ArgumentParser("Construct trajectory preference dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="intercode_sql",
        help="task name",
    )
    parser.add_argument(
        "--golden_traj_path",
        type=str,
        default="expert_trajectories/intercode_sql_sft_iter.json",
        help="path of the golden trajectory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='intercode_sql_pref_data/7b/traj_preference_step_7b.json',
        help="output path of the trajectory preference dataset",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default="sampled_trajectories_1/7b/intercode_sql",
        help="task name",
    )
    parser.add_argument(
        "--step_action",
        action="store_true",
        help="use step action or entire trajectory",
    )
    parser.add_argument(
        "--global_traj",
        action="store_true",
        help="use global trajectory"
    )
    parser.add_argument(
        "--local_traj",
        action="store_true",
        help="use local trajectory"
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="mc_samples_iter_1"
    )
    parser.add_argument(
        "--step_threshold",
        type=float,
        default=0.3
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
