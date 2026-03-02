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
        logging.FileHandler('construct_preference_traj.log')
    ]
)
logger = logging.getLogger(__name__)

instruction_part = {
    "webshop": [
        {
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
    "alfworld": [
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
    "intercode_sql": [
        {
            "from": "human",
            "value": "You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will interact with a MySQL Database system using SQL queries to answer a question.\nAt each turn, you should first provide your step-by-step thinking for solving the task. Your thought process should start with \"Thought: \", for example: Thought: I should write a SQL query that gets the average GNP and total population from nations whose government is US territory.\nAfter that, you have two options:\n1) Interact with a mysql programming environment and receive the corresponding output. Your code should start with \"Action: \" and should be surrounded with ```sql ``` tag, for example: Action: \\n```sql\\nSELECT AVG(GNP), SUM(population) FROM nations WHERE government = 'US Territory'\\n```\n2) Directly submit the result, for example: Action: submit.\nYou should use this format: \"Thought: your thought\\nAction: \\n```sql\\n<the mysql command>\\n```\". You will receive the corresponding output for your sql command.\nYour output should contain only one \"Action\" part.\nThe \"Action\" part should be executed with a mysql interpreter or propose an answer. Any natural language in it should be commented out.\nThe SQL query and submit parts can not appear in your output simutaneously."
        },
        {
            "from": "gpt",
            "value": "OK"
        }
    ]
}


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
    
    traj_threshold = args.traj_threshold
    
    for item in explored_traj:
        id = item["id"]
        iteration = item['iteration']
        if iteration != 0:
            continue
        agent_final_reward = item['agent_final_reward']
        gpt_reward = golden_raw[f"{id}_0"]["gpt_reward"]
        gpt_conversations = golden_raw[f"{id}_0"]['gpt_conversations']
        agent_conversations = template_change(item['agent_conversations'])
        if is_empty_conversations(agent_conversations) or is_empty_conversations(gpt_conversations):
            continue
        if agent_final_reward > gpt_reward + traj_threshold:
            win += 1
            global_traj += 1
            pm_data.append({
                "id": f"{id}_{iteration}",
                "prompt": gpt_conversations[:len(instruction_part[args.task])+1],
                "chosen": agent_conversations[len(instruction_part[args.task])+1: -1],
                "rejected": gpt_conversations[len(instruction_part[args.task])+1:],
            })
        elif gpt_reward > agent_final_reward + traj_threshold:
            lose += 1
            global_traj += 1
            pm_data.append({
                "id": f"{id}_{iteration}",
                "prompt": gpt_conversations[:len(instruction_part[args.task])+1],
                "chosen": gpt_conversations[len(instruction_part[args.task])+1:],
                "rejected": agent_conversations[len(instruction_part[args.task])+1: -1],
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
        default="expert_trajectories/alfworld_sft_iter.json",
        help="path of the golden trajectory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='intercode_sql_pref_data/7b/traj_preference_traj_7b.json',
        help="output path of the trajectory preference dataset",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default="sampled_trajectories_1/7b/intercode_sql",
        help="task name",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="mc_samples_iter_1"
    )
    parser.add_argument(
        "--traj_threshold",
        type=float,
        default=0.01
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
