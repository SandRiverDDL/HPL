import os

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
from copy import deepcopy

from webshop.web_agent_site.envs import WebAgentTextEnv
from glob import glob

import tasks
import agents
import envs
import time


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generation_trajectory.log')
    ]
)
logger = logging.getLogger(__name__)


prompt_for_environment = {
    "webshop": [{
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        }
    ],
    "alfworld": [{
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. take {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        }
    ],
    "intercode_sql":[
        {
            "from": "human",
            "value": "You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will interact with a MySQL Database system using SQL queries to answer a question.\nAt each turn, you should first provide your step-by-step thinking for solving the task. Your thought process should start with \"Thought: \", for example: Thought: I should write a SQL query that gets the average GNP and total population from nations whose government is US territory.\nAfter that, you have two options:\n1) Interact with a mysql programming environment and receive the corresponding output. Your code should start with \"Action: \" and should be surrounded with ```sql ``` tag, for example: Action: \\n```sql\\nSELECT AVG(GNP), SUM(population) FROM nations WHERE government = 'US Territory'\\n```\n2) Directly submit the result, for example: Action: submit.\nYou should use this format: \"Thought: your thought\\nAction: \\n```sql\\n<the mysql command>\\n```\". You will receive the corresponding output for your sql command.\nYour output should contain only one \"Action\" part.\nThe \"Action\" part should be executed with a mysql interpreter or propose an answer. Any natural language in it should be commented out.\nThe SQL query and submit parts can not appear in your output simutaneously."
        },
        {
            "from": "gpt",
            "value": "OK"
        }
    ]}


def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item['from'] == "gpt":
            message['role'] = "assistant"
            message['content'] = item['value']
        else:
            message['role'] = "user"
            message['content'] = item['value']
        messages.append(message)
    return messages


def construct_llm_data(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name
    if args.api_base is not None:
        agent_config['config']['api_base'] = args.api_base
    if args.api_key is not None:
        agent_config['config']['api_key'] = args.api_key
    if args.temperature is not None:
        agent_config['config']['temperature'] = args.temperature

    env_config = exp_config["env_config"]
    task_name = args.exp_config
    
    if 'env_class' in env_config and env_config['env_class'] == 'WebShopEnv':
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)

    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    # initialize all the tasks    
    task_config: Dict[str, Any] = exp_config["task"]
    
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    
    all_tasks, n_tasks = task_class.load_tasks(path=task_config.get("filepath", ""), split="train", 
                                               part_num=args.part_num, part_idx=args.part_idx)
    if task_name == "alfworld":
        sft_data = json.load(open(f"expert_trajectories/{task_name}_sft_30.json"))
    elif task_name == "intercode_sql":
        sft_data = json.load(open(f"expert_trajectories/{task_name}_sft_10.json"))
    else:
        sft_data = json.load(open(f"expert_trajectories/{task_name}_sft.json"))
    temp = {}
    for item in sft_data:
        temp[item['id']] = item
    sft_data = temp
    
    print("Start generating trajectories")

    if task_name == "alfworld":
        new_sft_data = {}
        for value in sft_data.values():
            new_sft_data[value['game_file']] = value
    
    output_file_path = f"sampled_trajectories_{args.iter}/{args.model}/{task_name}/{task_name}_traj_{args.part_idx}_of_{args.part_num}.jsonl"
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        pbar = tqdm(total=n_tasks)
        for task in all_tasks:
            env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
            
            if task_name == "webshop" or task_name == "intercode_sql":
                if task.task_id in sft_data:
                    cur_sft_data = sft_data[task.task_id]
                else:
                    pbar.update(1)
                    continue
            elif task_name == "alfworld":
                game_file = "data/alfworld/json_2.1.1/train/" + task.task_name
                if game_file in new_sft_data:
                    cur_sft_data = new_sft_data[game_file]
                else:
                    pbar.update(1)
                    continue
            
            conversations = cur_sft_data['conversations']
            iteration_nums = len(conversations)//2-1

            for j in range(iteration_nums):
                if args.global_only and j != 0:
                    continue
                start_time = time.time()

                conversations_until_current_group = conversations[2:2*j+3]

                observation, state = env.reset()
                env.state.history = template_change(prompt_for_environment[task_name] + [conversations_until_current_group[0]])
                for i in range(1, len(conversations_until_current_group), 2):
                    observation, state = env.step(conversations_until_current_group[i]['value'])

                cur_step_llm_output: str = agent(env.state.history)
                observation, state = env.step(cur_step_llm_output)
                cur_step_llm_reward = state.reward
                cur_step_conversation = deepcopy(state.history)
                
                while not state.finished:
                    try:
                        llm_output: str = agent(state.history)
                    except Exception as e:
                        state.success = False
                        state.finished = True
                        state.terminate_reason = "exceeding maximum input length"
                        break
                    # environment step
                    observation, state = env.step(llm_output)
                    # color the state in blue
                    if state.finished:
                        break

                end_time = time.time()
                
                new_item = {
                    "id": cur_sft_data['id'],
                    "iteration": j,
                    'agent_step_output': cur_step_llm_output,
                    'agent_step_reward': cur_step_llm_reward,
                    "agent_step_conversations": cur_step_conversation,
                    "agent_conversations": state.history,
                    'agent_final_reward': state.reward if (task_name == "webshop" or task_name=="intercode_sql") else state.success,
                    'time': end_time - start_time,
                }

                if task_name == "alfworld":
                    new_item['game_file'] = cur_sft_data['game_file']

                output_file.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            pbar.update(1)
            output_file.flush()
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="agent",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="Llama-2-7b-hf-webshop-sft-explore-0",
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_false",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--global_only",
        action="store_true",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        required=False,
        help="Agent base url. It will override the 'api_base' in agent_config.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        help="Agent api key. It will override the 'api_key' in agent_config.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.0,
        help="Temperature for the agent. Default is 0.0. It will override the 'temperature' in agent_config.",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1.5b",
    )
    
    args = parser.parse_args()
    construct_llm_data(args)