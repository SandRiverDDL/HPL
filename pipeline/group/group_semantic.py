import os
import re
import json
import asyncio
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from time import sleep
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


ALFWORLD_GROUP_TEMPLATE = """I need you to help me divide the trajectory of an agent's interaction with the environment into multiple action groups based on semantic relevance.

Below is an interaction trajectory, which contains the environment description received by the agent and the sequence of actions performed:
{trajectory}

Please divide the action sequence in this trajectory into multiple semantically related groups, each group represents a set of actions to complete a sub-goal or sub-task.
Please follow the following principles when dividing:
1. Actions in the same group should be semantically closely related and complete a clear subtask together
2. When the purpose of an action changes, it should be divided into a new group
3. For each group, briefly describe the common goal of the group of actions

Please use the following format to return the results:
<action_groups>
Group 1 (action index: 0-2): Find the target item
- Action 0: go to toiletpaperhanger 1
- Action 1: go to toilet 1
- Action 2: take toiletpaper 1 from toilet 1

Group 2 (action index: 3-4): Complete the main task
- Action 3: go to toiletpaperhanger 1
- Action 4: put toiletpaper 1 in/on toiletpaperhanger 1
</action_groups>
"""

WEBSHOP_GROUP_TEMPLATE = """I need you to divide a sequence of actions into groups based on semantic relevance.

A possible grouping example:

Group 1 (action index: 0-0): Initial search phase
- Action 0: search[size 5 patent-beige high heel]

Group 2 (action index: 1-1): Preliminary screening and click to view product details
- Action 1: click[b09gxnyjcd]

Group 3 (action index: 2-3): Specification confirmation and detailed screening stage
- Action 2: click[beige-almond toe-patent leather]
- Action 3: click[5]

Group 4 (action index: 4-4): Purchase decision stage
- Action 4: click[buy now]

Your output then should be in the following format:
[[0, 0], [1, 1], [2, 3], [4, 4]]

Below is the interaction trajectory:
{trajectory}

Please group the actions by their indices. Your response MUST be a valid JSON array of arrays of integers, where each inner array represents a group of action indices.

Follow these rules STRICTLY:
1. Each action must belong to exactly one group.
2. The indices must be contiguous and cover the entire range from 0 to {num_actions} - 1.
3. The final output MUST NOT contain any text, explanations, code blocks, or markdown formatting outside of the JSON array itself. It should be a raw JSON string.
4. The last number in the last group MUST be {num_actions} - 1.

Example for a trajectory with 5 actions (indices 0, 1, 2, 3, 4):
[[0, 1], [2, 3], [4, 4]]

Another valid example:
[[0, 0], [1, 2], [3, 4]]

Your output must be only the JSON, like this:
[[0, 1], [2, 3], [4, 4]]
"""

INTERCODE_SQL_GROUP_TEMPLATE = """I need you to divide a sequence of actions into groups based on semantic relevance.

A possible grouping example:

Group 1 (action index: 0-1): Task initialization and data structure exploration phase
- Action 0: SHOW TABLES
- Action 1: DESC university

Group 2 (action index: 2-2): Query construction and execution phase
- Action 2: SELECT Enrollment, Primary_conference FROM university ORDER BY Founded ASC LIMIT 1

Group 3 (action index: 3-3): Result confirmation and submission stage
- Action 3: submit

Your output then shoul be in the following format:
[[0, 1], [2, 2], [3, 3]]

Below is the interaction trajectory:
{trajectory}

Please group the actions by their indices. Your response MUST be a valid JSON array of arrays of integers, where each inner array represents a group of action indices.

Follow these rules STRICTLY:
1. Each action must belong to exactly one group.
2. The indices must be contiguous and cover the entire range from 0 to {num_actions} - 1.
3. The final output MUST NOT contain any text, explanations, code blocks, or markdown formatting outside of the JSON array itself. It should be a raw JSON string.
4. The last number in the last group MUST be {num_actions} - 1.

Example for a trajectory with 5 actions (indices 0, 1, 2, 3, 4):
[[0, 1], [2, 3], [4, 4]]

Another valid example:
[[0, 0], [1, 2], [3, 4]]

Your output must be only the JSON, like this:
[[0, 1], [2, 3], [4, 4]]
"""


class ActionGroupGenerator:
    def __init__(self, task_type="alfworld", base_url="", api_key=None, model_name="gpt-4o", temperature=0):
        self.task_type = task_type
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.template = self._get_template()
        self.temperature = temperature
    
    def _get_template(self):
        if self.task_type == "webshop":
            return WEBSHOP_GROUP_TEMPLATE
        elif self.task_type == "intercode_sql":
            return INTERCODE_SQL_GROUP_TEMPLATE
        else:
            return ALFWORLD_GROUP_TEMPLATE

    def _extract_trajectory(self, item):
        trajectory = ""
        
        if self.task_type == "alfworld":
            for conv in item["conversations"]:
                if conv["from"] == "human" and "Your task is to:" in conv["value"]:
                    trajectory += f"Task: {conv['value']}\n\n"
                    break
        else:
            trajectory += f"Task: {item['conversations'][2]['value']}\n\n"
        
        actions = []
        observations = []
        
        for i, conv in enumerate(item["conversations"]):
            if conv["from"] == "human" and i > 0 and "Observation:" in conv["value"]:
                observations.append(conv["value"])
            elif conv["from"] == "gpt" and "Action:" in conv["value"]:
                action_match = re.search(r'Action: (.*)', conv["value"], re.DOTALL)
                if action_match:
                    actions.append(action_match.group(1))
        
        trajectory += "Interaction Sequence:\n"
        for i, (action, observation) in enumerate(zip(actions, observations)):
            trajectory += f"Action {i}: {action}\n"
            trajectory += f"Observation: {observation}\n\n"
        
        return trajectory, len(actions)

    async def async_query_openai(self, query, item_id, sample_num=1):
        n = sample_num
        aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        try_times = 3
        for i in range(try_times):
            try:
                completion = await aclient.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=1024,
                    n=n,
                    temperature=self.temperature,
                )
                res = [completion.choices[i].message.content for i in range(n)]
                break
            except Exception as e:
                print(e)
                res = ["ERROR"]
                if i < try_times-1:
                    sleep(60)
        
        return {"res": res, "id": item_id}

    def parse_action_groups(self, response_text, num_actions):
        try:
            groups = json.loads(response_text)
            
            if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
                logging.warning(f"Response is not a list of list. Response: {response_text}")
                return response_text

            if num_actions == 0:
                return [] if not groups else response_text

            if not groups:
                logging.warning(f"Response is empty for {num_actions} actions. Response: {response_text}")
                return response_text

            all_indices = [idx for g in groups for idx in g]
            expected_indices = list(range(num_actions))
            
            if sorted(set(all_indices)) != expected_indices:
                logging.warning(f"Index wrong. Expected 0-{num_actions-1}. Got {sorted(set(all_indices))}. Response: {response_text}")

                if groups and groups[-1]:
                    groups[-1].pop()
                    if (num_actions - 1) not in groups[-1]:
                        groups[-1].append(num_actions - 1)
                    groups[-1].sort()
                elif num_actions > 0:
                     logging.error(f"Empty group, expected {num_actions} actions.")
                     return response_text
                
            return groups
        except json.JSONDecodeError:
            logging.warning(f"Error: cannot decode JSON, {response_text}")
            return response_text

    async def generate_action_groups(self, input_file, output_file, sample_num=1, batch_size=5):
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            logging.info(f"Loading {len(raw_data)} items from {input_file}.")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        existing_data = []
        done_ids = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        done_ids = {item['id'] for item in existing_data if 'id' in item}
                        logging.info(f"{len(done_ids)} items already done, will skip them.")
                    else:
                        logging.warning(f"{output_file} data is not a list. Starting over.")
                        existing_data = []
            except (json.JSONDecodeError, FileNotFoundError):
                logging.warning(f"Cannot read existing output file {output_file}. Starting over.")

        tasks_to_process = [item for item in raw_data if item.get("id") not in done_ids]
        if not tasks_to_process:
            logging.info("No new tasks to process. Exiting.")
            return
            
        logging.info(f"Will process {len(tasks_to_process)} items.")
        
        all_new_results = []
        for i in range(0, len(tasks_to_process), batch_size):
            batch = tasks_to_process[i:i+batch_size]
            
            queries_with_meta = []
            for item in batch:
                item_id = item.get("id")
                if not item_id:
                    continue
                
                trajectory, num_actions = self._extract_trajectory(item)
                if num_actions == 0:
                    all_new_results.append({"id": item_id, "original": item, "action_groups_idx": []})
                    continue
                
                query = self.template.format(trajectory=trajectory, num_actions=num_actions)
                queries_with_meta.append({'query': query, 'id': item_id, 'item': item, 'num_actions': num_actions})

            if not queries_with_meta:
                continue

            api_tasks = [self.async_query_openai(q['query'], q['id'], sample_num) for q in queries_with_meta]
            api_results = await tqdm.gather(*api_tasks)

            results_map = {res['id']: res['res'] for res in api_results}
            for q_meta in queries_with_meta:
                item_id = q_meta['id']
                if item_id in results_map:
                    response_text = results_map[item_id][0]
                    parsed_groups = self.parse_action_groups(response_text, q_meta['num_actions'])
                    
                    output_item = {
                        "id": item_id,
                        "original": q_meta['item'],
                        "action_groups_idx": parsed_groups
                    }
                    all_new_results.append(output_item)
        
        final_data = existing_data + all_new_results
        with open(output_file, "w", encoding='utf-8') as fw:
            json.dump(final_data, fw, ensure_ascii=False, indent=4)
        
        logging.info(f"Success. The number of data is {len(final_data)}.")

async def main(task_type, input_file, output_file, sample_num, base_url, api_key, model_name, temperature):
    generator = ActionGroupGenerator(task_type, base_url, api_key, model_name, temperature)
    await generator.generate_action_groups(
        input_file,
        output_file,
        sample_num
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True,
                        help="Task type: alfworld, intercode_sql, or webshop")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file path")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path")
    parser.add_argument("--sample_num", type=int, default=1,
                        help="Sample size")
    parser.add_argument("--base_url", type=str,
                        help="Base URL")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="Model name for action grouping")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature parameter")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        args.task_type,
        args.input_file,
        args.output_file,
        args.sample_num,
        args.base_url,
        args.api_key,
        args.model_name,
        args.temperature
    )) 