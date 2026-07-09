import argparse
import os
import uuid
import json
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple, Protocol
from datetime import datetime
from time import sleep
from browsergym.experiments import EnvArgs
import gymnasium as gym
import browsergym.core
from dotenv import load_dotenv
import browsergym.webarena
import browsergym.stwebagentbench
import warnings
import numpy as np
from st_bench_example import DemoAgentArgs, action_set
from stwebagentbench.utils.args import parse_arguments
from stwebagentbench.utils.data_collector import DataCollector
from collections import Counter
from playwright.sync_api import Route
import time
import re

# Suppress the specific warnings
warnings.filterwarnings("ignore", message="WARN: env.chat to get variables from other wrappers is deprecated")
warnings.filterwarnings("ignore", message="WARN: env.shape to get variables from other wrappers is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="beartype")
warnings.filterwarnings("ignore", category=UserWarning, message="Field .* has conflict with protected namespace .*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="WARN: The obs returned by the `reset()` method is not within the observation space.")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="WARN: env.page to get variables from other wrappers is deprecated")

__SLOW_MO = 1000 if "DISPLAY_BROWSER" in os.environ else None
__HEADLESS = False if "DISPLAY_BROWSER" in os.environ else True

STWEBAGENTBENCH = "STWebAgentBenchEnv"

def forced_timeout_mapping(action_str, timeout_ms=10000):
    code = action_set.to_python_code(action_str).strip()

    # If click/fill already has timeout, replace it
    if "timeout=" in code:
        return re.sub(r"timeout\s*=\s*\d+", f"timeout={timeout_ms}", code)

    # Otherwise append timeout as the last kwarg
    if code.endswith(")"):
        code = code[:-1] + f", timeout={timeout_ms})"
        code = code.replace("(,", "(")

    return code

class EvaluationFramework:
    def __init__(self, args):
        # self.agent = self.init_agent(args)
        load_dotenv()
        self.args = args
        self.SUPPORTED_ENVS = {STWEBAGENTBENCH: self.run_st_bench,
                               }

        self.run_id = str(uuid.uuid4())
        # Where collected trajectories are written. Override with TRAINING_DATA_DIR;
        # defaults to a local ./training_data next to this script.
        self.base_data_path = os.environ.get(
            'TRAINING_DATA_DIR',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data'))
        os.makedirs(self.base_data_path, exist_ok=True)
        self.data_collector = None
        
        #####################
        # Trajectory saving setup
        #####################
        # Add "safe" to filename if safe mode is enabled
        trajectories_filename = 'safe_trajectories_new.json' if args.safe else 'vanilla_trajectories_new.json'
        self.trajectories_file = os.path.join(self.base_data_path, trajectories_filename)
        self._ensure_trajectories_file()
        print(f"Trajectories file: {self.trajectories_file}")
        #####################

        self.env_args = EnvArgs(
            task_name=args.env_id,
            max_steps=40,
            headless=args.headless,
            viewport={"width": 1500, "height": 1280},
            slow_mo=args.slow_mo,
        )

    def init_data_collector(self, env_id, task_name, exp_i):
        self.data_collector = DataCollector(self.base_data_path, env_id, task_name, exp_i)
    
    #####################
    # Trajectory file management methods
    #####################
    def _ensure_trajectories_file(self):
        """Ensure trajectories.json exists with empty structure if it doesn't exist."""
        if not os.path.exists(self.trajectories_file):
            with open(self.trajectories_file, 'w') as f:
                json.dump({}, f, indent=2)
    
    def _load_trajectories(self) -> dict:
        """Load trajectories from JSON file."""
        try:
            if os.path.exists(self.trajectories_file):
                with open(self.trajectories_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load trajectories file: {e}. Starting fresh.")
        return {}
    
    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy arrays and other non-serializable types to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_trajectories(self, trajectories: dict):
        """Save trajectories to JSON file."""
        try:
            # Convert numpy arrays to lists before saving
            serializable_trajectories = self._convert_to_json_serializable(trajectories)
            with open(self.trajectories_file, 'w') as f:
                json.dump(serializable_trajectories, f, indent=2)
        except IOError as e:
            print(f"Error: Could not save trajectories file: {e}")
    
    def _get_next_trajectory_number(self, task_id: str) -> int:
        """Get the next trajectory number for a given task."""
        trajectories = self._load_trajectories()
        if task_id not in trajectories:
            return 0
        if "trajectories" not in trajectories[task_id]:
            return 0
        existing_numbers = [int(k) for k in trajectories[task_id]["trajectories"].keys() if k.isdigit()]
        if not existing_numbers:
            return 0
        return max(existing_numbers) + 1
    
    def _save_trajectory(self, task_id: str, trajectory_num: int, policy: list, 
                        meta_data: dict, trajectory_data: dict):
        """Save a single trajectory to the JSON file."""
        trajectories = self._load_trajectories()
        
        if task_id not in trajectories:
            trajectories[task_id] = {
                "policy": policy,
                "trajectories": {}
            }
        else:
            # Update policy if it exists (should be the same, but update for consistency)
            trajectories[task_id]["policy"] = policy
            if "trajectories" not in trajectories[task_id]:
                trajectories[task_id]["trajectories"] = {}
        
        trajectories[task_id]["trajectories"][str(trajectory_num)] = {
            "meta_data": meta_data,
            "data": trajectory_data
        }
        
        self._save_trajectories(trajectories)
        print(f"Saved trajectory {trajectory_num} for task {task_id}")
    #####################

    def load_exp_args(self, policies=None):
        self.agent = self.init_agent(args, policies)

    def init_agent(self, args, policies):
        print(f"\n{'='*60}")
        print(f"Initializing agent in collect_trajectories.py")
        print(f"Model name from args: {args.model_name}")
        print(f"{'='*60}\n")
        return DemoAgentArgs(model_name=args.model_name,safe=args.safe).make_agent()

    def eval(self):
        try:
            self.SUPPORTED_ENVS[self.args.env_id]()
        except Exception as e:
            import traceback
            if self.data_collector is not None:
                self.data_collector.record_failure(str(e), traceback.format_exc())
            print(f"Error: {str(e)}")
            # Print the traceback
            print(traceback.format_exc())
        finally:
            if self.data_collector is not None:
                self.data_collector.save_to_csv()
                self.data_collector.save_to_json()

    def setup_webarena(self):
        pass

    @staticmethod
    def get_next_experiment_number(base_path, env_id, task_name):
        exp_path = os.path.join(base_path, env_id, task_name)
        if not os.path.exists(exp_path):
            return 1
        existing_exps = [d for d in os.listdir(exp_path) if
                         d.startswith('exp_') and os.path.isdir(os.path.join(exp_path, d))]
        if not existing_exps:
            return 1
        return max([int(d.split('_')[1]) for d in existing_exps]) + 1

    def agent_loop(self, env, obs, info, max_steps, task=None):
        page = env.page

        print(f"[debug] starting agent_loop url={page.url} goal={obs.get('goal','')[:120]}")
        
        # Extract target values and locator from eval spec for debugging
        target_values = []
        target_locator = None
        target_url = None
        if task and hasattr(task, "config"):
            eval_config = task.config.get("eval", {})
            program_html = eval_config.get("program_html", [])
            if program_html and len(program_html) > 0:
                target_config = program_html[0]
                required_contents = target_config.get("required_contents", {})
                target_values = required_contents.get("must_include", [])
                target_locator = target_config.get("locator", "")
                target_url = target_config.get("url", "last")

        # Optional: Playwright listener counters
        perf = {
            "step": 0,
            "req_total": 0,
            "req_inflight": 0,
            "req_by_type": Counter(),
            "nav_main": 0,
        }

        def on_request(req):
            perf["req_total"] += 1
            perf["req_inflight"] += 1
            try:
                perf["req_by_type"][req.resource_type] += 1
            except Exception:
                pass

        def on_req_done(req):
            perf["req_inflight"] -= 1

        def on_frame_navigated(frame):
            try:
                if frame == page.main_frame:
                    perf["nav_main"] += 1
            except Exception:
                pass

        page.on("request", on_request)
        page.on("requestfinished", on_req_done)
        page.on("requestfailed", on_req_done)
        page.on("framenavigated", on_frame_navigated)

        def ready_state():
            try:
                return page.evaluate("document.readyState")
            except Exception:
                return "?"




        pointer_env = self.get_pointer_env(env)

        state = {
            "next": "",  # Initialize with an empty string or appropriate default
            "pages": [],  # Initialize with an empty list
            "page": page,
            "input": obs["goal"],
            "prediction": None,  # Initialize with None or create a default Prediction object
            "scratchpad": [],
            "observation": "",  # Initialize with an empty string
            "img": "",  # Initialize with an empty string or generate a base64 encoded screenshot
            "annotations": obs,
            "extension_obj": None,  # Initialize with None or self.extension_obj if available
            "actions": "",  # Initialize with an empty string
            "url": page.url,  # Get the current page URL
            "elements": "",  # Initialize with an empty string or fetch elements if possible
            "messages": [],  # Initialize with an empty list
            "env": env,  # Pass the environment object
            "pointer_env": pointer_env,
            "policy": "",
            "update_policy_reason": "First policy to be created",
            "read_page": "",  # Initialize with an empty string or fetch the outer text of the page
            "env_policies": obs.get("policies", ""),
        }

        # thread = {"configurable": {"thread_id": "1"}}

        loop_data = {
            'steps': [],
            'total_reward': 0,
            'terminated': False,
            'truncated': False,
            'agent_predictions': []  # New field to store agent predictions
        }

        done = False
        violated_policies = []
        # Track which policies have already been added to avoid duplicates
        # Key: (policy_template_id, validator_type) for form_validation, or just policy_template_id for others
        seen_violations = set()

        #####################
        # State-action pair collection
        #####################
        # Capture goal once at the beginning
        goal = obs.get('goal', '')
        trajectory_steps = {}  # Will store step_1, step_2, etc.
        step_counter = 1
        #####################

        # Collect agent predictions

        while not done:
            perf["step"] += 1

            new_obs = self.agent.obs_preprocessor(obs)
            action = self.agent.get_action(new_obs)
            
            #####################
            # Capture preprocessed observation and action
            # Remove goal and policies from state (saved once at trajectory level)
            #####################
            step_state = {k: v for k, v in new_obs.items() if k not in ['goal', 'policies']}
            trajectory_steps[f'step_{step_counter}'] = {
                'state': step_state,
                'action': action,
            }
            step_counter += 1
            #####################

            env.chat.add_message('assistant', action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            t_step1 = time.perf_counter()
            
            # Update trajectory step with reward and info after action execution
            if f'step_{step_counter - 1}' in trajectory_steps:
                trajectory_steps[f'step_{step_counter - 1}']['reward'] = reward
                trajectory_steps[f'step_{step_counter - 1}']['terminated'] = terminated
                trajectory_steps[f'step_{step_counter - 1}']['truncated'] = truncated
            
            if "safety_report" in info:
                for policy_report in info['safety_report']:
                    if policy_report['violated']:
                        # Create unique key for this violation
                        policy = policy_report.get('policy', {})
                        policy_id = policy.get('policy_template_id', 'unknown')
                        eval_config = policy.get('eval', {})
                        validator_type = eval_config.get('validator_type', '')
                        
                        # For form_validation, include validator_type to distinguish phone/email/state violations
                        if policy_id == 'is_form_validation' and validator_type:
                            violation_key = (policy_id, validator_type)
                        else:
                            violation_key = (policy_id,)
                        
                        # Only add if we haven't seen this violation before
                        if violation_key not in seen_violations:
                            seen_violations.add(violation_key)
                            violated_policies.append(policy_report)
            step_data = {
                'action': action,
                'observation': obs,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                # 'info': info
            }
            loop_data['steps'].append(step_data)
            loop_data['total_reward'] += reward
            loop_data['terminated'] = terminated
            loop_data['truncated'] = truncated
            loop_data['task_info_gt'] = info if not 'task_info' in loop_data else loop_data[
                                                                                      'task_info_gt'] + info
            # Terminate if: environment says so, reward > 0 (task succeeded), or finish() was called
            done = terminated or truncated or reward > 0 or action.startswith("finish(")
        
        #####################
        # Store trajectory data in loop_data for saving
        #####################
        loop_data['trajectory_steps'] = trajectory_steps
        loop_data['goal'] = goal  # Store goal once for the trajectory
        #####################

        print("Violated policies", violated_policies)
        success_under_policies = len(violated_policies) == 0 and reward == 1.0
        print("Success under policy", success_under_policies)
        loop_data['success_under_policies'] = success_under_policies
        loop_data['violated_policies'] = violated_policies

        if done:
            print("Environment terminated or truncated.\n")
            print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}\n")
            return loop_data

        print("Agent loop completed.\n")
        return loop_data

    def run_st_bench(self):
        # action_set = self.get_action_set()
        if self.args.specific_tasks_range:  # Specific tasks filtering
            spec = self.args.specific_tasks_range
            if ',' in spec:
                # Comma-separated list of actual task IDs
                requested_ids = set(int(x.strip()) for x in spec.split(','))
                tasks = [t for t in browsergym.stwebagentbench.ALL_ST_BENCH_TASK_IDS
                         if int(t.split('.')[-1]) in requested_ids]
            else:
                # Range format: "start-end" (indices into ALL_ST_BENCH_TASK_IDS)
                start, end = map(int, spec.split('-'))
                tasks = browsergym.stwebagentbench.ALL_ST_BENCH_TASK_IDS[start:end + 1]
            if not tasks:
                print("No tasks found for the specified range.")
                return
        else:
            tasks = browsergym.stwebagentbench.ALL_ST_BENCH_TASK_IDS

        reset_script_path = os.path.join(os.path.dirname(__file__), 'suitecrm_setup', 'reset.sh')
        num_runs_per_task = args.num_runs_per_task  # Number of times to run all tasks (each run includes all tasks, then reset)
        
        # Determine safe modes to run
        safe_modes = []
        if hasattr(self.args, 'both') and self.args.both:
            safe_modes = [True, False]  # Run with safe=True first, then safe=False
        else:
            safe_modes = [self.args.safe]  # Run with current safe setting
        
        all_total_rewards = []
        
        for safe_mode_idx, safe_mode in enumerate(safe_modes):
            print("")
            print("=" * 70)
            print(f"Running with safe={safe_mode}")
            print("=" * 70)
            
            # Update safe mode for this run
            original_safe = self.args.safe
            self.args.safe = safe_mode
            
            # Update trajectories filename based on safe mode
            trajectories_filename = 'safe_trajectories_new.json' if safe_mode else 'vanilla_trajectories_new.json'
            self.trajectories_file = os.path.join(self.base_data_path, trajectories_filename)
            self._ensure_trajectories_file()
            print(f"Trajectories file: {self.trajectories_file}")
            
            # Outer loop: number of times to run all tasks
            for run_idx in range(num_runs_per_task):
                print("")
                print("=" * 70)
                print(f"Run {run_idx + 1}/{num_runs_per_task} of all tasks (safe={safe_mode})")
                print("=" * 70)
                
                total_rewards = []
                
                # Inner loop: run all tasks
                for task_idx, task in enumerate(tasks):
                    print("")
                    print("=" * 50)
                    print(f"TASK: {task}")
                    print(f"Starting task {task_idx + 1}/{len(tasks)}: {task}")
                    print(f"Safe mode: {safe_mode}")
                    print(f"Run: {run_idx + 1}/{num_runs_per_task}")
                    print("=" * 50)
                    
                    env_id = self.args.env_id.split('.')[0]
                    exp_i = self.get_next_experiment_number(self.base_data_path, env_id, task)
                    self.init_data_collector(env_id, task, exp_i)

                    task_data = {
                        'task_name': str(task),
                        'run_number': run_idx + 1,
                        'safe_mode': safe_mode,
                        'start_time': datetime.now().isoformat()
                    }

                    env = gym.make(task,
                                  headless=False,
                                   action_mapping=forced_timeout_mapping,
                                   )

                    obs, info = env.reset()

                    # One of these will work depending on wrappers
                    task_obj = getattr(env.unwrapped, "task", None) or getattr(getattr(env.unwrapped, "env", None), "task", None)
                    if task_obj and hasattr(task_obj, "config"):
                        print("EVAL SPEC:", json.dumps(task_obj.config.get("eval", {}), indent=2))
                        print("INTENT:", task_obj.config.get("intent"))

                    # Handle special policies provided by the environment for the task
                    policies = obs['policies'] if 'policies' in obs else ''

                    ###### Initialize the agent #####
                    self.load_exp_args(policies)

                    task_data['initial_observation'] = obs

                    # Cheat functions use Playwright to automatically solve the task
                    env.chat.add_message(role="assistant", msg="On it. Please wait...")

                    loop_data = self.agent_loop(env, obs, info, self.args.max_steps, task=task_obj)

                    task_data.update(loop_data)

                    reward = loop_data['total_reward']
                    success_under_policies = loop_data.get('success_under_policies', False)
                    trajectory_length = len(loop_data.get('trajectory_steps', {}))

                    task_data.update({
                        'end_time': datetime.now().isoformat()
                    })
                    self.data_collector.collect_data(task_data)
                    # self.data_collector.save_checkpoint()
                    self.data_collector.save_to_csv()
                    self.data_collector.save_to_json()

                    #####################
                    # Save trajectory to JSON file
                    #####################
                    task_id = str(task)
                    trajectory_num = self._get_next_trajectory_number(task_id)
                    
                    # Format policies for storage
                    policy_list = []
                    if isinstance(policies, list):
                        for p in policies:
                            if isinstance(p, dict):
                                policy_list.append(p.get('description', p.get('policy_template', str(p))))
                            else:
                                policy_list.append(str(p))
                    elif policies:
                        policy_list = [str(policies)]
                    
                    goal = loop_data.get('goal', task_data.get('initial_observation', {}).get('goal', ''))
                    violated_policies_list = loop_data.get('violated_policies', [])
                    
                    # Extract task template ID from task_obj config
                    task_template_id = None
                    if task_obj and hasattr(task_obj, "config"):
                        task_template_id = task_obj.config.get("intent_template_id")
                    
                    meta_data = {
                        'goal': goal,
                        'reward': reward,
                        'completion_under_policy': success_under_policies,
                        'length': trajectory_length,
                        'safe_mode': safe_mode,
                        'violated_policies': violated_policies_list,
                        'task_template_id': task_template_id,
                    }
                    
                    trajectory_data = loop_data.get('trajectory_steps', {})
                    
                    self._save_trajectory(
                        task_id=task_id,
                        trajectory_num=trajectory_num,
                        policy=policy_list,
                        meta_data=meta_data,
                        trajectory_data=trajectory_data
                    )
                    #####################

                    total_rewards.append(reward)

                    sleep(3)
                    env.close()
                    
                    print(f"Completed task {task_idx + 1}/{len(tasks)}: {task} (Reward: {reward}, Safe: {safe_mode})")
                
                # # Reset SuiteCRM after running all tasks in this run
                # print("")
                # print("=" * 50)
                # print(f"Completed all {len(tasks)} tasks in run {run_idx + 1}/{num_runs_per_task}")
                # print(f"Resetting SuiteCRM environment...")
                # print("=" * 50)
                # try:
                #     subprocess.run(['bash', reset_script_path], check=True)
                #     print("Reset completed. Waiting 60 seconds...")
                #     sleep(60)  # Wait a minute after reset
                #     print("Ready for next run.")
                # except subprocess.CalledProcessError as e:
                #     print(f"Warning: Reset script failed: {e}")
                #     print("Continuing anyway...")
                # except Exception as e:
                #     print(f"Warning: Error running reset script: {e}")
                #     print("Continuing anyway...")
                
                # Statistics for this run
                print("")
                print("=" * 70)
                print(f"Results for run {run_idx + 1}/{num_runs_per_task} (safe={safe_mode}):")
                print(f"  Total rewards: {sum(total_rewards)}")
                print(f"  Average reward: {sum(total_rewards) / len(total_rewards) if total_rewards else 0:.3f}")
                print("=" * 70)
                
                all_total_rewards.extend(total_rewards)
            
            # Restore original safe setting
            self.args.safe = original_safe
            
            # Statistics for this safe mode
            print("")
            print("=" * 70)
            print(f"Overall results for safe={safe_mode}:")
            print(f"  Total runs: {num_runs_per_task}")
            print(f"  Total tasks completed: {num_runs_per_task * len(tasks)}")
            print("=" * 70)

        # Overall statistics
        print("")
        print("=" * 70)
        print("Overall Statistics:")
        print(f"  Total rewards: {sum(all_total_rewards)}")
        print(f"  Average reward: {sum(all_total_rewards) / len(all_total_rewards) if all_total_rewards else 0:.3f}")
        print(f"  Total tasks completed: {len(all_total_rewards)}")
        print("=" * 70)

    # def get_action_set(self):
    #     send_message_to_user: callable = None
    #
    #     def answer(message):  # Need to verify that its suitable to Ido and Sammi conventions
    #         """
    #         When the task is done, this function should be called
    #         Examples:
    #             answer("I finished the task.")
    #             answer("I finished the task, the answer is 'value'")
    #         """
    #         send_message_to_user(message)
    #
    #     action_set = HighLevelActionSet(custom_actions=[answer], subsets=["bid", "chat", 'custom'], strict=False,
    #                                     multiaction=True, demo_mode='off').to_python_code
    #     return action_set

    @staticmethod
    def get_pointer_env(env):
        # For every task except WorkArena tasks env has a wrapper object env.env.env
        if hasattr(env, 'spec'):
            if env.spec.id.split('.')[0] in [STWEBAGENTBENCH]:
                pointer_env = env.env.env
            else:
                pointer_env = env
        else:
            pointer_env = env

        return pointer_env


def main_sync(args):
    eval_framework = EvaluationFramework(args)
    print("Starting evaluation...")
    eval_framework.eval()
    print("Evaluation completed.")


if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run the agent')
    args = parse_arguments(parser)
    args.env_id = STWEBAGENTBENCH
    args.specific_tasks_range = "30006,30052,30053,30007,30181,30183"
    args.model_name = "gpt-5.2"
    args.safe = True
    args.both = False
    args.num_runs_per_task = 1
    #args.model_name = "/home/fodl/tomerslor/safe-control/Llama-3.2-1B-Instruct"
    main_sync(args)