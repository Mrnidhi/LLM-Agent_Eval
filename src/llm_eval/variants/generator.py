import json
import os
import sys
from copy import deepcopy
from itertools import product
from typing import List

import yaml
from jsonschema import validate

from llm_eval.core.config import load_agent_configuration


def load_schema(schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_variants(variants_path: str):
    with open(variants_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_base_config(base_config_path: str):
    if base_config_path and os.path.isfile(base_config_path):
        with open(base_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "AgentConfiguration": {
            "agent_name": "rag_agent",
            "description": "RAG Agent configuration",
            "config_version": "1.2",
            "application_version": "1.0",
            "application_name": "rag_llmops_workshop",
            "deployment": {
                "model_name": "gpt-4o-mini",
                "model_version": "2024-08-01",
                "name": "gpt-4o-mini",
                "endpoint": "https://example.com/gpt-4o-mini",
                "openai_api_version": "2024-10-01-preview",
            },
            "model_parameters": {"temperature": 0.0, "seed": 42},
            "retrieval": {
                "parameters": {
                    "search_type": "hybrid",
                    "top_k": 5,
                    "index_name": "tech-stack-index-0",
                    "index_semantic_configuration_name": "vector-index-semantic-configuration",
                },
                "deployment": {
                    "model_name": "text-embedding-ada-002",
                    "name": "text-embedding-ada-002",
                    "endpoint": "https://example.com/text-embedding-ada-002",
                    "openai_api_version": "2024-10-01-preview",
                },
            },
            "intent_system_prompt": "Your task is to extract the user's intent by reformulating their latest question...",
            "chat_system_prompt": "You are a knowledgeable assistant specializing only in technology domain...",
            "human_template": "question: {input}",
        }
    }


def float_range(start, end, step):
    """Inclusive float range. Uses epsilon tolerance to avoid missing the end value due to float drift."""
    vals = []
    current = float(start)
    while current <= end + 1e-9:
        vals.append(round(current, 5))
        current += step
    return vals


def int_range(start, end, step):
    return list(range(start, end + 1, step))


def is_active(d):
    """Variants JSON uses mixed types for 'active' (bool, string, int) -- handle all of them."""
    if "active" not in d:
        return True
    return d["active"] in [True, "true", "True", 1, "1"]


def parse_param_info(param_info):
    if not is_active(param_info):
        return []

    if "set" in param_info and isinstance(param_info["set"], list):
        return param_info["set"]

    if "range" in param_info and "step" in param_info:
        rng = param_info["range"]
        if isinstance(rng, list) and len(rng) == 2:
            start, end = rng
            step = param_info["step"]
            if any(isinstance(x, float) for x in [start, end, step]):
                return float_range(float(start), float(end), float(step))
            return int_range(int(start), int(end), int(step))

    if "default" in param_info:
        return [param_info["default"]]

    return []


def list_of_dicts_to_param_dict(list_of_dicts):
    output = {}
    for d in list_of_dicts:
        if not is_active(d):
            continue
        if "name" in d:
            param_name = d["name"]
            param_def = {k: v for k, v in d.items() if k not in ["name", "active"]}
            output[param_name] = param_def
    return output


def build_combinations_for_section(section_variants):
    if not isinstance(section_variants, dict) or not section_variants:
        return [{}]

    param_names = []
    value_lists = []

    for param_name, param_info in section_variants.items():
        possible_vals = parse_param_info(param_info)
        if possible_vals:
            param_names.append(param_name)
            value_lists.append(possible_vals)

    if not param_names:
        return [{}]

    return [{name: combo[i] for i, name in enumerate(param_names)} for combo in product(*value_lists)]


def build_value_combinations(variants, key_name):
    if key_name not in variants:
        return [{}]

    data = variants[key_name]

    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        return build_combinations_for_section(list_of_dicts_to_param_dict(data))

    if isinstance(data, dict):
        return build_combinations_for_section(data)

    return [{}]


def set_prompt_variants(variants: dict, prompt_variants: dict):
    # The {context} placeholder is required -- it gets filled with retrieved documents at runtime
    prompts = [p["prompt"] + "  \n<context>\n{context}\n</context>" for p in prompt_variants["prompts"]]
    variants["chat_system_prompt"] = prompts
    return variants


def generate_variants(schema_yaml: str, agent_folder: str, agent_config_file: str,
                      max_variants: int, prompt_variants: dict, variants: dict, output_dir: str):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    schema_yaml = os.path.join(project_root, schema_yaml)
    output_dir = os.path.join(project_root, output_dir)

    variants = set_prompt_variants(variants, prompt_variants)

    schema = load_schema(schema_yaml)
    base_config = load_agent_configuration(agent_folder, agent_config_file)

    if "intent_system_prompt" in variants and isinstance(variants["intent_system_prompt"], str):
        base_config["AgentConfiguration"]["intent_system_prompt"] = variants["intent_system_prompt"]

    if "human_template" in variants and isinstance(variants["human_template"], str):
        base_config["AgentConfiguration"]["human_template"] = variants["human_template"]

    try:
        validate(instance=base_config, schema=schema)
    except Exception as e:
        sys.stderr.write(f"WARNING: base_config does not conform to schema: {e}\n")

    agent_conf = base_config["AgentConfiguration"]

    main_deployments = variants.get("deployment", [])
    if isinstance(main_deployments, list) and main_deployments:
        main_deployments = [d for d in main_deployments if is_active(d)]
        if not main_deployments:
            main_deployments = [agent_conf["deployment"]]
    else:
        main_deployments = [agent_conf["deployment"]]

    model_param_combinations = build_value_combinations(variants, "model_parameters")

    retrieval_section = variants.get("retrieval", {})
    if not isinstance(retrieval_section, dict):
        retrieval_section = {}

    retrieval_deployments = retrieval_section.get("deployment", [])
    if isinstance(retrieval_deployments, list) and retrieval_deployments:
        retrieval_deployments = [d for d in retrieval_deployments if is_active(d)]
        if not retrieval_deployments:
            retrieval_deployments = [agent_conf["retrieval"]["deployment"]]
    else:
        retrieval_deployments = [agent_conf["retrieval"]["deployment"]]

    retrieval_param_combinations = build_value_combinations(retrieval_section, "parameters")

    chat_prompts = variants.get("chat_system_prompt")
    if not chat_prompts or not isinstance(chat_prompts, list):
        chat_prompts = [agent_conf["chat_system_prompt"]]

    # Cartesian product of all dimensions; truncated to max_variants to cap total configs
    all_combinations = list(product(
        main_deployments, model_param_combinations,
        retrieval_deployments, retrieval_param_combinations, chat_prompts,
    ))[:max_variants]

    try:
        version_float = float(agent_conf.get("config_version", 1.0))
    except ValueError:
        version_float = 1.0

    os.makedirs(output_dir, exist_ok=True)

    valid_count = 0
    for idx, (dep, model_params, ret_dep, ret_params, chat_prompt) in enumerate(all_combinations, start=1):
        new_conf = deepcopy(base_config)
        new_agent_conf = new_conf["AgentConfiguration"]

        # deepcopy deployment dicts to avoid cross-variant mutation
        new_agent_conf["deployment"] = deepcopy(dep)
        new_agent_conf["model_parameters"].update(model_params)
        new_agent_conf["retrieval"]["deployment"] = deepcopy(ret_dep)
        new_agent_conf["retrieval"]["parameters"].update(ret_params)
        new_agent_conf["chat_system_prompt"] = chat_prompt
        new_agent_conf["config_version"] = f"{version_float + 0.1 * idx:.1f}"

        try:
            validate(instance=new_conf, schema=schema)
        except Exception as e:
            sys.stderr.write(f"Skipping invalid variant #{idx}: {e}\n")
            continue

        output_filename = os.path.join(output_dir, f"rag_agent_config_{idx}.yaml")
        with open(output_filename, "w", encoding="utf-8") as out_f:
            yaml.dump(new_conf, out_f, sort_keys=False)

        valid_count += 1

    print(f"Generated {valid_count} YAML variant files.")
