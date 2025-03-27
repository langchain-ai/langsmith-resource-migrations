import requests
from langsmith import Client
from typing import Dict, Literal
import json

class LangsmithMigrator:
    
    def __init__(self, old_api_key: str, new_api_key: str):
        self.old_headers = {"X-API-Key": old_api_key}
        self.new_headers = {"X-API-Key": new_api_key}
        self.base_url = "https://api.smith.langchain.com/api/v1"
        self.old_client = Client(api_key=old_api_key)
        self.new_client = Client(api_key=new_api_key)
        
    def migrate_dataset(
            self, 
            original_dataset_id: str, 
            check_if_already_exists=True, 
            migration_mode: Literal["EXAMPLES", "EXAMPLES_AND_EXPERIMENTS", "DATASET_ONLY"] = "EXAMPLES"
        ) -> str:
        """
        Migrate a dataset and all its examples from old to new instance.
        Returns the new dataset ID.
        """
        # Get original dataset
        response = requests.get(
            f"{self.base_url}/datasets/{original_dataset_id}",
            headers=self.old_headers
        )
        original_dataset = response.json()
        
        # Check if dataset already exists in new instance
        if check_if_already_exists:
            response = requests.get(
                f"{self.base_url}/datasets?name={original_dataset['name']}",
                headers=self.new_headers
            )
            if "detail" not in response.json():
                maybe_existing_datasets = response.json()
                if len(maybe_existing_datasets) > 1:
                    raise ValueError(f"Found multiple datasets with name {original_dataset['name']} in new instance")
                elif len(maybe_existing_datasets) == 1:
                    return maybe_existing_datasets[0]["id"]
        
        # Create new dataset
        create_dataset_payload = {
            "name": original_dataset["name"],
            "description": original_dataset["description"],
            "created_at": original_dataset["created_at"],
            "inputs_schema_definition": original_dataset["inputs_schema_definition"],
            "outputs_schema_definition": original_dataset["outputs_schema_definition"],
            "externally_managed": original_dataset["externally_managed"],
            "transformations": original_dataset["transformations"] if original_dataset["transformations"] else [],
            "data_type": original_dataset["data_type"],
        }
        create_response = requests.post(
            f"{self.base_url}/datasets",
            headers=self.new_headers,
            json=create_dataset_payload
        )
        new_dataset_id = create_response.json()['id']
        
        # Migrate examples, if requested
        if migration_mode == "EXAMPLES":
            self.migrate_dataset_examples(original_dataset_id, new_dataset_id)
        elif migration_mode == "EXAMPLES_AND_EXPERIMENTS":
            original_to_new_example_ids = self.migrate_dataset_examples(original_dataset_id, new_dataset_id)
            self.migrate_dataset_experiments(original_dataset_id, new_dataset_id, original_to_new_example_ids)
        elif migration_mode == "DATASET_ONLY":  
            pass
        
        return new_dataset_id
    
    def migrate_dataset_examples(self, original_dataset_id: str, new_dataset_id: str) -> Dict[str, str]:
        """
        Migrate all examples from old dataset to new dataset.
        Returns mapping of old example IDs to new example IDs.
        """
        # Get all examples from old dataset
        offset = 0
        last_request_size = 100
        original_examples = []
        while last_request_size == 100:
            response = requests.get(
                f"{self.base_url}/examples",
                params={"dataset": original_dataset_id, "offset": offset},
                headers=self.old_headers,
            )
            examples = response.json()
            last_request_size = len(examples)
            original_examples += examples
            offset = len(original_examples)
            
        # Create examples in new dataset
        new_examples_payload = [
            {
                "dataset_id": new_dataset_id,
                "inputs": example["inputs"],
                "outputs": example["outputs"],
                "metadata": example["metadata"],
                "created_at": example["created_at"],
                "split": example["metadata"].get("dataset_split", "base") if example["metadata"] else "base",
            }
            for example in original_examples
        ]
        response = requests.post(
            f"{self.base_url}/examples/bulk",
            headers=self.new_headers,
            json=new_examples_payload
        )
        new_examples = response.json()
        
        # Create ID mapping
        return {
            original_examples[i]["id"]: new_examples[i]["id"]
            for i in range(len(new_examples))
        }
    
    def migrate_dataset_experiments(self, original_dataset_id: str, new_dataset_id: str, original_to_new_example_ids: Dict[str, str]):
        """
        Migrate all experiments from old dataset to new dataset.
        """
        # Get all experiments from old dataset
        offset = 0
        last_request_size = 100
        experiments = []
        while last_request_size == 100:
            response = requests.get(
                f"{self.base_url}/sessions?reference_dataset={original_dataset_id}&offset={offset}",   
                headers=self.old_headers,
            )
            experiment_batch = response.json()
            last_request_size = len(experiment_batch)
            experiments += experiment_batch
            offset = len(experiments)  

        # Create experiments in new dataset
        original_to_new_experiment_ids = {}
        for experiment in experiments:
            create_tracer_payload = {
                "name": experiment["name"],
                "description": experiment["description"],
                "reference_dataset_id": new_dataset_id,
                "default_dataset_id": experiment["default_dataset_id"],
                "start_time": experiment["start_time"],
                "end_time": experiment["end_time"],
                "extra": experiment["extra"],
                "trace_tier": experiment.get("trace_tier"),
            }
            response = requests.post(
                f"{self.base_url}/sessions",
                headers=self.new_headers,
                json=create_tracer_payload
            )
            new_experiment_id = response.json()["id"]
            original_to_new_experiment_ids[experiment["id"]] = new_experiment_id   

        # Pull runs from old experiments and push to new experiments
        get_runs_payload = {
            "session": [experiment["id"] for experiment in experiments],
            "skip_pagination": False,
        }
        while True:
            get_runs_response = requests.post(
                f"{self.base_url}/runs/query",
                headers=self.old_headers,
                json=get_runs_payload
            )
            original_runs = get_runs_response.json()["runs"]
            new_runs_payload = {
                "post": [
                    {
                        "name": run["name"],
                        "inputs": run["inputs"],
                        "run_type": run["run_type"],
                        "start_time": run["start_time"],
                        "end_time": run["end_time"],
                        "extra": run["extra"],
                        "error": run.get("error"),
                        "serialized": run.get("serialized", {}),
                        "outputs": run["outputs"],
                        "parent_run_id": run.get("parent_run_id"),
                        "events": run.get("events", []),
                        "tags": run.get("tags", []),
                        "trace_id": run["trace_id"],
                        "id": run["id"],
                        "dotted_order": run["dotted_order"],
                        "session_id": original_to_new_experiment_ids[run["session_id"]],  # Map to new session ID
                        "session_name": run.get("session_name"),
                        "reference_example_id": original_to_new_example_ids.get(run.get("reference_example_id")),  # Map to new example ID if exists
                        "input_attachments": run.get("input_attachments", {}),
                        "output_attachments": run.get("output_attachments", {})
                    }
                    for run in original_runs
                ]
            }
            # Send the request to create runs
            requests.post(
                f"{self.base_url}/runs/batch",
                headers=self.new_headers,
                json=new_runs_payload
            )
            if get_runs_response.json()["cursors"]["next"] is None:
                break
            else:
                get_runs_payload["cursor"] = get_runs_response.json()["cursors"]["next"]

    def migrate_annotation_queue(self,
                                 old_annotation_queue_id: str,
                                 check_if_already_exists=True, 
                                 migration_mode: Literal["QUEUE_AND_DATASET", "QUEUE_ONLY"] = "QUEUE_AND_DATASET"
                                 ) -> str:
        """
        Migrate an annotation queue from old to new instance.
        """
        # Get original annotation queue
        response = requests.get(
            f"{self.base_url}/annotation-queues/{old_annotation_queue_id}",
            headers=self.old_headers
        )
        original_annotation_queue = response.json()
        
        # Check if annotation queue already exists in new instance
        if check_if_already_exists:
            response = requests.get(
                f"{self.base_url}/annotation_queues?name={original_annotation_queue['name']}",
                headers=self.new_headers
            )
            if "detail" not in response.json():
                maybe_existing_annotation_queues = response.json()
                if len(maybe_existing_annotation_queues) > 1:
                    raise ValueError(f"Found multiple annotation queues with name {original_annotation_queue['name']} in new instance")
                elif len(maybe_existing_annotation_queues) == 1:
                    return maybe_existing_annotation_queues[0]["id"]
            
        # Migrate dataset, if requested
        default_dataset = None
        if migration_mode == "QUEUE_AND_DATASET" and original_annotation_queue["default_dataset"] is not None:
            default_dataset = self.migrate_dataset(
                original_annotation_queue["default_dataset"],
                check_if_already_exists=True,
                migration_mode="EXAMPLES"
            )
        elif migration_mode == "QUEUE_ONLY":
            pass
        
        # Create new annotation queue
        create_annotation_queue_payload = {
            "name": original_annotation_queue["name"],
            "description": original_annotation_queue["description"],
            "created_at": original_annotation_queue["created_at"],
            "updated_at": original_annotation_queue["updated_at"],
            "default_dataset": default_dataset,
            "num_reviewers_per_item": original_annotation_queue["num_reviewers_per_item"],
            "enable_reservations": original_annotation_queue["enable_reservations"],
            "reservation_minutes": original_annotation_queue["reservation_minutes"],
            "rubric_items": original_annotation_queue["rubric_items"],
            "rubric_instructions": original_annotation_queue["rubric_instructions"],
            "session_ids": []
        }
        response = requests.post(
            f"{self.base_url}/annotation-queues",
            headers=self.new_headers,
            json=create_annotation_queue_payload
        )
        new_annotation_queue_id = response.json()["id"]
        return new_annotation_queue_id


    def migrate_project_rules(self, old_project_id: str, new_project_id: str) -> str:
        """
        Migrate all rules from a tracing project from old to new instance
        """
        # Get original rules
        response = requests.get(
            f"{self.base_url}/runs/rules?session_id={old_project_id}",
            headers=self.old_headers
        )
        old_rules = response.json()
        
        # Handle dataset migration if needed
        for old_rule in old_rules:
            # This should never have a dataset_id
            if old_rule["dataset_id"] is not None:
                continue

            # Get old dataset name, if it doesn't exist in new instance yet, create it
            add_to_dataset_id = None
            if old_rule["add_to_dataset_id"] is not None:
                add_to_dataset_id = self.migrate_dataset(
                    old_rule["add_to_dataset_id"], 
                    check_if_already_exists=True,
                    migration_mode="EXAMPLES"
                )
            
            # Get old annotation queue name, if it doesn't exist in new instance yet, create it
            add_to_annotation_queue_id = None
            if old_rule["add_to_annotation_queue_id"] is not None:
                add_to_annotation_queue_id = self.migrate_annotation_queue(
                    old_rule["add_to_annotation_queue_id"], 
                    check_if_already_exists=True,
                    migration_mode="QUEUE_AND_DATASET"
                )
        
            # Create new rule
            create_rule_payload = {
                "display_name": old_rule["display_name"],
                "session_id": new_project_id,
                "is_enabled": old_rule["is_enabled"],
                "dataset_id": None,
                "sampling_rate": old_rule["sampling_rate"],
                "filter": old_rule["filter"],
                "trace_filter": old_rule["trace_filter"],
                "tree_filter": old_rule["tree_filter"],
                "add_to_annotation_queue_id": add_to_annotation_queue_id,
                "add_to_dataset_id": add_to_dataset_id,
                "add_to_dataset_prefer_correction": old_rule["add_to_dataset_prefer_correction"],
                "use_corrections_dataset": old_rule["use_corrections_dataset"],
                "num_few_shot_examples": old_rule["num_few_shot_examples"],
                "extend_only": old_rule["extend_only"],
                "transient": old_rule["transient"],
                "backfill_from": old_rule["backfill_from"],
                "evaluators": old_rule["evaluators"],
                "code_evaluators": old_rule["code_evaluators"],
                "alerts": old_rule["alerts"],
                "webhooks": old_rule["webhooks"]
            }
            response = requests.post(
                f"{self.base_url}/runs/rules",
                headers=self.new_headers,
                json=create_rule_payload
            )

    def migrate_prompt(self, original_prompt_id: str) -> str:
        """
        Migrate a prompt from original instance to new instance.
        """
        prompt_object = self.old_client.pull_prompt_commit(
            original_prompt_id, include_model=True
        )
        self.new_client.push_prompt(prompt_identifier=original_prompt_id, object=prompt_object.manifest)

