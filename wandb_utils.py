def create_run_name(task_dict: dict) -> str:
    run_name_content = task_dict["run_name_content"]

    if not run_name_content:
        return "Default name"
    
    run_name_elements = []
    for parameter_name in run_name_content:
        run_name_elements.append(f"{parameter_name.replace('_', '-')}={task_dict[parameter_name]}")

    return run_name_elements.join("_")
