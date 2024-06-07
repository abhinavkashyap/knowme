from pathlib import Path


def update_env_file(new_env_vars):
    # This is the path that is going to exist
    current_file = Path(__file__)
    two_levels_up = current_file.parents[2]

    env_file_path = f"{two_levels_up}/.env"
    env_file_path = Path(env_file_path)

    if env_file_path.exists():
        with open(env_file_path, "r") as f:
            existing_vars = f.readlines()
    else:
        existing_vars = []

    # Convert existing variables to a dictionary
    existing_vars_dict = {}
    for line in existing_vars:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=", 1)
            existing_vars_dict[key] = value

    # Update existing variables with new values
    existing_vars_dict.update(new_env_vars)

    # Write the updated variables back to the .env file
    with env_file_path.open("w") as f:
        for key, value in existing_vars_dict.items():
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    update_env_file({"a": "b"})
