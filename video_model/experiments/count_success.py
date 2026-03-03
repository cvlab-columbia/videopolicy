import os
import json
import csv

def find_first_json_file(folder_path):
    """Find the first .json file in the given folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            return os.path.join(folder_path, filename)
    return None

def count_successes_and_totals(folder_path):
    """Load the first .json file in the folder and calculate success rate."""
    json_file = find_first_json_file(folder_path)
    if not json_file:
        print("No .json file found in the folder.")
        return None, None, {}

    print(f"Using JSON file: {json_file}")
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Count successes and totals
    total_count = 0
    success_count = 0
    environment_success_counts = {}

    for env_name, env in data["environments"].items():
        env_success_count = 0
        env_total_count = 0

        for experiment in env["experiments"].values():
            if experiment["success"] >= 0:
                env_total_count += 1
                env_success_count += experiment["success"]

        environment_success_counts[env_name] = (env_success_count, env_total_count)
        total_count += env_total_count
        success_count += env_success_count

    return success_count, total_count, environment_success_counts

def update_csv(csv_file, experiment_name, success_rate, env_success_counts):
    """Update the CSV file with the experiment's success and total counts."""
    if not os.path.exists(csv_file):
        # Create the CSV file if it doesn't exist
        with open(csv_file, 'w', newline='') as file:
           pass

    # Read existing data
    rows = []
    updated = False

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Update or add the experiment's success and total counts
    for i, row in enumerate(rows):
        if row[0] == experiment_name:
            new_row = [experiment_name, f"{success_rate[0]} / {success_rate[1]}"] + [f"{env_counts[0]} / {env_counts[1]}" for env_counts in env_success_counts.values()]
            rows[i] = new_row
            header = ["", "Overall"] + [f"{env}" for env in env_success_counts.keys()]
            rows[i-1] = header
            updated = True
            break

    if not updated:
        new_row = [experiment_name, f"{success_rate[0]} / {success_rate[1]}"] + [f"{env_counts[0]} / {env_counts[1]}" for env_counts in env_success_counts.values()]
        header = ["", "Overall"] + [f"{env}" for env in env_success_counts.keys()]
        rows.append(header)
        rows.append(new_row)

    # Write back to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
        

folder_path = 'example_inference'

success_count, total_count, environment_success_counts = count_successes_and_totals(folder_path)

if success_count is not None and total_count is not None:
    success_rate = (success_count, total_count)
    print(f"Total number of successes: {success_count}")
    print(f"Total number of experiments: {total_count}")
    print(f"Success / Total: {success_count} / {total_count}")

    for env, counts in environment_success_counts.items():
        print(f"Environment: {env}, Success / Total: {counts[0]} / {counts[1]}")

    csv_file = f'experiment_success_rates.csv'
    experiment_name = os.path.basename(folder_path)
    update_csv(csv_file, experiment_name, success_rate, environment_success_counts)
    print(f"Updated success counts for '{experiment_name}' in '{csv_file}'.")
