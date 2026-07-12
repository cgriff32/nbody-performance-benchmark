import os

workspace_dir = "/home/chrisgriffiths/agent_workspace/performance_test"

# Files to remove
files_to_remove = [
    "nbody",
    "output.csv",
    "time_output.txt",
    "simulation_time.txt",
    "libenergy.so"
]

cleaned_count = 0

for root, dirs, files in os.walk(workspace_dir):
    # Skip .git folders
    if ".git" in root:
        continue
        
    for file in files:
        if file in files_to_remove:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Removed: {os.path.relpath(file_path, workspace_dir)}")
                cleaned_count += 1
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

print(f"Cleanup complete! Removed {cleaned_count} binary and log files.")
