import os

workspace_dir = "/home/chrisgriffiths/agent_workspace/performance_test"
brain_dir = "/home/chrisgriffiths/.gemini/antigravity-cli/brain/d1dab082-6b30-4976-b98c-c9bed8709380"

abs_pattern_url = "file:///home/chrisgriffiths/agent_workspace/performance_test/"
abs_pattern_path = "/home/chrisgriffiths/agent_workspace/performance_test/"

# Walk recursively
for root, dirs, files in os.walk(workspace_dir):
    # Skip .git folders
    if ".git" in root:
        continue
        
    for file in files:
        if file.endswith(".md"):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
                
            if abs_pattern_url in content or abs_pattern_path in content:
                content = content.replace(abs_pattern_url, "")
                content = content.replace(abs_pattern_path, "")
                with open(path, 'w') as f:
                    f.write(content)
                print(f"Recursively converted: {os.path.relpath(path, workspace_dir)}")
                
                # If it's one of the top level files, sync to brain dir
                if file in ["README.md", "benchmark_summary.md", "benchmark_summary_100k.md", "agent_profiling_report.md"] and root == workspace_dir:
                    brain_path = os.path.join(brain_dir, file)
                    if os.path.exists(brain_dir):
                        with open(brain_path, 'w') as f:
                            f.write(content)
                        print(f"Synced {file} to system brain directory.")
