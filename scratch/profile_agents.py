import os
import json
import re
from datetime import datetime

brain_dir = "/home/chrisgriffiths/.gemini/antigravity-cli/brain"
results = []

# Regex to match target directory patterns
dir_pattern = re.compile(r"target directory is:\s+\S+/performance_test/(\S+)", re.IGNORECASE)
iter_pattern = re.compile(r"iter_[a-zA-Z0-9_]+")

for item in os.listdir(brain_dir):
    path = os.path.join(brain_dir, item)
    if not os.path.isdir(path):
        continue
        
    log_path = os.path.join(path, ".system_generated/logs/transcript.jsonl")
    if not os.path.exists(log_path):
        continue
        
    try:
        with open(log_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if not lines:
            continue
            
        # Parse first valid JSON line
        first_step = None
        for line in lines:
            try:
                first_step = json.loads(line)
                break
            except Exception:
                continue
                
        if not first_step:
            continue
            
        first_content = first_step.get("content", "")
        
        # Parse the assigned target directory
        dir_match = dir_pattern.search(first_content)
        if dir_match:
            target_iter = dir_match.group(1).strip(".\r\n\t ")
        else:
            # Fallbacks
            found_iters = iter_pattern.findall(first_content)
            if found_iters:
                target_iter = found_iters[0]
            else:
                target_iter = "parent_or_unknown"
                
        # Override specific known conversations
        if item == "d1dab082-6b30-4976-b98c-c9bed8709380":
            target_iter = "parent_orchestrator"
        elif item == "8b6a7690-dfe9-4ccc-83d3-e82274776afc":
            target_iter = "iter_1_100k"
        elif item == "43c1bd2a-fa52-46ea-83bb-756f4b6ea5b2":
            target_iter = "iter_2_100k"
        elif item == "e0b76acf-2fe4-4fc6-b80d-001dfe6337ee":
            target_iter = "iter_3_100k"
        elif item == "9cb8b7f6-e860-4c3d-b058-129bd389a961":
            target_iter = "iter_blank_1_100k"
        elif item == "c26187a1-db39-4f88-9932-c0868cf20af2":
            target_iter = "iter_blank_1"
        elif item == "f67d0471-63f2-4d5d-ae83-2afde5133750":
            target_iter = "iter_4_100k"
        elif item == "cc9e77a3-b5be-4432-904a-ff79acf94679":
            target_iter = "iter_5_100k"
        elif item == "10c68675-1223-4588-b777-3a24ff9d82cd":
            target_iter = "iter_7"
        elif item == "5deba1d3-4466-444b-b0a6-b1d00f607537":
            target_iter = "iter_6_100k"
            
        # Calculate stats
        valid_steps = []
        for line in lines:
            try:
                valid_steps.append(json.loads(line))
            except Exception:
                continue
                
        if not valid_steps:
            continue
            
        total_steps = len(valid_steps)
        first_time = datetime.strptime(valid_steps[0]["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        last_time = datetime.strptime(valid_steps[-1]["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        
        duration_seconds = int((last_time - first_time).total_seconds())
        
        input_chars = 0
        output_chars = 0
        tool_call_count = 0
        
        for step in valid_steps:
            source = step.get("source", "")
            content = step.get("content", "") or ""
            thinking = step.get("thinking", "") or ""
            tool_calls = step.get("tool_calls", []) or []
            
            if source == "MODEL":
                output_chars += len(content) + len(thinking)
                if tool_calls:
                    tool_call_count += len(tool_calls)
                    output_chars += len(json.dumps(tool_calls))
            else:
                input_chars += len(content)
                
        # Estimate tokens (average 3.8 characters per token for source code/prompts)
        est_input_tokens = int(input_chars / 3.8)
        est_output_tokens = int(output_chars / 3.8)
        
        results.append({
            "conv_id": item,
            "iteration": target_iter,
            "steps": total_steps,
            "tool_calls": tool_call_count,
            "duration_s": duration_seconds,
            "input_tokens": est_input_tokens,
            "output_tokens": est_output_tokens,
            "total_tokens": est_input_tokens + est_output_tokens
        })
    except Exception as e:
        print(f"Error parsing folder {item}: {e}")

# Filter out older testing runs that don't belong to the project
results = [r for r in results if r["iteration"] != "parent_or_unknown"]

# Sort results: iterations first, then parent orchestrator
def sort_key(x):
    iter_name = x["iteration"]
    if iter_name == "parent_orchestrator":
        return "zzzzz"
    return iter_name

results.sort(key=sort_key)

# Generate Markdown Report
report = []
report.append("# Post-Hoc Agent Execution Profiling Report\n")
report.append("This report profiles the execution time, step complexity, tool usage, and estimated token consumption for each of the AI optimization sub-agents spawned during this project.\n")
report.append("| Iteration | Conversation ID | Runtime | Steps | Tool Calls | Est. Input Tokens | Est. Output Tokens | Total Tokens |")
report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for r in results:
    # Format duration
    mins = r["duration_s"] // 60
    secs = r["duration_s"] % 60
    duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    
    report.append(f"| **`{r['iteration']}`** | `{r['conv_id'][:8]}...` | {duration_str} | {r['steps']} | {r['tool_calls']} | {r['input_tokens']:,} | {r['output_tokens']:,} | {r['total_tokens']:,} |")

report_content = "\n".join(report)

# Write report to workspace
with open("/home/chrisgriffiths/agent_workspace/performance_test/agent_profiling_report.md", "w") as f:
    f.write(report_content)

print("Agent profiling report written successfully!")
