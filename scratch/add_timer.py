import os

targets = [
    "baseline/nbody.cpp",
    "iter_1/nbody.cpp",
    "iter_2/nbody.cpp",
    "iter_3/nbody.cpp",
    "iter_4/nbody.cpp",
    "iter_5/nbody.cpp",
    "iter_variant_1/nbody.cpp",
    "iter_variant_2/nbody.cpp",
    "iter_variant_3/nbody.cpp"
]

base_dir = "/home/chrisgriffiths/agent_workspace/performance_test"

for target in targets:
    path = os.path.join(base_dir, target)
    if not os.path.exists(path):
        print(f"Skipping {target}: does not exist.")
        continue
        
    with open(path, 'r') as f:
        content = f.read()
        
    # Check if already has simulation_time.txt
    if "simulation_time.txt" in content:
        print(f"Skipping {target}: already instrumented.")
        continue
        
    if target == "iter_variant_1/nbody.cpp":
        # Pure C style
        target_str = 'printf("Simulation loop time: %f seconds\\n", elapsed);'
        replacement_str = target_str + '\n    FILE* time_file = fopen("simulation_time.txt", "w");\n    if (time_file) {\n        fprintf(time_file, "%.9f\\n", elapsed);\n        fclose(time_file);\n    }'
    else:
        # C++ Chrono style
        target_str = 'std::cout << "Simulation loop time: " << elapsed.count() << " seconds" << std::endl;'
        replacement_str = target_str + '\n    FILE* time_file = fopen("simulation_time.txt", "w");\n    if (time_file) {\n        fprintf(time_file, "%.9f\\n", elapsed.count());\n        fclose(time_file);\n    }'
        
    if target_str in content:
        content = content.replace(target_str, replacement_str)
        with open(path, 'w') as f:
            f.write(content)
        print(f"Successfully instrumented {target}")
    else:
        print(f"Error: Target pattern not found in {target}")
