#!/bin/bash
set -e

base_dir="/home/chrisgriffiths/agent_workspace/performance_test"

echo "=== RE-BENCHMARKING ALL ITERATIONS & VARIANTS ==="

echo "1. Re-benchmarking baseline..."
./run_iteration.sh "${base_dir}/baseline"

echo "2. Re-benchmarking iter_1..."
./run_iteration.sh "${base_dir}/iter_1"

echo "3. Re-benchmarking iter_2..."
./run_iteration.sh "${base_dir}/iter_2"

echo "4. Re-benchmarking iter_3..."
./run_iteration.sh "${base_dir}/iter_3"

echo "5. Re-benchmarking iter_4..."
./run_iteration.sh "${base_dir}/iter_4"

echo "6. Re-benchmarking iter_5..."
./run_iteration.sh "${base_dir}/iter_5"

echo "7. Re-benchmarking iter_6..."
./run_iteration_var4.sh "${base_dir}/iter_6"

echo "8. Re-benchmarking iter_variant_1 (memory limit)..."
./run_iteration_var1.sh "${base_dir}/iter_variant_1"

echo "9. Re-benchmarking iter_variant_2 (precision limit)..."
./run_iteration_var2.sh "${base_dir}/iter_variant_2"

echo "10. Re-benchmarking iter_variant_3 (watchdog limit)..."
./run_iteration_var3.sh "${base_dir}/iter_variant_3"

echo "=== ALL RE-BENCHMARKS COMPLETED SUCCESSFULLY ==="
