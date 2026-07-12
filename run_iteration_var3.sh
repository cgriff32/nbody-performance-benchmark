#!/bin/bash
set -e

# Target directory must be provided
if [ -z "$1" ]; then
    echo "Usage: $0 <target_dir_absolute_path>"
    exit 1
fi

TARGET_DIR="$1"
VALIDATION_DIR="/home/chrisgriffiths/agent_workspace/performance_test/validation_sidecar"

ITER_NAME=$(basename "$TARGET_DIR")
CONTAINER_A="ContainerA_${ITER_NAME}"
CONTAINER_B="ContainerB_${ITER_NAME}"

# Setup folder paths
GEN0_FILE="/home/chrisgriffiths/agent_workspace/performance_test/generation_0.csv"

# Copy generation_0.csv to target directory so the container has local access
cp "$GEN0_FILE" "${TARGET_DIR}/generation_0.csv"

# Function to clean up containers
cleanup() {
    echo "Cleaning up containers..."
    docker rm -f "$CONTAINER_A" "$CONTAINER_B" >/dev/null 2>&1 || true
}

# Trap exit to cleanup
trap cleanup EXIT

cleanup

echo "Starting ContainerA (rw app sandbox)..."
# Pin to CPU 2, limit memory to 512M
docker run -d --name "$CONTAINER_A" \
    --cpuset-cpus="2" \
    -m 512m \
    -v "${TARGET_DIR}:/app:rw" \
    nbody-benchmark tail -f /dev/null

echo "Starting ContainerB (ro harness, shares PID namespace)..."
# Pin to CPU 3, limit memory to 512M, join ContainerA's PID namespace
docker run -d --name "$CONTAINER_B" \
    --pid=container:"$CONTAINER_A" \
    --cpuset-cpus="3" \
    -m 512m \
    -v "${TARGET_DIR}:/app:ro" \
    -v "${VALIDATION_DIR}:/validation:ro" \
    nbody-benchmark tail -f /dev/null

# Compile in ContainerA
echo "Compiling code in ContainerA..."
if ! docker exec "$CONTAINER_A" make -C /app clean > /dev/null 2>&1; then
    echo "Warning: make clean failed or no Makefile."
fi

# Capture make output
COMPILE_OUTPUT=$(docker exec "$CONTAINER_A" make -C /app 2>&1)
COMPILE_STATUS=$?

if [ $COMPILE_STATUS -ne 0 ]; then
    echo "Compilation FAILED!"
    echo "$COMPILE_OUTPUT"
    # Write failure json
    cat <<EOF > "${TARGET_DIR}/profile_results.json"
{
  "status": "failed",
  "elapsed_time": 0.0,
  "cpu_time": 0.0,
  "max_rss_kb": 0,
  "initial_energy": 0.0,
  "final_energy": 0.0,
  "energy_drift": 0.0,
  "error": "Compilation failed: $(echo "$COMPILE_OUTPUT" | tr '\n' ' ' | sed 's/"/\\"/g')"
}
EOF
    exit 0
fi

echo "Compilation successful."

# Remove previous output files if any
rm -f "${TARGET_DIR}/output.csv" "${TARGET_DIR}/time_output.txt" "${TARGET_DIR}/profile_results.json"

# Start the background profiler in ContainerB (writes to its local /tmp/profile_results.json)
echo "Launching background profiler in ContainerB..."
docker exec -d "$CONTAINER_B" python3 /validation/profile_and_validate.py \
    /app/generation_0.csv \
    /app/output.csv \
    /app/time_output.txt \
    /tmp/profile_results.json \
    nbody

# Allow profiler to bind/wait
sleep 0.5

# Run the simulation in ContainerA under time command with strict 150ms watchdog timeout
echo "Running simulator in ContainerA with 150ms timeout..."
set +e
SIM_OUTPUT=$(timeout -s SIGKILL 0.15s docker exec "$CONTAINER_A" /usr/bin/time -v -o /app/time_output.txt /app/nbody /app/generation_0.csv /app/output.csv 100 0.01 2>&1)
SIM_STATUS=$?
set -e

if [ $SIM_STATUS -eq 137 ]; then
    echo "Watchdog Timer triggered: Simulator timed out (> 150ms) and was killed."
    cat <<EOF > "${TARGET_DIR}/profile_results.json"
{
  "status": "failed",
  "elapsed_time": 0.0,
  "cpu_time": 0.0,
  "max_rss_kb": 0,
  "initial_energy": 0.0,
  "final_energy": 0.0,
  "energy_drift": 0.0,
  "error": "Watchdog Timer triggered: Simulator exceeded 150ms timeout limit."
}
EOF
    exit 0
elif [ $SIM_STATUS -ne 0 ]; then
    echo "Simulator execution failed with exit code $SIM_STATUS"
    echo "$SIM_OUTPUT"
    cat <<EOF > "${TARGET_DIR}/profile_results.json"
{
  "status": "failed",
  "elapsed_time": 0.0,
  "cpu_time": 0.0,
  "max_rss_kb": 0,
  "initial_energy": 0.0,
  "final_energy": 0.0,
  "energy_drift": 0.0,
  "error": "Simulator crashed or exited with non-zero code $SIM_STATUS. Output: $(echo "$SIM_OUTPUT" | tr '\n' ' ' | sed 's/"/\\"/g')"
}
EOF
    exit 0
fi

# Wait for validation results to be written in ContainerB (max 10 seconds)
echo "Waiting for validation results in ContainerB..."
VALID=0
for i in {1..20}; do
    if docker exec "$CONTAINER_B" test -f /tmp/profile_results.json >/dev/null 2>&1; then
        VALID=1
        break
    fi
    sleep 0.5
done

if [ $VALID -eq 1 ]; then
    # Copy the profile results file from ContainerB to the host directory
    docker exec "$CONTAINER_B" cat /tmp/profile_results.json > "${TARGET_DIR}/profile_results.json"
else
    echo "Timeout waiting for profiler to output results!"
    cat <<EOF > "${TARGET_DIR}/profile_results.json"
{
  "status": "failed",
  "elapsed_time": 0.0,
  "cpu_time": 0.0,
  "max_rss_kb": 0,
  "initial_energy": 0.0,
  "final_energy": 0.0,
  "energy_drift": 0.0,
  "error": "Timeout waiting for profiler results."
}
EOF
fi

echo "--- BENCHMARK COMPLETE ---"
cat "${TARGET_DIR}/profile_results.json"
echo ""
