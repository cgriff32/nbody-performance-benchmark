import os
import sys
import time
import math
import csv
import json
import ctypes

G = 1.0
SOFTENING_SQ = 0.25

# Try to load libenergy.so
try:
    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libenergy.so'))
    lib.calculate_potential_energy.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    lib.calculate_potential_energy.restype = ctypes.c_double
    HAS_LIBENERGY = True
except Exception as e:
    print(f"Profiler: Failed to load libenergy.so, falling back to pure Python: {e}")
    HAS_LIBENERGY = False

def load_particles(filename):
    particles = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            particles.append({
                'id': int(row['id']),
                'mass': float(row['mass']),
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row['z']),
                'vx': float(row['vx']),
                'vy': float(row['vy']),
                'vz': float(row['vz'])
            })
    return particles

def calculate_energy(particles):
    kinetic = 0.0
    n = len(particles)
    
    for i in range(n):
        p_i = particles[i]
        v_sq = p_i['vx']**2 + p_i['vy']**2 + p_i['vz']**2
        kinetic += 0.5 * p_i['mass'] * v_sq
        
    if HAS_LIBENERGY:
        # Prepare arrays
        x_arr = (ctypes.c_double * n)(*[p['x'] for p in particles])
        y_arr = (ctypes.c_double * n)(*[p['y'] for p in particles])
        z_arr = (ctypes.c_double * n)(*[p['z'] for p in particles])
        m_arr = (ctypes.c_double * n)(*[p['mass'] for p in particles])
        potential = lib.calculate_potential_energy(x_arr, y_arr, z_arr, m_arr, n)
    else:
        potential = 0.0
        for i in range(n):
            p_i = particles[i]
            for j in range(i + 1, n):
                p_j = particles[j]
                dx = p_j['x'] - p_i['x']
                dy = p_j['y'] - p_i['y']
                dz = p_j['z'] - p_i['z']
                dist = math.sqrt(dx**2 + dy**2 + dz**2 + SOFTENING_SQ)
                potential -= G * p_i['mass'] * p_j['mass'] / dist
                
    return kinetic + potential

def find_pid_by_name(name):
    for pid_str in os.listdir('/proc'):
        if pid_str.isdigit():
            pid = int(pid_str)
            try:
                with open(f'/proc/{pid}/comm', 'r') as f:
                    comm = f.read().strip()
                if comm == name:
                    return pid
            except IOError:
                continue
    return None

def parse_time_output(time_file_path):
    metrics = {
        'user_time': 0.0,
        'sys_time': 0.0,
        'elapsed_time': 0.0,
        'max_rss': 0
    }
    if not os.path.exists(time_file_path):
        return metrics
    
    with open(time_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'User time (seconds):' in line:
                metrics['user_time'] = float(line.split(':')[-1].strip())
            elif 'System time (seconds):' in line:
                metrics['sys_time'] = float(line.split(':')[-1].strip())
            elif 'Elapsed (wall clock) time' in line:
                time_str = line.split(':')[-1].strip()
                parts = time_str.split(':')
                if len(parts) == 3:
                    elapsed = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    elapsed = int(parts[0]) * 60 + float(parts[1])
                else:
                    elapsed = float(parts[0])
                metrics['elapsed_time'] = elapsed
            elif 'Maximum resident set size (kbytes):' in line:
                metrics['max_rss'] = int(line.split(':')[-1].strip())
                
    return metrics

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 profile_and_validate.py <gen0_csv> <output_csv> <time_output_txt> <results_json> [process_name]")
        sys.exit(1)
        
    gen0_csv = sys.argv[1]
    output_csv = sys.argv[2]
    time_output_txt = sys.argv[3]
    results_json = sys.argv[4]
    process_name = sys.argv[5] if len(sys.argv) > 5 else "nbody"
    
    print(f"Profiler: Monitoring process '{process_name}'...")
    
    # Poll for the process to start
    start_wait = time.time()
    pid = None
    while pid is None:
        pid = find_pid_by_name(process_name)
        if time.time() - start_wait > 10.0:
            break
        time.sleep(0.001)
        
    peak_rss = 0
    if pid:
        print(f"Profiler: Found process '{process_name}' with PID {pid}. Monitoring memory...")
        while os.path.exists(f'/proc/{pid}'):
            try:
                with open(f'/proc/{pid}/status', 'r') as f:
                    for line in f:
                        if 'VmHWM' in line:
                            peak_rss = int(line.split()[1])
                            break
            except IOError:
                break
            time.sleep(0.01)
        print("Profiler: Process finished.")
    else:
        print("Profiler: Process was not detected or finished too quickly for PID detection.")
        
    time.sleep(0.5)
    
    time_metrics = parse_time_output(time_output_txt)
    
    max_rss = max(time_metrics['max_rss'], peak_rss)
    
    # Look for high-precision timing file written by the simulator itself
    sim_time_file = os.path.join(os.path.dirname(output_csv), "simulation_time.txt")
    if os.path.exists(sim_time_file):
        try:
            with open(sim_time_file, 'r') as f:
                high_prec_time = float(f.read().strip())
            if high_prec_time > 0.0:
                elapsed_time = high_prec_time
                cpu_time = high_prec_time
                print(f"Profiler: Found high-precision timing file: {high_prec_time:.6f} seconds")
            else:
                elapsed_time = time_metrics['elapsed_time']
                cpu_time = time_metrics['user_time'] + time_metrics['sys_time']
        except Exception as e:
            print(f"Profiler: Failed to parse high-precision file: {e}")
            elapsed_time = time_metrics['elapsed_time']
            cpu_time = time_metrics['user_time'] + time_metrics['sys_time']
    else:
        elapsed_time = time_metrics['elapsed_time']
        cpu_time = time_metrics['user_time'] + time_metrics['sys_time']
    
    results = {
        'status': 'failed',
        'elapsed_time': elapsed_time,
        'cpu_time': cpu_time,
        'max_rss_kb': max_rss,
        'initial_energy': 0.0,
        'final_energy': 0.0,
        'energy_drift': 0.0,
        'error': None
    }
    
    if not os.path.exists(gen0_csv):
        results['error'] = f"Gen0 file {gen0_csv} not found."
        write_results(results, results_json)
        return
        
    if not os.path.exists(output_csv):
        results['error'] = f"Output file {output_csv} not found."
        write_results(results, results_json)
        return
        
    try:
        gen0_particles = load_particles(gen0_csv)
        output_particles = load_particles(output_csv)
        
        initial_energy = calculate_energy(gen0_particles)
        final_energy = calculate_energy(output_particles)
        energy_drift = abs(final_energy - initial_energy)
        
        results['initial_energy'] = initial_energy
        results['final_energy'] = final_energy
        results['energy_drift'] = energy_drift
        
        if energy_drift > 1e-5:
            results['error'] = f"Energy drift {energy_drift:.6e} exceeds limit 1e-5. Hamiltonian energy conservation violated."
        else:
            results['status'] = 'success'
            
    except Exception as e:
        results['error'] = f"Error during energy calculation: {str(e)}"
        
    write_results(results, results_json)

def write_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Profiler: Results written to {path}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
