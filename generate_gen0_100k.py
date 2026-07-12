import json
import csv
import random

def generate_initial_state(num_particles=100000, json_filename="generation_0_100k.json", csv_filename="generation_0_100k.csv"):
    # Seed the random number generator to ensure deterministic output
    random.seed(42)
    
    particles = []
    for i in range(num_particles):
        x = random.uniform(-100.0, 100.0)
        y = random.uniform(-100.0, 100.0)
        z = random.uniform(-100.0, 100.0)
        
        # Velocities scaled down
        vx = random.uniform(-0.01, 0.01)
        vy = random.uniform(-0.01, 0.01)
        vz = random.uniform(-0.01, 0.01)
        
        # Masses scaled down
        mass = random.uniform(1e-6, 1e-5)
        
        particles.append({
            "id": i,
            "mass": mass,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz
        })
        
    # Write CSV
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "mass", "x", "y", "z", "vx", "vy", "vz"])
        for p in particles:
            writer.writerow([p["id"], p["mass"], p["x"], p["y"], p["z"], p["vx"], p["vy"], p["vz"]])
            
    print(f"Generated {num_particles} particles in {csv_filename}")

if __name__ == "__main__":
    generate_initial_state()
