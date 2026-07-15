import os
import sys
import csv
import time
import ctypes
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set double precision as default for PyTorch to ensure high energy conservation
torch.set_default_dtype(torch.float64)

class PairwisePotentialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1, bias=True)
        # Initialize weights randomly to simulate training
        nn.init.uniform_(self.fc.weight, -1.0, 1.0)
        nn.init.uniform_(self.fc.bias, -1.0, 1.0)
        
    def forward(self, r):
        # r is of shape [num_pairs, 1]
        f1 = r
        f2 = 1.0 / torch.sqrt(r**2 + 0.25)
        f3 = 1.0 / (r**2 + 0.25)
        features = torch.cat([f1, f2, f3], dim=1)
        return self.fc(features)

def train_model(model):
    # Generate 10,000 training points for fitting potential
    r_train = torch.linspace(0.0, 250.0, 10000).unsqueeze(1)
    y_train = -1.0 / torch.sqrt(r_train**2 + 0.25)
    
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, 
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-16, tolerance_change=1e-16)
    
    def closure():
        optimizer.zero_grad()
        outputs = model(r_train)
        loss = torch.mean((outputs - y_train)**2)
        loss.backward()
        return loss
        
    optimizer.step(closure)

# Load the shared library
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnbody.so")
lib = ctypes.CDLL(lib_path)

lib.compute_accelerations_cpp.argtypes = [
    ctypes.POINTER(ctypes.c_double), # x
    ctypes.POINTER(ctypes.c_double), # y
    ctypes.POINTER(ctypes.c_double), # z
    ctypes.POINTER(ctypes.c_double), # mass
    ctypes.POINTER(ctypes.c_double), # ax
    ctypes.POINTER(ctypes.c_double), # ay
    ctypes.POINTER(ctypes.c_double), # az
    ctypes.c_int,                    # n
    ctypes.POINTER(ctypes.c_double)  # w
]
lib.compute_accelerations_cpp.restype = None

def compute_accelerations(pos, mass, w):
    n = pos.shape[0]
    
    pos_np = pos.cpu().numpy()
    mass_np = mass.cpu().numpy()
    w_np = w.cpu().numpy()
    
    x = np.ascontiguousarray(pos_np[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(pos_np[:, 1], dtype=np.float64)
    z = np.ascontiguousarray(pos_np[:, 2], dtype=np.float64)
    mass_c = np.ascontiguousarray(mass_np, dtype=np.float64)
    w_c = np.ascontiguousarray(w_np, dtype=np.float64)
    
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    az = np.zeros(n, dtype=np.float64)
    
    lib.compute_accelerations_cpp(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mass_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ay.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        az.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    accel = np.stack([ax, ay, az], axis=1)
    return torch.from_numpy(accel)

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 simulate_ml.py <input_csv> <output_csv> <num_steps> <dt>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_steps = int(sys.argv[3])
    dt = float(sys.argv[4])
    
    # Load particles from CSV
    ids = []
    masses = []
    positions = []
    velocities = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row['id']))
            masses.append(float(row['mass']))
            positions.append([float(row['x']), float(row['y']), float(row['z'])])
            velocities.append([float(row['vx']), float(row['vy']), float(row['vz'])])
            
    n = len(ids)
    
    # Convert to PyTorch tensors
    mass = torch.tensor(masses)
    pos = torch.tensor(positions)
    vel = torch.tensor(velocities)
    
    # Train the neural network
    model = PairwisePotentialNet()
    train_model(model)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Extract trained weights
    w = model.fc.weight.data[0]
    
    # Compute initial accelerations
    accel = compute_accelerations(pos, mass, w)
    
    start_time = time.time()
    
    # Simulation loop (Velocity Verlet)
    for step in range(num_steps):
        # 1. Update positions
        pos = pos + vel * dt + 0.5 * accel * dt**2
        
        # 2. Compute new accelerations
        accel_new = compute_accelerations(pos, mass, w)
        
        # 3. Update velocities
        vel = vel + 0.5 * (accel + accel_new) * dt
        accel = accel_new
        
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Simulation loop time: {elapsed:.9f} seconds")
    
    # Write simulation time to the same folder as the output CSV file
    out_dir = os.path.dirname(os.path.abspath(output_file))
    sim_time_path = os.path.join(out_dir, "simulation_time.txt")
    with open(sim_time_path, "w") as f:
        f.write(f"{elapsed:.9f}\n")
        
    # Write final state to output CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        for i in range(n):
            writer.writerow([
                ids[i],
                masses[i],
                pos[i, 0].item(),
                pos[i, 1].item(),
                pos[i, 2].item(),
                vel[i, 0].item(),
                vel[i, 1].item(),
                vel[i, 2].item()
            ])

if __name__ == "__main__":
    main()
