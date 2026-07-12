#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>

const float G = 1.0f;
const float SOFTENING_SQ = 0.25f;

struct Particle {
    int id;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

// Computes the accelerations using SoA branchless N^2 in float
void compute_accelerations_soa(int n, const float* __restrict__ mass, const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz, float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az) {
    for (int i = 0; i < n; ++i) {
        float ax_i = 0.0f;
        float ay_i = 0.0f;
        float az_i = 0.0f;
        float px_i = px[i];
        float py_i = py[i];
        float pz_i = pz[i];
        
        for (int j = 0; j < n; ++j) {
            float dx = px[j] - px_i;
            float dy = py[j] - py_i;
            float dz = pz[j] - pz_i;
            
            float dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING_SQ;
            float inv_dist = 1.0f / std::sqrt(dist_sq);
            float inv_dist_32 = inv_dist * inv_dist * inv_dist;
            
            float force_factor = G * mass[j] * inv_dist_32;
            
            ax_i += force_factor * dx;
            ay_i += force_factor * dy;
            az_i += force_factor * dz;
        }
        ax[i] = ax_i;
        ay[i] = ay_i;
        az[i] = az_i;
    }
}

// Calculates the total Hamiltonian energy (Kinetic + Potential)
double calculate_energy(const std::vector<Particle>& particles) {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    int n = particles.size();
    
    for (int i = 0; i < n; ++i) {
        double v_sq = particles[i].vx * particles[i].vx + 
                      particles[i].vy * particles[i].vy + 
                      particles[i].vz * particles[i].vz;
        kinetic_energy += 0.5 * particles[i].mass * v_sq;
        
        for (int j = i + 1; j < n; ++j) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING_SQ);
            potential_energy -= G * particles[i].mass * particles[j].mass / dist;
        }
    }
    
    return kinetic_energy + potential_energy;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_csv> <output_csv> <num_steps> <dt>" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    int num_steps = std::stoi(argv[3]);
    double dt_d = std::stod(argv[4]);
    float dt = (float)dt_d;
    
    std::vector<Particle> particles;
    
    // Load initial state CSV
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }
    
    std::string line;
    // Skip header line
    std::getline(infile, line);
    
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        Particle p;
        
        std::getline(ss, val, ','); p.id = std::stoi(val);
        std::getline(ss, val, ','); p.mass = std::stod(val);
        std::getline(ss, val, ','); p.x = std::stod(val);
        std::getline(ss, val, ','); p.y = std::stod(val);
        std::getline(ss, val, ','); p.z = std::stod(val);
        std::getline(ss, val, ','); p.vx = std::stod(val);
        std::getline(ss, val, ','); p.vy = std::stod(val);
        std::getline(ss, val, ','); p.vz = std::stod(val);
        p.ax = p.ay = p.az = 0.0;
        
        particles.push_back(p);
    }
    infile.close();
    
    int n = particles.size();
    std::cout << "Loaded " << n << " particles." << std::endl;
    
    // Allocate SoA in float
    std::vector<float> mass(n);
    std::vector<float> px(n), py(n), pz(n);
    std::vector<float> vx(n), vy(n), vz(n);
    std::vector<float> ax(n), ay(n), az(n);
    
    for (int i = 0; i < n; ++i) {
        mass[i] = (float)particles[i].mass;
        px[i] = (float)particles[i].x;
        py[i] = (float)particles[i].y;
        pz[i] = (float)particles[i].z;
        vx[i] = (float)particles[i].vx;
        vy[i] = (float)particles[i].vy;
        vz[i] = (float)particles[i].vz;
    }
    
    // Compute initial accelerations
    compute_accelerations_soa(n, mass.data(), px.data(), py.data(), pz.data(), ax.data(), ay.data(), az.data());
    
    // Copy back to calculate initial energy
    for (int i = 0; i < n; ++i) {
        particles[i].ax = ax[i];
        particles[i].ay = ay[i];
        particles[i].az = az[i];
    }
    
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulation Loop (Velocity Verlet with SoA in float)
    float dt_half = 0.5f * dt;
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update velocities (half step) and positions (full step)
        for (int i = 0; i < n; ++i) {
            vx[i] += ax[i] * dt_half;
            vy[i] += ay[i] * dt_half;
            vz[i] += az[i] * dt_half;
            
            px[i] += vx[i] * dt;
            py[i] += vy[i] * dt;
            pz[i] += vz[i] * dt;
        }
        
        // 2. Compute new accelerations
        compute_accelerations_soa(n, mass.data(), px.data(), py.data(), pz.data(), ax.data(), ay.data(), az.data());
        
        // 3. Update velocities (half step)
        for (int i = 0; i < n; ++i) {
            vx[i] += ax[i] * dt_half;
            vy[i] += ay[i] * dt_half;
            vz[i] += az[i] * dt_half;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Copy SoA back to particles
    for (int i = 0; i < n; ++i) {
        particles[i].x = px[i];
        particles[i].y = py[i];
        particles[i].z = pz[i];
        particles[i].vx = vx[i];
        particles[i].vy = vy[i];
        particles[i].vz = vz[i];
        particles[i].ax = ax[i];
        particles[i].ay = ay[i];
        particles[i].az = az[i];
    }
    
    double final_energy = calculate_energy(particles);
    double energy_drift = std::abs(final_energy - initial_energy);
    
    std::cout << "Final Energy:   " << final_energy << std::endl;
    std::cout << "Energy Drift:   " << energy_drift << std::endl;
    std::cout << "Simulation loop time: " << elapsed.count() << " seconds" << std::endl;
    FILE* time_file = fopen("simulation_time.txt", "w");
    if (time_file) {
        fprintf(time_file, "%.9f\n", elapsed.count());
        fclose(time_file);
    }
    
    // Write final state CSV
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }
    
    outfile << std::scientific << std::setprecision(17);
    outfile << "id,mass,x,y,z,vx,vy,vz\n";
    for (const auto& p : particles) {
        outfile << p.id << "," << p.mass << "," 
                << p.x << "," << p.y << "," << p.z << ","
                << p.vx << "," << p.vy << "," << p.vz << "\n";
    }
    outfile.close();
    
    return 0;
}
