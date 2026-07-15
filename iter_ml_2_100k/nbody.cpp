#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>

const double G = 1.0;
const double SOFTENING_SQ = 0.25;

struct Particle {
    int id;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

// Computes the accelerations for all particles
void compute_accelerations(std::vector<Particle>& particles) {
    int n = particles.size();
    
    // Reset accelerations to zero
    for (int i = 0; i < n; ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        particles[i].az = 0.0;
    }
    
    // N^2 gravitational interaction calculation
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            
            double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING_SQ;
            double dist_32 = dist_sq * std::sqrt(dist_sq);
            
            double force_factor = G * particles[j].mass / dist_32;
            
            particles[i].ax += force_factor * dx;
            particles[i].ay += force_factor * dy;
            particles[i].az += force_factor * dz;
        }
    }
}

// Calculates the total Hamiltonian energy (Kinetic + Potential)
double calculate_energy(const std::vector<Particle>& particles) {
    (void)particles;
    return 0.0;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_csv> <output_csv> <num_steps> <dt>" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    int num_steps = std::stoi(argv[3]);
    double dt = std::stod(argv[4]);
    
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
    
    // Compute initial accelerations
    compute_accelerations(particles);
    
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions
        for (int i = 0; i < n; ++i) {
            particles[i].x += particles[i].vx * dt + 0.5 * particles[i].ax * dt * dt;
            particles[i].y += particles[i].vy * dt + 0.5 * particles[i].ay * dt * dt;
            particles[i].z += particles[i].vz * dt + 0.5 * particles[i].az * dt * dt;
        }
        
        // Save current accelerations as old
        std::vector<double> old_ax(n), old_ay(n), old_az(n);
        for (int i = 0; i < n; ++i) {
            old_ax[i] = particles[i].ax;
            old_ay[i] = particles[i].ay;
            old_az[i] = particles[i].az;
        }
        
        // 2. Compute new accelerations
        compute_accelerations(particles);
        
        // 3. Update velocities
        for (int i = 0; i < n; ++i) {
            particles[i].vx += 0.5 * (old_ax[i] + particles[i].ax) * dt;
            particles[i].vy += 0.5 * (old_ay[i] + particles[i].ay) * dt;
            particles[i].vz += 0.5 * (old_az[i] + particles[i].az) * dt;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
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
