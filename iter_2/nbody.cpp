#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>

const double G = 1.0;
const double SOFTENING_SQ = 0.25;

struct Particle {
    int id;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

// Computes the accelerations for all particles using SOA with alignment
void compute_accelerations_soa(
    int n,
    const double* __restrict x,
    const double* __restrict y,
    const double* __restrict z,
    const double* __restrict mass,
    double* __restrict ax,
    double* __restrict ay,
    double* __restrict az) 
{
    const double* __restrict x_al = (const double*)__builtin_assume_aligned(x, 64);
    const double* __restrict y_al = (const double*)__builtin_assume_aligned(y, 64);
    const double* __restrict z_al = (const double*)__builtin_assume_aligned(z, 64);
    const double* __restrict mass_al = (const double*)__builtin_assume_aligned(mass, 64);
    double* __restrict ax_al = (double*)__builtin_assume_aligned(ax, 64);
    double* __restrict ay_al = (double*)__builtin_assume_aligned(ay, 64);
    double* __restrict az_al = (double*)__builtin_assume_aligned(az, 64);

    // Reset accelerations to zero
    for (int i = 0; i < n; ++i) {
        ax_al[i] = 0.0;
        ay_al[i] = 0.0;
        az_al[i] = 0.0;
    }
    
    // N^2 gravitational interaction calculation
    for (int i = 0; i < n; ++i) {
        double xi = x_al[i];
        double yi = y_al[i];
        double zi = z_al[i];
        double axi = 0.0;
        double ayi = 0.0;
        double azi = 0.0;
        
        #pragma omp simd reduction(+:axi,ayi,azi)
        for (int j = 0; j < n; ++j) {
            double dx = x_al[j] - xi;
            double dy = y_al[j] - yi;
            double dz = z_al[j] - zi;
            
            double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING_SQ;
            double dist_32 = dist_sq * std::sqrt(dist_sq);
            
            double force_factor = mass_al[j] / dist_32;
            
            axi += force_factor * dx;
            ayi += force_factor * dy;
            azi += force_factor * dz;
        }
        ax_al[i] = axi;
        ay_al[i] = ayi;
        az_al[i] = azi;
    }
}

// Keep original compute_accelerations for initial state
void compute_accelerations(std::vector<Particle>& particles) {
    int n = particles.size();
    for (int i = 0; i < n; ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        particles[i].az = 0.0;
    }
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
    double dt = std::stod(argv[4]);
    
    std::vector<Particle> particles;
    
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }
    std::string line;
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
    
    compute_accelerations(particles);
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    // Allocate 64-byte aligned memory
    double* mass = (double*)aligned_alloc(64, n * sizeof(double));
    double* x = (double*)aligned_alloc(64, n * sizeof(double));
    double* y = (double*)aligned_alloc(64, n * sizeof(double));
    double* z = (double*)aligned_alloc(64, n * sizeof(double));
    double* vx = (double*)aligned_alloc(64, n * sizeof(double));
    double* vy = (double*)aligned_alloc(64, n * sizeof(double));
    double* vz = (double*)aligned_alloc(64, n * sizeof(double));
    double* ax = (double*)aligned_alloc(64, n * sizeof(double));
    double* ay = (double*)aligned_alloc(64, n * sizeof(double));
    double* az = (double*)aligned_alloc(64, n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        mass[i] = particles[i].mass;
        x[i] = particles[i].x;
        y[i] = particles[i].y;
        z[i] = particles[i].z;
        vx[i] = particles[i].vx;
        vy[i] = particles[i].vy;
        vz[i] = particles[i].vz;
        ax[i] = particles[i].ax;
        ay[i] = particles[i].ay;
        az[i] = particles[i].az;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions and half-step velocities
        for (int i = 0; i < n; ++i) {
            x[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
            y[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
            z[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
            
            vx[i] += 0.5 * ax[i] * dt;
            vy[i] += 0.5 * ay[i] * dt;
            vz[i] += 0.5 * az[i] * dt;
        }
        
        // 2. Compute new accelerations
        compute_accelerations_soa(n, x, y, z, mass, ax, ay, az);
        
        // 3. Update velocities (second half-step)
        for (int i = 0; i < n; ++i) {
            vx[i] += 0.5 * ax[i] * dt;
            vy[i] += 0.5 * ay[i] * dt;
            vz[i] += 0.5 * az[i] * dt;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Copy back
    for (int i = 0; i < n; ++i) {
        particles[i].x = x[i];
        particles[i].y = y[i];
        particles[i].z = z[i];
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
    
    // Free aligned memories
    free(mass); free(x); free(y); free(z);
    free(vx); free(vy); free(vz);
    free(ax); free(ay); free(az);
    
    return 0;
}
