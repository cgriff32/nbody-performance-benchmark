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

// C++17 compliant 64-byte aligned allocator
template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }
};

struct ParticlesSoA {
    std::vector<float, AlignedAllocator<float>> mass;
    std::vector<float, AlignedAllocator<float>> x, y, z;
    std::vector<float, AlignedAllocator<float>> vx, vy, vz;
    std::vector<float, AlignedAllocator<float>> ax, ay, az;
    std::vector<int> id;
    
    void resize(size_t n) {
        mass.resize(n);
        x.resize(n);
        y.resize(n);
        z.resize(n);
        vx.resize(n);
        vy.resize(n);
        vz.resize(n);
        ax.resize(n);
        ay.resize(n);
        az.resize(n);
        id.resize(n);
    }
};

// Computes the accelerations for all particles using SoA and SIMD
void compute_accelerations_soa(const ParticlesSoA& p, std::vector<float, AlignedAllocator<float>>& ax, std::vector<float, AlignedAllocator<float>>& ay, std::vector<float, AlignedAllocator<float>>& az) {
    int n = p.x.size();
    
    const float* __restrict px = (const float*)__builtin_assume_aligned(p.x.data(), 64);
    const float* __restrict py = (const float*)__builtin_assume_aligned(p.y.data(), 64);
    const float* __restrict pz = (const float*)__builtin_assume_aligned(p.z.data(), 64);
    const float* __restrict pm = (const float*)__builtin_assume_aligned(p.mass.data(), 64);
    
    float* __restrict pax = (float*)__builtin_assume_aligned(ax.data(), 64);
    float* __restrict pay = (float*)__builtin_assume_aligned(ay.data(), 64);
    float* __restrict paz = (float*)__builtin_assume_aligned(az.data(), 64);
    
    const float SOFTENING_SQ_f = 0.25f;
    
    for (int i = 0; i < n; ++i) {
        float xi = px[i];
        float yi = py[i];
        float zi = pz[i];
        
        float cur_ax = 0.0f;
        float cur_ay = 0.0f;
        float cur_az = 0.0f;
        
        #pragma omp simd reduction(+:cur_ax,cur_ay,cur_az)
        for (int j = 0; j < n; ++j) {
            float dx = px[j] - xi;
            float dy = py[j] - yi;
            float dz = pz[j] - zi;
            
            float dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING_SQ_f;
            float dist_inv = 1.0f / std::sqrt(dist_sq);
            float dist_32_inv = dist_inv * dist_inv * dist_inv;
            
            float force_factor = pm[j] * dist_32_inv;
            
            cur_ax += force_factor * dx;
            cur_ay += force_factor * dy;
            cur_az += force_factor * dz;
        }
        
        pax[i] = cur_ax;
        pay[i] = cur_ay;
        paz[i] = cur_az;
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
    
    // Initialize temporary SoA to run compute_accelerations
    ParticlesSoA p_soa;
    p_soa.resize(n);
    for (int i = 0; i < n; ++i) {
        p_soa.id[i] = particles[i].id;
        p_soa.mass[i] = (float)particles[i].mass;
        p_soa.x[i] = (float)particles[i].x;
        p_soa.y[i] = (float)particles[i].y;
        p_soa.z[i] = (float)particles[i].z;
        p_soa.vx[i] = (float)particles[i].vx;
        p_soa.vy[i] = (float)particles[i].vy;
        p_soa.vz[i] = (float)particles[i].vz;
        p_soa.ax[i] = (float)particles[i].ax;
        p_soa.ay[i] = (float)particles[i].ay;
        p_soa.az[i] = (float)particles[i].az;
    }
    
    // Compute initial accelerations
    compute_accelerations_soa(p_soa, p_soa.ax, p_soa.ay, p_soa.az);
    
    // Write initial accelerations back to particles for initial energy check
    for (int i = 0; i < n; ++i) {
        particles[i].ax = p_soa.ax[i];
        particles[i].ay = p_soa.ay[i];
        particles[i].az = p_soa.az[i];
    }
    
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const float dt_f = (float)dt;
    const float dt_half = 0.5f * dt_f;
    const float dt_sq_half = 0.5f * dt_f * dt_f;
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions & half-step velocities
        for (int i = 0; i < n; ++i) {
            p_soa.x[i]  += p_soa.vx[i] * dt_f + p_soa.ax[i] * dt_sq_half;
            p_soa.y[i]  += p_soa.vy[i] * dt_f + p_soa.ay[i] * dt_sq_half;
            p_soa.z[i]  += p_soa.vz[i] * dt_f + p_soa.az[i] * dt_sq_half;
            
            p_soa.vx[i] += p_soa.ax[i] * dt_half;
            p_soa.vy[i] += p_soa.ay[i] * dt_half;
            p_soa.vz[i] += p_soa.az[i] * dt_half;
        }
        
        // 2. Compute new accelerations
        compute_accelerations_soa(p_soa, p_soa.ax, p_soa.ay, p_soa.az);
        
        // 3. Complete velocity update
        for (int i = 0; i < n; ++i) {
            p_soa.vx[i] += p_soa.ax[i] * dt_half;
            p_soa.vy[i] += p_soa.ay[i] * dt_half;
            p_soa.vz[i] += 	p_soa.az[i] * dt_half;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Copy SoA state back to particles
    for (int i = 0; i < n; ++i) {
        particles[i].x = p_soa.x[i];
        particles[i].y = p_soa.y[i];
        particles[i].z = p_soa.z[i];
        particles[i].vx = p_soa.vx[i];
        particles[i].vy = p_soa.vy[i];
        particles[i].vz = p_soa.vz[i];
        particles[i].ax = p_soa.ax[i];
        particles[i].ay = p_soa.ay[i];
        particles[i].az = p_soa.az[i];
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
