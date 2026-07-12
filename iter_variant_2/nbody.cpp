#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <algorithm>
#include <cstdlib>
#include <new>

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U>&) {}
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 32, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) {
        free(p);
    }
};

const double G = 1.0;
const double SOFTENING_SQ = 0.25;

struct Particle {
    int id;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

struct ParticlesSoA {
    int n; // padded to multiple of 8
    int original_n;
    std::vector<double, AlignedAllocator<double>> mass;
    std::vector<double, AlignedAllocator<double>> x, y, z;
    std::vector<double, AlignedAllocator<double>> vx, vy, vz;
    std::vector<double, AlignedAllocator<double>> ax, ay, az;

    ParticlesSoA(int num) : n((num + 7) / 8 * 8), original_n(num), mass(n, 0.0), x(n, 0.0), y(n, 0.0), z(n, 0.0), vx(n, 0.0), vy(n, 0.0), vz(n, 0.0), ax(n, 0.0), ay(n, 0.0), az(n, 0.0) {}

    void from_aos(const std::vector<Particle>& aos) {
        for (int i = 0; i < original_n; ++i) {
            mass[i] = aos[i].mass;
            x[i] = aos[i].x;
            y[i] = aos[i].y;
            z[i] = aos[i].z;
            vx[i] = aos[i].vx;
            vy[i] = aos[i].vy;
            vz[i] = aos[i].vz;
            ax[i] = aos[i].ax;
            ay[i] = aos[i].ay;
            az[i] = aos[i].az;
        }
    }

    void to_aos(std::vector<Particle>& aos) const {
        for (int i = 0; i < original_n; ++i) {
            aos[i].x = x[i];
            aos[i].y = y[i];
            aos[i].z = z[i];
            aos[i].vx = vx[i];
            aos[i].vy = vy[i];
            aos[i].vz = vz[i];
            aos[i].ax = ax[i];
            aos[i].ay = ay[i];
            aos[i].az = az[i];
        }
    }
};

inline double reduce_m256d(__m256d v) {
    __m128d lo = _mm256_extractf128_pd(v, 0);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_shuffle_pd(sum, sum, 1);
    __m128d total = _mm_add_pd(sum, shuf);
    return _mm_cvtsd_f64(total);
}

void compute_accelerations_soa(ParticlesSoA& soa) {
    int n = soa.n;
    int limit = soa.original_n;
    
    double* __restrict__ ax = soa.ax.data();
    double* __restrict__ ay = soa.ay.data();
    double* __restrict__ az = soa.az.data();
    const double* __restrict__ x = soa.x.data();
    const double* __restrict__ y = soa.y.data();
    const double* __restrict__ z = soa.z.data();
    const double* __restrict__ mass = soa.mass.data();

    // Reset accelerations
    for (int i = 0; i < n; ++i) {
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
    }

    __m256d v_soft = _mm256_set1_pd(SOFTENING_SQ);

    int i = 0;
    for (; i < limit - 1; i += 2) {
        __m256d v_px0 = _mm256_set1_pd(x[i]);
        __m256d v_py0 = _mm256_set1_pd(y[i]);
        __m256d v_pz0 = _mm256_set1_pd(z[i]);

        __m256d v_px1 = _mm256_set1_pd(x[i+1]);
        __m256d v_py1 = _mm256_set1_pd(y[i+1]);
        __m256d v_pz1 = _mm256_set1_pd(z[i+1]);

        __m256d v_ax0 = _mm256_setzero_pd();
        __m256d v_ay0 = _mm256_setzero_pd();
        __m256d v_az0 = _mm256_setzero_pd();

        __m256d v_ax1 = _mm256_setzero_pd();
        __m256d v_ay1 = _mm256_setzero_pd();
        __m256d v_az1 = _mm256_setzero_pd();

        for (int j = 0; j < n; j += 4) {
            __m256d v_xj = _mm256_load_pd(x + j);
            __m256d v_yj = _mm256_load_pd(y + j);
            __m256d v_zj = _mm256_load_pd(z + j);
            __m256d v_mj = _mm256_load_pd(mass + j);

            // Stream 0
            __m256d v_dx0 = _mm256_sub_pd(v_xj, v_px0);
            __m256d v_dy0 = _mm256_sub_pd(v_yj, v_py0);
            __m256d v_dz0 = _mm256_sub_pd(v_zj, v_pz0);

            __m256d v_dist_sq0 = v_soft;
            v_dist_sq0 = _mm256_fmadd_pd(v_dx0, v_dx0, v_dist_sq0);
            v_dist_sq0 = _mm256_fmadd_pd(v_dy0, v_dy0, v_dist_sq0);
            v_dist_sq0 = _mm256_fmadd_pd(v_dz0, v_dz0, v_dist_sq0);

            __m256d v_sqrt0 = _mm256_sqrt_pd(v_dist_sq0);
            __m256d v_dist_32_0 = _mm256_mul_pd(v_dist_sq0, v_sqrt0);
            __m256d v_ff0 = _mm256_div_pd(v_mj, v_dist_32_0);

            v_ax0 = _mm256_fmadd_pd(v_ff0, v_dx0, v_ax0);
            v_ay0 = _mm256_fmadd_pd(v_ff0, v_dy0, v_ay0);
            v_az0 = _mm256_fmadd_pd(v_ff0, v_dz0, v_az0);

            // Stream 1
            __m256d v_dx1 = _mm256_sub_pd(v_xj, v_px1);
            __m256d v_dy1 = _mm256_sub_pd(v_yj, v_py1);
            __m256d v_dz1 = _mm256_sub_pd(v_zj, v_pz1);

            __m256d v_dist_sq1 = v_soft;
            v_dist_sq1 = _mm256_fmadd_pd(v_dx1, v_dx1, v_dist_sq1);
            v_dist_sq1 = _mm256_fmadd_pd(v_dy1, v_dy1, v_dist_sq1);
            v_dist_sq1 = _mm256_fmadd_pd(v_dz1, v_dz1, v_dist_sq1);

            __m256d v_sqrt1 = _mm256_sqrt_pd(v_dist_sq1);
            __m256d v_dist_32_1 = _mm256_mul_pd(v_dist_sq1, v_sqrt1);
            __m256d v_ff1 = _mm256_div_pd(v_mj, v_dist_32_1);

            v_ax1 = _mm256_fmadd_pd(v_ff1, v_dx1, v_ax1);
            v_ay1 = _mm256_fmadd_pd(v_ff1, v_dy1, v_ay1);
            v_az1 = _mm256_fmadd_pd(v_ff1, v_dz1, v_az1);
        }

        ax[i] = reduce_m256d(v_ax0);
        ay[i] = reduce_m256d(v_ay0);
        az[i] = reduce_m256d(v_az0);

        ax[i+1] = reduce_m256d(v_ax1);
        ay[i+1] = reduce_m256d(v_ay1);
        az[i+1] = reduce_m256d(v_az1);
    }

    // Cleanup loop for odd limit
    for (; i < limit; ++i) {
        __m256d v_px = _mm256_set1_pd(x[i]);
        __m256d v_py = _mm256_set1_pd(y[i]);
        __m256d v_pz = _mm256_set1_pd(z[i]);

        __m256d v_ax = _mm256_setzero_pd();
        __m256d v_ay = _mm256_setzero_pd();
        __m256d v_az = _mm256_setzero_pd();

        for (int j = 0; j < n; j += 4) {
            __m256d v_xj = _mm256_load_pd(x + j);
            __m256d v_yj = _mm256_load_pd(y + j);
            __m256d v_zj = _mm256_load_pd(z + j);
            __m256d v_mj = _mm256_load_pd(mass + j);

            __m256d v_dx = _mm256_sub_pd(v_xj, v_px);
            __m256d v_dy = _mm256_sub_pd(v_yj, v_py);
            __m256d v_dz = _mm256_sub_pd(v_zj, v_pz);

            __m256d v_dist_sq = v_soft;
            v_dist_sq = _mm256_fmadd_pd(v_dx, v_dx, v_dist_sq);
            v_dist_sq = _mm256_fmadd_pd(v_dy, v_dy, v_dist_sq);
            v_dist_sq = _mm256_fmadd_pd(v_dz, v_dz, v_dist_sq);

            __m256d v_sqrt = _mm256_sqrt_pd(v_dist_sq);
            __m256d v_dist_32 = _mm256_mul_pd(v_dist_sq, v_sqrt);
            __m256d v_ff = _mm256_div_pd(v_mj, v_dist_32);

            v_ax = _mm256_fmadd_pd(v_ff, v_dx, v_ax);
            v_ay = _mm256_fmadd_pd(v_ff, v_dy, v_ay);
            v_az = _mm256_fmadd_pd(v_ff, v_dz, v_az);
        }

        ax[i] = reduce_m256d(v_ax);
        ay[i] = reduce_m256d(v_ay);
        az[i] = reduce_m256d(v_az);
    }
}

// Computes the accelerations for all particles (AoS wrapper)
void compute_accelerations(std::vector<Particle>& particles) {
    int n = particles.size();
    ParticlesSoA soa(n);
    soa.from_aos(particles);
    compute_accelerations_soa(soa);
    soa.to_aos(particles);
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
    
    // Compute initial accelerations
    compute_accelerations(particles);
    
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ParticlesSoA soa(n);
    soa.from_aos(particles);
    
    std::vector<double> old_ax(n), old_ay(n), old_az(n);
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions
        double* __restrict__ x = soa.x.data();
        double* __restrict__ y = soa.y.data();
        double* __restrict__ z = soa.z.data();
        const double* __restrict__ vx = soa.vx.data();
        const double* __restrict__ vy = soa.vy.data();
        const double* __restrict__ vz = soa.vz.data();
        const double* __restrict__ ax = soa.ax.data();
        const double* __restrict__ ay = soa.ay.data();
        const double* __restrict__ az = soa.az.data();
        
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            x[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
            y[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
            z[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
        }
        
        // Save current accelerations as old
        std::copy(soa.ax.begin(), soa.ax.begin() + n, old_ax.begin());
        std::copy(soa.ay.begin(), soa.ay.begin() + n, old_ay.begin());
        std::copy(soa.az.begin(), soa.az.begin() + n, old_az.begin());
        
        // 2. Compute new accelerations
        compute_accelerations_soa(soa);
        
        // 3. Update velocities
        double* __restrict__ n_vx = soa.vx.data();
        double* __restrict__ n_vy = soa.vy.data();
        double* __restrict__ n_vz = soa.vz.data();
        const double* __restrict__ n_ax = soa.ax.data();
        const double* __restrict__ n_ay = soa.ay.data();
        const double* __restrict__ n_az = soa.az.data();
        
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            n_vx[i] += 0.5 * (old_ax[i] + n_ax[i]) * dt;
            n_vy[i] += 0.5 * (old_ay[i] + n_ay[i]) * dt;
            n_vz[i] += 0.5 * (old_az[i] + n_az[i]) * dt;
        }
    }
    
    soa.to_aos(particles);
    
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
