#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <immintrin.h>

const double G = 1.0;
const double SOFTENING_SQ = 0.25;

struct Particle {
    int id;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

// Helper for aligned allocation
float* allocate_aligned_float(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size * sizeof(float)) != 0) {
        std::cerr << "Memory allocation failed!" << std::endl;
        exit(1);
    }
    return static_cast<float*>(ptr);
}

// Horizontal sum of __m256 elements
inline float hsum_8(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v4 = _mm_add_ps(vlow, vhigh);
    __m128 v2 = _mm_add_ps(v4, _mm_shuffle_ps(v4, v4, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 v1 = _mm_add_ps(v2, _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtss_f32(v1);
}

// Highly optimized AVX2 acceleration calculation helper
void compute_accelerations_soa(
    int n,
    int n_pad,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ mass,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az
) {
    const float* __restrict__ x_al = (const float*)__builtin_assume_aligned(x, 64);
    const float* __restrict__ y_al = (const float*)__builtin_assume_aligned(y, 64);
    const float* __restrict__ z_al = (const float*)__builtin_assume_aligned(z, 64);
    const float* __restrict__ mass_al = (const float*)__builtin_assume_aligned(mass, 64);
    float* __restrict__ ax_al = (float*)__builtin_assume_aligned(ax, 64);
    float* __restrict__ ay_al = (float*)__builtin_assume_aligned(ay, 64);
    float* __restrict__ az_al = (float*)__builtin_assume_aligned(az, 64);

    __m256 softening_v = _mm256_set1_ps(SOFTENING_SQ);
    __m256 half_v = _mm256_set1_ps(0.5f);
    __m256 three_v = _mm256_set1_ps(3.0f);

    for (int i = 0; i < n; ++i) {
        __m256 xi_v = _mm256_set1_ps(x_al[i]);
        __m256 yi_v = _mm256_set1_ps(y_al[i]);
        __m256 zi_v = _mm256_set1_ps(z_al[i]);

        __m256 ax_accum = _mm256_setzero_ps();
        __m256 ay_accum = _mm256_setzero_ps();
        __m256 az_accum = _mm256_setzero_ps();

        for (int j = 0; j < n_pad; j += 16) {
            __m256 xj_v0 = _mm256_load_ps(&x_al[j]);
            __m256 yj_v0 = _mm256_load_ps(&y_al[j]);
            __m256 zj_v0 = _mm256_load_ps(&z_al[j]);
            __m256 mj_v0 = _mm256_load_ps(&mass_al[j]);

            __m256 xj_v1 = _mm256_load_ps(&x_al[j+8]);
            __m256 yj_v1 = _mm256_load_ps(&y_al[j+8]);
            __m256 zj_v1 = _mm256_load_ps(&z_al[j+8]);
            __m256 mj_v1 = _mm256_load_ps(&mass_al[j+8]);

            __m256 dx0 = _mm256_sub_ps(xj_v0, xi_v);
            __m256 dy0 = _mm256_sub_ps(yj_v0, yi_v);
            __m256 dz0 = _mm256_sub_ps(zj_v0, zi_v);

            __m256 dx1 = _mm256_sub_ps(xj_v1, xi_v);
            __m256 dy1 = _mm256_sub_ps(yj_v1, yi_v);
            __m256 dz1 = _mm256_sub_ps(zj_v1, zi_v);

            __m256 dist_sq0 = _mm256_fmadd_ps(dz0, dz0, softening_v);
            dist_sq0 = _mm256_fmadd_ps(dy0, dy0, dist_sq0);
            dist_sq0 = _mm256_fmadd_ps(dx0, dx0, dist_sq0);

            __m256 dist_sq1 = _mm256_fmadd_ps(dz1, dz1, softening_v);
            dist_sq1 = _mm256_fmadd_ps(dy1, dy1, dist_sq1);
            dist_sq1 = _mm256_fmadd_ps(dx1, dx1, dist_sq1);

            __m256 inv_sqrt0 = _mm256_rsqrt_ps(dist_sq0);
            __m256 inv_sqrt1 = _mm256_rsqrt_ps(dist_sq1);

            __m256 y2_0 = _mm256_mul_ps(inv_sqrt0, inv_sqrt0);
            __m256 y2_1 = _mm256_mul_ps(inv_sqrt1, inv_sqrt1);

            __m256 nr_term0 = _mm256_fnmadd_ps(dist_sq0, y2_0, three_v);
            __m256 nr_term1 = _mm256_fnmadd_ps(dist_sq1, y2_1, three_v);

            inv_sqrt0 = _mm256_mul_ps(_mm256_mul_ps(half_v, inv_sqrt0), nr_term0);
            inv_sqrt1 = _mm256_mul_ps(_mm256_mul_ps(half_v, inv_sqrt1), nr_term1);

            __m256 inv_dist_3_0 = _mm256_mul_ps(_mm256_mul_ps(inv_sqrt0, inv_sqrt0), inv_sqrt0);
            __m256 inv_dist_3_1 = _mm256_mul_ps(_mm256_mul_ps(inv_sqrt1, inv_sqrt1), inv_sqrt1);

            __m256 force_factor0 = _mm256_mul_ps(mj_v0, inv_dist_3_0);
            __m256 force_factor1 = _mm256_mul_ps(mj_v1, inv_dist_3_1);

            ax_accum = _mm256_fmadd_ps(force_factor0, dx0, ax_accum);
            ay_accum = _mm256_fmadd_ps(force_factor0, dy0, ay_accum);
            az_accum = _mm256_fmadd_ps(force_factor0, dz0, az_accum);

            ax_accum = _mm256_fmadd_ps(force_factor1, dx1, ax_accum);
            ay_accum = _mm256_fmadd_ps(force_factor1, dy1, ay_accum);
            az_accum = _mm256_fmadd_ps(force_factor1, dz1, az_accum);
        }

        ax_al[i] = hsum_8(ax_accum);
        ay_al[i] = hsum_8(ay_accum);
        az_al[i] = hsum_8(az_accum);
    }
}

// Calculates the total Hamiltonian energy (Kinetic + Potential) in double precision
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
    
    // Load initial state CSV using fast C I/O
    FILE* infile = fopen(input_file.c_str(), "r");
    if (!infile) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }
    
    // Skip header line
    char header[512];
    if (!fgets(header, sizeof(header), infile)) {
        // ignore
    }
    
    Particle p;
    while (fscanf(infile, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", 
                  &p.id, &p.mass, &p.x, &p.y, &p.z, &p.vx, &p.vy, &p.vz) == 8) {
        p.ax = p.ay = p.az = 0.0;
        particles.push_back(p);
    }
    fclose(infile);
    
    int n = particles.size();
    
    // Pad n to multiple of 16
    int n_pad = (n + 15) & ~15;
    
    // Allocate 64-byte aligned memory for SoA arrays
    float* mass = allocate_aligned_float(n_pad);
    float* x = allocate_aligned_float(n_pad);
    float* y = allocate_aligned_float(n_pad);
    float* z = allocate_aligned_float(n_pad);
    float* vx = allocate_aligned_float(n_pad);
    float* vy = allocate_aligned_float(n_pad);
    float* vz = allocate_aligned_float(n_pad);
    float* ax = allocate_aligned_float(n_pad);
    float* ay = allocate_aligned_float(n_pad);
    float* az = allocate_aligned_float(n_pad);

    // Initialize arrays
    for (int i = 0; i < n; ++i) {
        mass[i] = (float)particles[i].mass;
        x[i] = (float)particles[i].x;
        y[i] = (float)particles[i].y;
        z[i] = (float)particles[i].z;
        vx[i] = (float)particles[i].vx;
        vy[i] = (float)particles[i].vy;
        vz[i] = (float)particles[i].vz;
        ax[i] = 0.0f;
        ay[i] = 0.0f;
        az[i] = 0.0f;
    }
    // Pad elements
    for (int i = n; i < n_pad; ++i) {
        mass[i] = 0.0f;
        x[i] = 0.0f;
        y[i] = 0.0f;
        z[i] = 0.0f;
        vx[i] = 0.0f;
        vy[i] = 0.0f;
        vz[i] = 0.0f;
        ax[i] = 0.0f;
        ay[i] = 0.0f;
        az[i] = 0.0f;
    }
    
    // Compute initial accelerations
    compute_accelerations_soa(n, n_pad, x, y, z, mass, ax, ay, az);
    
    // Copy back to particles for initial energy check
    for (int i = 0; i < n; ++i) {
        particles[i].ax = ax[i];
        particles[i].ay = ay[i];
        particles[i].az = az[i];
    }
    
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    float dt_f = (float)dt;
    float dt_half = 0.5f * dt_f;
    
    // 1. Initial half-step velocity update
    for (int i = 0; i < n; ++i) {
        vx[i] += ax[i] * dt_half;
        vy[i] += ay[i] * dt_half;
        vz[i] += az[i] * dt_half;
    }
    
    // 2. Main simulation loop (Velocity Verlet)
    // Step 0
    for (int i = 0; i < n; ++i) {
        x[i] += vx[i] * dt_f;
        y[i] += vy[i] * dt_f;
        z[i] += vz[i] * dt_f;
    }
    compute_accelerations_soa(n, n_pad, x, y, z, mass, ax, ay, az);
    
    // Steps 1 to num_steps - 1
    for (int step = 1; step < num_steps; ++step) {
        for (int i = 0; i < n; ++i) {
            vx[i] += ax[i] * dt_f;
            vy[i] += ay[i] * dt_f;
            vz[i] += az[i] * dt_f;
            x[i] += vx[i] * dt_f;
            y[i] += vy[i] * dt_f;
            z[i] += vz[i] * dt_f;
        }
        compute_accelerations_soa(n, n_pad, x, y, z, mass, ax, ay, az);
    }
    
    // 3. Final half-step velocity update
    for (int i = 0; i < n; ++i) {
        vx[i] += ax[i] * dt_half;
        vy[i] += ay[i] * dt_half;
        vz[i] += az[i] * dt_half;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Copy back to particles
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
    
    // Write final state CSV using fast C I/O
    FILE* outfile = fopen(output_file.c_str(), "w");
    if (!outfile) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }
    char write_buf[65536];
    setvbuf(outfile, write_buf, _IOFBF, sizeof(write_buf));
    
    fprintf(outfile, "id,mass,x,y,z,vx,vy,vz\n");
    for (const auto& p : particles) {
        fprintf(outfile, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                p.id, p.mass, p.x, p.y, p.z, p.vx, p.vy, p.vz);
    }
    fclose(outfile);
    
    // Free aligned memory
    free(mass); free(x); free(y); free(z);
    free(vx); free(vy); free(vz);
    free(ax); free(ay); free(az);
    
    return 0;
}
