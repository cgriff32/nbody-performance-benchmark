#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
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

// Highly optimized AVX2 Hamiltonian energy calculation
double calculate_energy_soa(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z,
    const std::vector<double>& vx,
    const std::vector<double>& vy,
    const std::vector<double>& vz,
    const std::vector<double>& mass,
    int n
) {
    double kinetic_energy = 0.0;
    for (int i = 0; i < n; ++i) {
        double v_sq = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        kinetic_energy += 0.5 * mass[i] * v_sq;
    }
    
    double potential_energy_sum = 0.0;
    __m256d softening_v = _mm256_set1_pd(SOFTENING_SQ);
    
    for (int i = 0; i < n; ++i) {
        double px = x[i];
        double py = y[i];
        double pz = z[i];
        
        __m256d px_v = _mm256_set1_pd(px);
        __m256d py_v = _mm256_set1_pd(py);
        __m256d pz_v = _mm256_set1_pd(pz);
        
        __m256d sum_v = _mm256_setzero_pd();
        
        for (int j = 0; j < n; j += 4) {
            __m256d xj_v = _mm256_loadu_pd(&x[j]);
            __m256d yj_v = _mm256_loadu_pd(&y[j]);
            __m256d zj_v = _mm256_loadu_pd(&z[j]);
            __m256d mj_v = _mm256_loadu_pd(&mass[j]);
            
            __m256d dx_v = _mm256_sub_pd(xj_v, px_v);
            __m256d dy_v = _mm256_sub_pd(yj_v, py_v);
            __m256d dz_v = _mm256_sub_pd(zj_v, pz_v);
            
            __m256d dist_sq_v = _mm256_fmadd_pd(dz_v, dz_v, softening_v);
            dist_sq_v = _mm256_fmadd_pd(dy_v, dy_v, dist_sq_v);
            dist_sq_v = _mm256_fmadd_pd(dx_v, dx_v, dist_sq_v);
            
            __m256d sqrt_dist_sq_v = _mm256_sqrt_pd(dist_sq_v);
            __m256d term_v = _mm256_div_pd(mj_v, sqrt_dist_sq_v);
            
            sum_v = _mm256_add_pd(sum_v, term_v);
        }
        
        __m128d sum_h = _mm256_extractf128_pd(sum_v, 1);
        __m128d sum_l = _mm256_extractf128_pd(sum_v, 0);
        __m128d sum_sum = _mm_add_pd(sum_h, sum_l);
        double sum_arr[2];
        _mm_storeu_pd(sum_arr, sum_sum);
        double sum_i = sum_arr[0] + sum_arr[1];
        
        // Subtract self-interaction: mass[i] / sqrt(SOFTENING_SQ)
        // Since SOFTENING_SQ = 0.25, sqrt(0.25) = 0.5, so term is mass[i] / 0.5 = 2.0 * mass[i]
        sum_i -= 2.0 * mass[i];
        
        potential_energy_sum += mass[i] * sum_i;
    }
    
    double potential_energy = -0.5 * G * potential_energy_sum;
    return kinetic_energy + potential_energy;
}

// Highly optimized AVX2 acceleration calculation helper
void compute_accelerations_soa(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z,
    const std::vector<double>& mass,
    std::vector<double>& ax,
    std::vector<double>& ay,
    std::vector<double>& az,
    int n,
    __m256d half,
    __m256d three
) {
    for (int i = 0; i < n; ++i) {
        double px = x[i];
        double py = y[i];
        double pz = z[i];
        
        __m256d px_v = _mm256_set1_pd(px);
        __m256d py_v = _mm256_set1_pd(py);
        __m256d pz_v = _mm256_set1_pd(pz);
        __m256d softening_v = _mm256_set1_pd(SOFTENING_SQ);
        
        __m256d ax_v = _mm256_setzero_pd();
        __m256d ay_v = _mm256_setzero_pd();
        __m256d az_v = _mm256_setzero_pd();
        
        for (int j = 0; j < n; j += 4) {
            __m256d xj_v = _mm256_loadu_pd(&x[j]);
            __m256d yj_v = _mm256_loadu_pd(&y[j]);
            __m256d zj_v = _mm256_loadu_pd(&z[j]);
            __m256d mj_v = _mm256_loadu_pd(&mass[j]);
            
            __m256d dx_v = _mm256_sub_pd(xj_v, px_v);
            __m256d dy_v = _mm256_sub_pd(yj_v, py_v);
            __m256d dz_v = _mm256_sub_pd(zj_v, pz_v);
            
            // dx*dx + dy*dy + dz*dz + SOFTENING_SQ using 3 FMAs:
            __m256d dist_sq_v = _mm256_fmadd_pd(dz_v, dz_v, softening_v);
            dist_sq_v = _mm256_fmadd_pd(dy_v, dy_v, dist_sq_v);
            dist_sq_v = _mm256_fmadd_pd(dx_v, dx_v, dist_sq_v);
            
            // 1/sqrt(dist_sq) via single-precision rsqrt + conversions
            __m128 dist_sq_f = _mm256_cvtpd_ps(dist_sq_v);
            __m128 inv_sqrt_f = _mm_rsqrt_ps(dist_sq_f);
            __m256d y = _mm256_cvtps_pd(inv_sqrt_f);
            
            // Newton-Raphson iteration: y = 0.5 * y * (3.0 - dist_sq * y * y)
            __m256d y2 = _mm256_mul_pd(y, y);
            __m256d nr1 = _mm256_fnmadd_pd(dist_sq_v, y2, three);
            y = _mm256_mul_pd(_mm256_mul_pd(half, y), nr1);
            
            // inv_dist_3 = y^3
            __m256d inv_dist_3 = _mm256_mul_pd(_mm256_mul_pd(y, y), y);
            
            // force_factor = mass * inv_dist_3
            __m256d force_factor_v = _mm256_mul_pd(mj_v, inv_dist_3);
            
            ax_v = _mm256_fmadd_pd(force_factor_v, dx_v, ax_v);
            ay_v = _mm256_fmadd_pd(force_factor_v, dy_v, ay_v);
            az_v = _mm256_fmadd_pd(force_factor_v, dz_v, az_v);
        }
        
        // Horizontal reduction
        __m128d ax_h = _mm256_extractf128_pd(ax_v, 1);
        __m128d ax_l = _mm256_extractf128_pd(ax_v, 0);
        __m128d ax_sum = _mm_add_pd(ax_h, ax_l);
        double ax_arr[2];
        _mm_storeu_pd(ax_arr, ax_sum);
        ax[i] = ax_arr[0] + ax_arr[1];
        
        __m128d ay_h = _mm256_extractf128_pd(ay_v, 1);
        __m128d ay_l = _mm256_extractf128_pd(ay_v, 0);
        __m128d ay_sum = _mm_add_pd(ay_h, ay_l);
        double ay_arr[2];
        _mm_storeu_pd(ay_arr, ay_sum);
        ay[i] = ay_arr[0] + ay_arr[1];
        
        __m128d az_h = _mm256_extractf128_pd(az_v, 1);
        __m128d az_l = _mm256_extractf128_pd(az_v, 0);
        __m128d az_sum = _mm_add_pd(az_h, az_l);
        double az_arr[2];
        _mm_storeu_pd(az_arr, az_sum);
        az[i] = az_arr[0] + az_arr[1];
    }
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
    std::cout << "Loaded " << n << " particles." << std::endl;
    
    // Copy to SoA layout
    std::vector<double> x(n), y(n), z(n);
    std::vector<double> vx(n), vy(n), vz(n);
    std::vector<double> ax(n), ay(n), az(n);
    std::vector<double> mass(n);
    for (int i = 0; i < n; ++i) {
        x[i] = particles[i].x;
        y[i] = particles[i].y;
        z[i] = particles[i].z;
        vx[i] = particles[i].vx;
        vy[i] = particles[i].vy;
        vz[i] = particles[i].vz;
        ax[i] = particles[i].ax;
        ay[i] = particles[i].ay;
        az[i] = particles[i].az;
        mass[i] = particles[i].mass;
    }
    
    // Constants for Newton-Raphson
    __m256d half = _mm256_set1_pd(0.5);
    __m256d three = _mm256_set1_pd(3.0);
    
    // Compute initial accelerations using the optimized AVX2 SoA code
    compute_accelerations_soa(x, y, z, mass, ax, ay, az, n, half, three);
    
    double initial_energy = calculate_energy_soa(x, y, z, vx, vy, vz, mass, n);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> old_ax(n), old_ay(n), old_az(n);
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions
        for (int i = 0; i < n; ++i) {
            x[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
            y[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
            z[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
        }
        
        // Save current accelerations as old
        for (int i = 0; i < n; ++i) {
            old_ax[i] = ax[i];
            old_ay[i] = ay[i];
            old_az[i] = az[i];
        }
        
        // 2. Compute new accelerations
        compute_accelerations_soa(x, y, z, mass, ax, ay, az, n, half, three);
        
        // 3. Update velocities
        for (int i = 0; i < n; ++i) {
            vx[i] += 0.5 * (old_ax[i] + ax[i]) * dt;
            vy[i] += 0.5 * (old_ay[i] + ay[i]) * dt;
            vz[i] += 0.5 * (old_az[i] + az[i]) * dt;
        }
    }
    
    // Copy SoA back to particles
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
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    double final_energy = calculate_energy_soa(x, y, z, vx, vy, vz, mass, n);
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
    
    return 0;
}
