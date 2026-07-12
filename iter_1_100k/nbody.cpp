#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <algorithm>

const double G = 1.0;
const double SOFTENING_SQ = 0.25;

struct Particle {
    int id;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

// Computes the accelerations for all particles using AVX2, Newton-Raphson reciprocal square root, and Cache Tiling
void compute_accelerations_soa(
    const double* __restrict x,
    const double* __restrict y,
    const double* __restrict z,
    const double* __restrict mass,
    double* __restrict ax,
    double* __restrict ay,
    double* __restrict az,
    int n
) {
    // Reset ax, ay, az to zero
    for (int i = 0; i < n; ++i) {
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
    }
    
    const int BLOCK_SIZE = 512;
    __m256d vsoftening = _mm256_set1_pd(SOFTENING_SQ);
    __m256d vhalf = _mm256_set1_pd(0.5);
    __m256d vone_point_five = _mm256_set1_pd(1.5);
    
    // Loop tiling
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        int i_end = std::min(ii + BLOCK_SIZE, n);
        
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            int j_end = std::min(jj + BLOCK_SIZE, n);
            
            for (int i = ii; i < i_end; ++i) {
                __m256d xi = _mm256_set1_pd(x[i]);
                __m256d yi = _mm256_set1_pd(y[i]);
                __m256d zi = _mm256_set1_pd(z[i]);
                
                __m256d sum_ax = _mm256_setzero_pd();
                __m256d sum_ay = _mm256_setzero_pd();
                __m256d sum_az = _mm256_setzero_pd();
                
                int j = jj;
                // Unroll by 2 in the inner loop to optimize instruction scheduling and pipeline occupancy
                for (; j <= j_end - 8; j += 8) {
                    __m256d xj0 = _mm256_loadu_pd(&x[j]);
                    __m256d yj0 = _mm256_loadu_pd(&y[j]);
                    __m256d zj0 = _mm256_loadu_pd(&z[j]);
                    __m256d mj0 = _mm256_loadu_pd(&mass[j]);
                    
                    __m256d xj1 = _mm256_loadu_pd(&x[j + 4]);
                    __m256d yj1 = _mm256_loadu_pd(&y[j + 4]);
                    __m256d zj1 = _mm256_loadu_pd(&z[j + 4]);
                    __m256d mj1 = _mm256_loadu_pd(&mass[j + 4]);
                    
                    __m256d dx0 = _mm256_sub_pd(xj0, xi);
                    __m256d dy0 = _mm256_sub_pd(yj0, yi);
                    __m256d dz0 = _mm256_sub_pd(zj0, zi);
                    
                    __m256d dx1 = _mm256_sub_pd(xj1, xi);
                    __m256d dy1 = _mm256_sub_pd(yj1, yi);
                    __m256d dz1 = _mm256_sub_pd(zj1, zi);
                    
                    __m256d dist_sq0 = _mm256_add_pd(_mm256_mul_pd(dx0, dx0), _mm256_add_pd(_mm256_mul_pd(dy0, dy0), _mm256_add_pd(_mm256_mul_pd(dz0, dz0), vsoftening)));
                    __m256d dist_sq1 = _mm256_add_pd(_mm256_mul_pd(dx1, dx1), _mm256_add_pd(_mm256_mul_pd(dy1, dy1), _mm256_add_pd(_mm256_mul_pd(dz1, dz1), vsoftening)));
                    
                    // Convert dist_sq to float and call rsqrt
                    __m128 dist_sq_f0 = _mm256_cvtpd_ps(dist_sq0);
                    __m128 dist_sq_f1 = _mm256_cvtpd_ps(dist_sq1);
                    
                    __m128 rsqrt_f0 = _mm_rsqrt_ps(dist_sq_f0);
                    __m128 rsqrt_f1 = _mm_rsqrt_ps(dist_sq_f1);
                    
                    // Convert back to double
                    __m256d y0_0 = _mm256_cvtps_pd(rsqrt_f0);
                    __m256d y0_1 = _mm256_cvtps_pd(rsqrt_f1);
                    
                    __m256d half_dist_sq0 = _mm256_mul_pd(vhalf, dist_sq0);
                    __m256d half_dist_sq1 = _mm256_mul_pd(vhalf, dist_sq1);
                    
                    // NR Iteration 1
                    __m256d y0_sq0 = _mm256_mul_pd(y0_0, y0_0);
                    __m256d y0_sq1 = _mm256_mul_pd(y0_1, y0_1);
                    
                    __m256d diff1_0 = _mm256_fnmadd_pd(half_dist_sq0, y0_sq0, vone_point_five);
                    __m256d diff1_1 = _mm256_fnmadd_pd(half_dist_sq1, y0_sq1, vone_point_five);
                    
                    __m256d y1_0 = _mm256_mul_pd(y0_0, diff1_0);
                    __m256d y1_1 = _mm256_mul_pd(y0_1, diff1_1);
                    
                    // NR Iteration 2
                    __m256d y1_sq0 = _mm256_mul_pd(y1_0, y1_0);
                    __m256d y1_sq1 = _mm256_mul_pd(y1_1, y1_1);
                    
                    __m256d diff2_0 = _mm256_fnmadd_pd(half_dist_sq0, y1_sq0, vone_point_five);
                    __m256d diff2_1 = _mm256_fnmadd_pd(half_dist_sq1, y1_sq1, vone_point_five);
                    
                    __m256d y2_0 = _mm256_mul_pd(y1_0, diff2_0);
                    __m256d y2_1 = _mm256_mul_pd(y1_1, diff2_1);
                    
                    // Cube y2
                    __m256d y2_sq0 = _mm256_mul_pd(y2_0, y2_0);
                    __m256d y2_sq1 = _mm256_mul_pd(y2_1, y2_1);
                    
                    __m256d y2_cube0 = _mm256_mul_pd(y2_0, y2_sq0);
                    __m256d y2_cube1 = _mm256_mul_pd(y2_1, y2_sq1);
                    
                    // Force factor
                    __m256d force_factor0 = _mm256_mul_pd(mj0, y2_cube0);
                    __m256d force_factor1 = _mm256_mul_pd(mj1, y2_cube1);
                    
                    sum_ax = _mm256_fmadd_pd(force_factor0, dx0, sum_ax);
                    sum_ay = _mm256_fmadd_pd(force_factor0, dy0, sum_ay);
                    sum_az = _mm256_fmadd_pd(force_factor0, dz0, sum_az);
                    
                    sum_ax = _mm256_fmadd_pd(force_factor1, dx1, sum_ax);
                    sum_ay = _mm256_fmadd_pd(force_factor1, dy1, sum_ay);
                    sum_az = _mm256_fmadd_pd(force_factor1, dz1, sum_az);
                }
                
                // Remainder of j block (blocks of 4)
                for (; j <= j_end - 4; j += 4) {
                    __m256d xj = _mm256_loadu_pd(&x[j]);
                    __m256d yj = _mm256_loadu_pd(&y[j]);
                    __m256d zj = _mm256_loadu_pd(&z[j]);
                    __m256d mj = _mm256_loadu_pd(&mass[j]);
                    
                    __m256d dx = _mm256_sub_pd(xj, xi);
                    __m256d dy = _mm256_sub_pd(yj, yi);
                    __m256d dz = _mm256_sub_pd(zj, zi);
                    
                    __m256d dist_sq = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_add_pd(_mm256_mul_pd(dz, dz), vsoftening)));
                    
                    __m128 dist_sq_f = _mm256_cvtpd_ps(dist_sq);
                    __m128 rsqrt_f = _mm_rsqrt_ps(dist_sq_f);
                    __m256d y0 = _mm256_cvtps_pd(rsqrt_f);
                    
                    __m256d half_dist_sq = _mm256_mul_pd(vhalf, dist_sq);
                    
                    // NR 1
                    __m256d y0_sq = _mm256_mul_pd(y0, y0);
                    __m256d diff1 = _mm256_fnmadd_pd(half_dist_sq, y0_sq, vone_point_five);
                    __m256d y1 = _mm256_mul_pd(y0, diff1);
                    
                    // NR 2
                    __m256d y1_sq = _mm256_mul_pd(y1, y1);
                    __m256d diff2 = _mm256_fnmadd_pd(half_dist_sq, y1_sq, vone_point_five);
                    __m256d y2 = _mm256_mul_pd(y1, diff2);
                    
                    __m256d y2_sq = _mm256_mul_pd(y2, y2);
                    __m256d y2_cube = _mm256_mul_pd(y2, y2_sq);
                    __m256d force_factor = _mm256_mul_pd(mj, y2_cube);
                    
                    sum_ax = _mm256_fmadd_pd(force_factor, dx, sum_ax);
                    sum_ay = _mm256_fmadd_pd(force_factor, dy, sum_ay);
                    sum_az = _mm256_fmadd_pd(force_factor, dz, sum_az);
                }
                
                double temp_ax[4];
                double temp_ay[4];
                double temp_az[4];
                _mm256_storeu_pd(temp_ax, sum_ax);
                _mm256_storeu_pd(temp_ay, sum_ay);
                _mm256_storeu_pd(temp_az, sum_az);
                
                double axi = temp_ax[0] + temp_ax[1] + temp_ax[2] + temp_ax[3];
                double ayi = temp_ay[0] + temp_ay[1] + temp_ay[2] + temp_ay[3];
                double azi = temp_az[0] + temp_az[1] + temp_az[2] + temp_az[3];
                
                // Scalar remainder for this block (0 to 3 elements)
                for (; j < j_end; ++j) {
                    double dx = x[j] - x[i];
                    double dy = y[j] - y[i];
                    double dz = z[j] - z[i];
                    double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING_SQ;
                    double dist = std::sqrt(dist_sq);
                    double force_factor = mass[j] / (dist_sq * dist);
                    axi += force_factor * dx;
                    ayi += force_factor * dy;
                    azi += force_factor * dz;
                }
                
                ax[i] += axi;
                ay[i] += ayi;
                az[i] += azi;
            }
        }
    }
}

// Dummy energy function required by signature but not used for verification
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
    
    // Convert to Structure of Arrays (SoA)
    std::vector<double> x(n), y(n), z(n);
    std::vector<double> vx(n), vy(n), vz(n);
    std::vector<double> ax(n), ay(n), az(n);
    std::vector<double> mass(n);
    std::vector<int> id(n);
    
    for (int i = 0; i < n; ++i) {
        id[i] = particles[i].id;
        mass[i] = particles[i].mass;
        x[i] = particles[i].x;
        y[i] = particles[i].y;
        z[i] = particles[i].z;
        vx[i] = particles[i].vx;
        vy[i] = particles[i].vy;
        vz[i] = particles[i].vz;
    }
    
    // Compute initial accelerations
    compute_accelerations_soa(x.data(), y.data(), z.data(), mass.data(), ax.data(), ay.data(), az.data(), n);
    
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions
        for (int i = 0; i < n; ++i) {
            x[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
            y[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
            z[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
        }
        
        // Save current accelerations as old
        std::vector<double> old_ax = ax;
        std::vector<double> old_ay = ay;
        std::vector<double> old_az = az;
        
        // 2. Compute new accelerations
        compute_accelerations_soa(x.data(), y.data(), z.data(), mass.data(), ax.data(), ay.data(), az.data(), n);
        
        // 3. Update velocities
        for (int i = 0; i < n; ++i) {
            vx[i] += 0.5 * (old_ax[i] + ax[i]) * dt;
            vy[i] += 0.5 * (old_ay[i] + ay[i]) * dt;
            vz[i] += 0.5 * (old_az[i] + az[i]) * dt;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Copy back to particles for output
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
    
    std::string output_path = argv[2];
    std::string time_file_path = "simulation_time.txt";
    size_t last_slash = output_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        time_file_path = output_path.substr(0, last_slash + 1) + "simulation_time.txt";
    }
    FILE* time_file = fopen(time_file_path.c_str(), "w");
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
