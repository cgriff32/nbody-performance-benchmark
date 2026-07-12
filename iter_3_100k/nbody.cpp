#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>

const float G = 1.0f;
const float SOFTENING_SQ = 0.25f;

// Computes the accelerations for all particles using AVX2 + FMA
void compute_accelerations(
    const float* __restrict__ mass,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    int n
) {
    // Reset accelerations to zero
    for (int i = 0; i < n; ++i) {
        ax[i] = 0.0f;
        ay[i] = 0.0f;
        az[i] = 0.0f;
    }

    __m256 softening_v = _mm256_set1_ps(SOFTENING_SQ);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one_point_five = _mm256_set1_ps(1.5f);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        __m256 xi_v = _mm256_set1_ps(x[i]);
        __m256 yi_v = _mm256_set1_ps(y[i]);
        __m256 zi_v = _mm256_set1_ps(z[i]);

        __m256 ax_v = _mm256_setzero_ps();
        __m256 ay_v = _mm256_setzero_ps();
        __m256 az_v = _mm256_setzero_ps();

        for (int j = 0; j < n; j += 8) {
            __m256 xj_v = _mm256_load_ps(&x[j]);
            __m256 yj_v = _mm256_load_ps(&y[j]);
            __m256 zj_v = _mm256_load_ps(&z[j]);
            __m256 mj_v = _mm256_load_ps(&mass[j]);

            __m256 dx = _mm256_sub_ps(xj_v, xi_v);
            __m256 dy = _mm256_sub_ps(yj_v, yi_v);
            __m256 dz = _mm256_sub_ps(zj_v, zi_v);

            // dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING_SQ
            __m256 dist_sq = softening_v;
            dist_sq = _mm256_fmadd_ps(dx, dx, dist_sq);
            dist_sq = _mm256_fmadd_ps(dy, dy, dist_sq);
            dist_sq = _mm256_fmadd_ps(dz, dz, dist_sq);

            // inv_dist = rsqrt(dist_sq)
            __m256 inv_dist = _mm256_rsqrt_ps(dist_sq);

            // Newton-Raphson step: y = y * (1.5 - 0.5 * x * y * y)
            __m256 inv_dist_sq = _mm256_mul_ps(inv_dist, inv_dist);
            __m256 inner = _mm256_fnmadd_ps(half, _mm256_mul_ps(dist_sq, inv_dist_sq), one_point_five);
            inv_dist = _mm256_mul_ps(inv_dist, inner);

            // force_factor = mass * inv_dist^3
            __m256 inv_dist3 = _mm256_mul_ps(_mm256_mul_ps(inv_dist, inv_dist), inv_dist);
            __m256 force_factor = _mm256_mul_ps(mj_v, inv_dist3);

            // ax_v += force_factor * dx
            ax_v = _mm256_fmadd_ps(force_factor, dx, ax_v);
            ay_v = _mm256_fmadd_ps(force_factor, dy, ay_v);
            az_v = _mm256_fmadd_ps(force_factor, dz, az_v);
        }

        // Horizontal sum of ax_v, ay_v, az_v
        alignas(32) float temp_x[8];
        alignas(32) float temp_y[8];
        alignas(32) float temp_z[8];
        _mm256_store_ps(temp_x, ax_v);
        _mm256_store_ps(temp_y, ay_v);
        _mm256_store_ps(temp_z, az_v);

        ax[i] = temp_x[0] + temp_x[1] + temp_x[2] + temp_x[3] + temp_x[4] + temp_x[5] + temp_x[6] + temp_x[7];
        ay[i] = temp_y[0] + temp_y[1] + temp_y[2] + temp_y[3] + temp_y[4] + temp_y[5] + temp_y[6] + temp_y[7];
        az[i] = temp_z[0] + temp_z[1] + temp_z[2] + temp_z[3] + temp_z[4] + temp_z[5] + temp_z[6] + temp_z[7];
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
    float dt = std::stof(argv[4]);
    
    std::vector<int> ids;
    std::vector<double> mass_d;
    std::vector<double> x_d, y_d, z_d;
    std::vector<double> vx_d, vy_d, vz_d;
    
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
        
        int id; double m, px, py, pz, pvx, pvy, pvz;
        std::getline(ss, val, ','); id = std::stoi(val);
        std::getline(ss, val, ','); m = std::stod(val);
        std::getline(ss, val, ','); px = std::stod(val);
        std::getline(ss, val, ','); py = std::stod(val);
        std::getline(ss, val, ','); pz = std::stod(val);
        std::getline(ss, val, ','); pvx = std::stod(val);
        std::getline(ss, val, ','); pvy = std::stod(val);
        std::getline(ss, val, ','); pvz = std::stod(val);
        
        ids.push_back(id);
        mass_d.push_back(m);
        x_d.push_back(px);
        y_d.push_back(py);
        z_d.push_back(pz);
        vx_d.push_back(pvx);
        vy_d.push_back(pvy);
        vz_d.push_back(pvz);
    }
    infile.close();
    
    int n = ids.size();
    std::cout << "Loaded " << n << " particles." << std::endl;
    
    // Pad arrays to multiple of 8 for SIMD alignment
    int n_padded = ((n + 7) / 8) * 8;
    
    float* mass = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* x = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* y = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* z = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* vx = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* vy = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* vz = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* ax = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* ay = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    float* az = (float*)_mm_malloc(n_padded * sizeof(float), 64);
    
    for (int i = 0; i < n_padded; ++i) {
        if (i < n) {
            mass[i] = (float)mass_d[i];
            x[i] = (float)x_d[i];
            y[i] = (float)y_d[i];
            z[i] = (float)z_d[i];
            vx[i] = (float)vx_d[i];
            vy[i] = (float)vy_d[i];
            vz[i] = (float)vz_d[i];
        } else {
            mass[i] = 0.0f;
            x[i] = 0.0f;
            y[i] = 0.0f;
            z[i] = 0.0f;
            vx[i] = 0.0f;
            vy[i] = 0.0f;
            vz[i] = 0.0f;
        }
        ax[i] = 0.0f;
        ay[i] = 0.0f;
        az[i] = 0.0f;
    }
    
    // Compute initial accelerations
    compute_accelerations(mass, x, y, z, ax, ay, az, n_padded);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions and half-update velocities
        for (int i = 0; i < n_padded; ++i) {
            vx[i] += 0.5f * ax[i] * dt;
            vy[i] += 0.5f * ay[i] * dt;
            vz[i] += 0.5f * az[i] * dt;
            
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
        }
        
        // 2. Compute new accelerations
        compute_accelerations(mass, x, y, z, ax, ay, az, n_padded);
        
        // 3. Complete velocity update
        for (int i = 0; i < n_padded; ++i) {
            vx[i] += 0.5f * ax[i] * dt;
            vy[i] += 0.5f * ay[i] * dt;
            vz[i] += 0.5f * az[i] * dt;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Simulation loop time: " << elapsed.count() << " seconds" << std::endl;
    FILE* time_file = fopen("simulation_time.txt", "w");
    if (time_file) {
        fprintf(time_file, "%.9f\n", elapsed.count());
        fclose(time_file);
    }
    
    // Copy back to double for output
    for (int i = 0; i < n; ++i) {
        x_d[i] = x[i];
        y_d[i] = y[i];
        z_d[i] = z[i];
        vx_d[i] = vx[i];
        vy_d[i] = vy[i];
        vz_d[i] = vz[i];
    }
    
    // Write final state CSV
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }
    
    outfile << std::scientific << std::setprecision(17);
    outfile << "id,mass,x,y,z,vx,vy,vz\n";
    for (int i = 0; i < n; ++i) {
        outfile << ids[i] << "," << mass_d[i] << "," 
                << x_d[i] << "," << y_d[i] << "," << z_d[i] << ","
                << vx_d[i] << "," << vy_d[i] << "," << vz_d[i] << "\n";
    }
    outfile.close();
    
    _mm_free(mass);
    _mm_free(x);
    _mm_free(y);
    _mm_free(z);
    _mm_free(vx);
    _mm_free(vy);
    _mm_free(vz);
    _mm_free(ax);
    _mm_free(ay);
    _mm_free(az);
    
    return 0;
}
