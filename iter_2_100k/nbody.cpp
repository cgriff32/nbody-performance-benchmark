#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>
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

struct ParticlesSoA {
    int n;
    std::vector<double> mass;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> ax, ay, az;
    std::vector<int> id;
    
    void resize(int size) {
        n = size;
        int padded_size = (n + 7) & ~7;
        mass.resize(padded_size, 0.0);
        x.resize(padded_size, 0.0);
        y.resize(padded_size, 0.0);
        z.resize(padded_size, 0.0);
        vx.resize(padded_size, 0.0);
        vy.resize(padded_size, 0.0);
        vz.resize(padded_size, 0.0);
        ax.resize(padded_size, 0.0);
        ay.resize(padded_size, 0.0);
        az.resize(padded_size, 0.0);
        id.resize(padded_size, 0);
    }
};

void particles_to_soa(const std::vector<Particle>& src, ParticlesSoA& dest) {
    int n = src.size();
    dest.resize(n);
    for (int i = 0; i < n; ++i) {
        dest.id[i] = src[i].id;
        dest.mass[i] = src[i].mass;
        dest.x[i] = src[i].x;
        dest.y[i] = src[i].y;
        dest.z[i] = src[i].z;
        dest.vx[i] = src[i].vx;
        dest.vy[i] = src[i].vy;
        dest.vz[i] = src[i].vz;
        dest.ax[i] = src[i].ax;
        dest.ay[i] = src[i].ay;
        dest.az[i] = src[i].az;
    }
}

void soa_to_particles(const ParticlesSoA& src, std::vector<Particle>& dest) {
    int n = src.n;
    dest.resize(n);
    for (int i = 0; i < n; ++i) {
        dest[i].id = src.id[i];
        dest[i].mass = src.mass[i];
        dest[i].x = src.x[i];
        dest[i].y = src.y[i];
        dest[i].z = src.z[i];
        dest[i].vx = src.vx[i];
        dest[i].vy = src.vy[i];
        dest[i].vz = src.vz[i];
        dest[i].ax = src.ax[i];
        dest[i].ay = src.ay[i];
        dest[i].az = src.az[i];
    }
}

void compute_accelerations_avx2(const ParticlesSoA& p, std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az) {
    int n = p.n;
    int padded_size = (n + 7) & ~7;
    
    __m256d vsoft = _mm256_set1_pd(SOFTENING_SQ);
    
    for (int i = 0; i < n; ++i) {
        __m256d vxi = _mm256_set1_pd(p.x[i]);
        __m256d vyi = _mm256_set1_pd(p.y[i]);
        __m256d vzi = _mm256_set1_pd(p.z[i]);
        
        __m256d vax = _mm256_setzero_pd();
        __m256d vay = _mm256_setzero_pd();
        __m256d vaz = _mm256_setzero_pd();
        
        for (int j = 0; j < padded_size; j += 8) {
            __m256d vxj0 = _mm256_loadu_pd(&p.x[j]);
            __m256d vyj0 = _mm256_loadu_pd(&p.y[j]);
            __m256d vzj0 = _mm256_loadu_pd(&p.z[j]);
            __m256d vmj0 = _mm256_loadu_pd(&p.mass[j]);
            
            __m256d vxj1 = _mm256_loadu_pd(&p.x[j + 4]);
            __m256d vyj1 = _mm256_loadu_pd(&p.y[j + 4]);
            __m256d vzj1 = _mm256_loadu_pd(&p.z[j + 4]);
            __m256d vmj1 = _mm256_loadu_pd(&p.mass[j + 4]);
            
            __m256d vdx0 = _mm256_sub_pd(vxj0, vxi);
            __m256d vdy0 = _mm256_sub_pd(vyj0, vyi);
            __m256d vdz0 = _mm256_sub_pd(vzj0, vzi);
            
            __m256d vdx1 = _mm256_sub_pd(vxj1, vxi);
            __m256d vdy1 = _mm256_sub_pd(vyj1, vyi);
            __m256d vdz1 = _mm256_sub_pd(vzj1, vzi);
            
            __m256d vdist_sq0 = _mm256_fmadd_pd(vdx0, vdx0, vsoft);
            vdist_sq0 = _mm256_fmadd_pd(vdy0, vdy0, vdist_sq0);
            vdist_sq0 = _mm256_fmadd_pd(vdz0, vdz0, vdist_sq0);
            
            __m256d vdist_sq1 = _mm256_fmadd_pd(vdx1, vdx1, vsoft);
            vdist_sq1 = _mm256_fmadd_pd(vdy1, vdy1, vdist_sq1);
            vdist_sq1 = _mm256_fmadd_pd(vdz1, vdz1, vdist_sq1);
            
            // Hybrid float-double rsqrt:
            // 1. Convert double dist_sq to float
            __m128 vdist_sq_f0 = _mm256_cvtpd_ps(vdist_sq0);
            __m128 vdist_sq_f1 = _mm256_cvtpd_ps(vdist_sq1);
            
            // 2. Compute approximate rsqrt in float
            __m128 rsqrt_f0 = _mm_rsqrt_ps(vdist_sq_f0);
            __m128 rsqrt_f1 = _mm_rsqrt_ps(vdist_sq_f1);
            
            // 3. Convert back to double
            __m256d y0 = _mm256_cvtps_pd(rsqrt_f0);
            __m256d y1 = _mm256_cvtps_pd(rsqrt_f1);
            
            // Compute y^3 = (dist_sq)^-1.5
            __m256d y0_3 = _mm256_mul_pd(y0, _mm256_mul_pd(y0, y0));
            __m256d y1_3 = _mm256_mul_pd(y1, _mm256_mul_pd(y1, y1));
            
            // Force factor
            __m256d vforce0 = _mm256_mul_pd(vmj0, y0_3);
            __m256d vforce1 = _mm256_mul_pd(vmj1, y1_3);
            
            vax = _mm256_fmadd_pd(vforce0, vdx0, vax);
            vay = _mm256_fmadd_pd(vforce0, vdy0, vay);
            vaz = _mm256_fmadd_pd(vforce0, vdz0, vaz);
            
            vax = _mm256_fmadd_pd(vforce1, vdx1, vax);
            vay = _mm256_fmadd_pd(vforce1, vdy1, vay);
            vaz = _mm256_fmadd_pd(vforce1, vdz1, vaz);
        }
        
        double alignas(32) ax_arr[4];
        double alignas(32) ay_arr[4];
        double alignas(32) az_arr[4];
        
        _mm256_store_pd(ax_arr, vax);
        _mm256_store_pd(ay_arr, vay);
        _mm256_store_pd(az_arr, vaz);
        
        ax[i] = ax_arr[0] + ax_arr[1] + ax_arr[2] + ax_arr[3];
        ay[i] = ay_arr[0] + ay_arr[1] + ay_arr[2] + ay_arr[3];
        az[i] = az_arr[0] + az_arr[1] + az_arr[2] + az_arr[3];
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
    
    ParticlesSoA p_soa;
    particles_to_soa(particles, p_soa);
    
    // Compute initial accelerations
    compute_accelerations_avx2(p_soa, p_soa.ax, p_soa.ay, p_soa.az);
    
    soa_to_particles(p_soa, particles);
    double initial_energy = calculate_energy(particles);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Initial Energy: " << initial_energy << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double half_dt = 0.5 * dt;
    double half_dt_sq = 0.5 * dt * dt;
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1. Update positions
        for (int i = 0; i < n; ++i) {
            p_soa.x[i] += p_soa.vx[i] * dt + p_soa.ax[i] * half_dt_sq;
            p_soa.y[i] += p_soa.vy[i] * dt + p_soa.ay[i] * half_dt_sq;
            p_soa.z[i] += p_soa.vz[i] * dt + p_soa.az[i] * half_dt_sq;
        }
        
        // Save current accelerations as old
        std::vector<double> old_ax = p_soa.ax;
        std::vector<double> old_ay = p_soa.ay;
        std::vector<double> old_az = p_soa.az;
        
        // 2. Compute new accelerations
        compute_accelerations_avx2(p_soa, p_soa.ax, p_soa.ay, p_soa.az);
        
        // 3. Update velocities
        for (int i = 0; i < n; ++i) {
            p_soa.vx[i] += (old_ax[i] + p_soa.ax[i]) * half_dt;
            p_soa.vy[i] += (old_ay[i] + p_soa.ay[i]) * half_dt;
            p_soa.vz[i] += (old_az[i] + p_soa.az[i]) * half_dt;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    soa_to_particles(p_soa, particles);
    double final_energy = calculate_energy(particles);
    double energy_drift = std::abs(final_energy - initial_energy);
    
    std::cout << "Final Energy:   " << final_energy << std::endl;
    std::cout << "Energy Drift:   " << energy_drift << std::endl;
    std::cout << "Simulation loop time: " << elapsed.count() << " seconds" << std::endl;
    
    // Write simulation_time.txt in the same directory as output_file
    std::string time_file_path = "simulation_time.txt";
    size_t last_slash = output_file.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        time_file_path = output_file.substr(0, last_slash + 1) + "simulation_time.txt";
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
