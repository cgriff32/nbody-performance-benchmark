#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <charconv>
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

// Highly optimized AVX2 acceleration calculation helper (Outer unrolled by 2, Inner unrolled by 2)
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

    constexpr int B_i = 1024;
    constexpr int B_j = 1024;

    #pragma omp parallel for schedule(static)
    for (int i_block = 0; i_block < n; i_block += B_i) {
        int i_end = (i_block + B_i < n) ? (i_block + B_i) : n;

        // Initialize accelerations for this block to 0
        for (int i = i_block; i < i_end; ++i) {
            ax_al[i] = 0.0f;
            ay_al[i] = 0.0f;
            az_al[i] = 0.0f;
        }

        for (int j_block = 0; j_block < n_pad; j_block += B_j) {
            int j_end = (j_block + B_j < n_pad) ? (j_block + B_j) : n_pad;

            int i = i_block;
            // Outer loop unrolled by 2
            for (; i < i_end - 1; i += 2) {
                __m256 xi_v0 = _mm256_set1_ps(x_al[i]);
                __m256 yi_v0 = _mm256_set1_ps(y_al[i]);
                __m256 zi_v0 = _mm256_set1_ps(z_al[i]);

                __m256 xi_v1 = _mm256_set1_ps(x_al[i+1]);
                __m256 yi_v1 = _mm256_set1_ps(y_al[i+1]);
                __m256 zi_v1 = _mm256_set1_ps(z_al[i+1]);

                __m256 ax_accum0 = _mm256_setzero_ps();
                __m256 ay_accum0 = _mm256_setzero_ps();
                __m256 az_accum0 = _mm256_setzero_ps();

                __m256 ax_accum1 = _mm256_setzero_ps();
                __m256 ay_accum1 = _mm256_setzero_ps();
                __m256 az_accum1 = _mm256_setzero_ps();

                // Inner loop unrolled by 2 (processes 16 elements per step)
                for (int j = j_block; j < j_end; j += 16) {
                    __m256 xj_v0 = _mm256_load_ps(&x_al[j]);
                    __m256 yj_v0 = _mm256_load_ps(&y_al[j]);
                    __m256 zj_v0 = _mm256_load_ps(&z_al[j]);
                    __m256 mj_v0 = _mm256_load_ps(&mass_al[j]);

                    __m256 xj_v1 = _mm256_load_ps(&x_al[j+8]);
                    __m256 yj_v1 = _mm256_load_ps(&y_al[j+8]);
                    __m256 zj_v1 = _mm256_load_ps(&z_al[j+8]);
                    __m256 mj_v1 = _mm256_load_ps(&mass_al[j+8]);

                    // Interaction of j0 with particle i0
                    __m256 dx0_0 = _mm256_sub_ps(xj_v0, xi_v0);
                    __m256 dy0_0 = _mm256_sub_ps(yj_v0, yi_v0);
                    __m256 dz0_0 = _mm256_sub_ps(zj_v0, zi_v0);

                    __m256 dist_sq0_0 = _mm256_fmadd_ps(dz0_0, dz0_0, softening_v);
                    dist_sq0_0 = _mm256_fmadd_ps(dy0_0, dy0_0, dist_sq0_0);
                    dist_sq0_0 = _mm256_fmadd_ps(dx0_0, dx0_0, dist_sq0_0);

                    __m256 inv_sqrt0_0 = _mm256_rsqrt_ps(dist_sq0_0);
                    __m256 inv_dist_3_0_0 = _mm256_mul_ps(_mm256_mul_ps(inv_sqrt0_0, inv_sqrt0_0), inv_sqrt0_0);
                    __m256 force_factor0_0 = _mm256_mul_ps(mj_v0, inv_dist_3_0_0);

                    ax_accum0 = _mm256_fmadd_ps(force_factor0_0, dx0_0, ax_accum0);
                    ay_accum0 = _mm256_fmadd_ps(force_factor0_0, dy0_0, ay_accum0);
                    az_accum0 = _mm256_fmadd_ps(force_factor0_0, dz0_0, az_accum0);

                    // Interaction of j0 with particle i1
                    __m256 dx0_1 = _mm256_sub_ps(xj_v0, xi_v1);
                    __m256 dy0_1 = _mm256_sub_ps(yj_v0, yi_v1);
                    __m256 dz0_1 = _mm256_sub_ps(zj_v0, zi_v1);

                    __m256 dist_sq0_1 = _mm256_fmadd_ps(dz0_1, dz0_1, softening_v);
                    dist_sq0_1 = _mm256_fmadd_ps(dy0_1, dy0_1, dist_sq0_1);
                    dist_sq0_1 = _mm256_fmadd_ps(dx0_1, dx0_1, dist_sq0_1);

                    __m256 inv_sqrt0_1 = _mm256_rsqrt_ps(dist_sq0_1);
                    __m256 inv_dist_3_0_1 = _mm256_mul_ps(_mm256_mul_ps(inv_sqrt0_1, inv_sqrt0_1), inv_sqrt0_1);
                    __m256 force_factor0_1 = _mm256_mul_ps(mj_v0, inv_dist_3_0_1);

                    ax_accum1 = _mm256_fmadd_ps(force_factor0_1, dx0_1, ax_accum1);
                    ay_accum1 = _mm256_fmadd_ps(force_factor0_1, dy0_1, ay_accum1);
                    az_accum1 = _mm256_fmadd_ps(force_factor0_1, dz0_1, az_accum1);

                    // Interaction of j1 with particle i0
                    __m256 dx1_0 = _mm256_sub_ps(xj_v1, xi_v0);
                    __m256 dy1_0 = _mm256_sub_ps(yj_v1, yi_v0);
                    __m256 dz1_0 = _mm256_sub_ps(zj_v1, zi_v0);

                    __m256 dist_sq1_0 = _mm256_fmadd_ps(dz1_0, dz1_0, softening_v);
                    dist_sq1_0 = _mm256_fmadd_ps(dy1_0, dy1_0, dist_sq1_0);
                    dist_sq1_0 = _mm256_fmadd_ps(dx1_0, dx1_0, dist_sq1_0);

                    __m256 inv_sqrt1_0 = _mm256_rsqrt_ps(dist_sq1_0);
                    __m256 inv_dist_3_1_0 = _mm256_mul_ps(_mm256_mul_ps(inv_sqrt1_0, inv_sqrt1_0), inv_sqrt1_0);
                    __m256 force_factor1_0 = _mm256_mul_ps(mj_v1, inv_dist_3_1_0);

                    ax_accum0 = _mm256_fmadd_ps(force_factor1_0, dx1_0, ax_accum0);
                    ay_accum0 = _mm256_fmadd_ps(force_factor1_0, dy1_0, ay_accum0);
                    az_accum0 = _mm256_fmadd_ps(force_factor1_0, dz1_0, az_accum0);

                    // Interaction of j1 with particle i1
                    __m256 dx1_1 = _mm256_sub_ps(xj_v1, xi_v1);
                    __m256 dy1_1 = _mm256_sub_ps(yj_v1, yi_v1);
                    __m256 dz1_1 = _mm256_sub_ps(zj_v1, zi_v1);

                    __m256 dist_sq1_1 = _mm256_fmadd_ps(dz1_1, dz1_1, softening_v);
                    dist_sq1_1 = _mm256_fmadd_ps(dy1_1, dy1_1, dist_sq1_1);
                    dist_sq1_1 = _mm256_fmadd_ps(dx1_1, dx1_1, dist_sq1_1);

                    __m256 inv_sqrt1_1 = _mm256_rsqrt_ps(dist_sq1_1);
                    __m256 inv_dist_3_1_1 = _mm256_mul_ps(_mm256_mul_ps(inv_sqrt1_1, inv_sqrt1_1), inv_sqrt1_1);
                    __m256 force_factor1_1 = _mm256_mul_ps(mj_v1, inv_dist_3_1_1);

                    ax_accum1 = _mm256_fmadd_ps(force_factor1_1, dx1_1, ax_accum1);
                    ay_accum1 = _mm256_fmadd_ps(force_factor1_1, dy1_1, ay_accum1);
                    az_accum1 = _mm256_fmadd_ps(force_factor1_1, dz1_1, az_accum1);
                }

                ax_al[i] += hsum_8(ax_accum0);
                ay_al[i] += hsum_8(ay_accum0);
                az_al[i] += hsum_8(az_accum0);

                ax_al[i+1] += hsum_8(ax_accum1);
                ay_al[i+1] += hsum_8(ay_accum1);
                az_al[i+1] += hsum_8(az_accum1);
            }

            // Scalar tail
            for (; i < i_end; ++i) {
                __m256 xi_v = _mm256_set1_ps(x_al[i]);
                __m256 yi_v = _mm256_set1_ps(y_al[i]);
                __m256 zi_v = _mm256_set1_ps(z_al[i]);

                __m256 ax_accum = _mm256_setzero_ps();
                __m256 ay_accum = _mm256_setzero_ps();
                __m256 az_accum = _mm256_setzero_ps();

                for (int j = j_block; j < j_end; j += 16) {
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

                ax_al[i] += hsum_8(ax_accum);
                ay_al[i] += hsum_8(ay_accum);
                az_al[i] += hsum_8(az_accum);
            }
        }
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
    
    // Load initial state CSV using fast std::from_chars buffered parser
    FILE* infile = fopen(input_file.c_str(), "rb");
    if (!infile) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }
    std::fseek(infile, 0, SEEK_END);
    long file_size = std::ftell(infile);
    std::fseek(infile, 0, SEEK_SET);
    
    std::vector<char> file_buf(file_size + 1);
    size_t read_bytes = std::fread(file_buf.data(), 1, file_size, infile);
    file_buf[read_bytes] = '\0';
    std::fclose(infile);
    
    const char* p_cursor = file_buf.data();
    const char* p_end = p_cursor + read_bytes;
    
    // Skip header line
    while (p_cursor < p_end && *p_cursor != '\n') {
        p_cursor++;
    }
    if (p_cursor < p_end && *p_cursor == '\n') {
        p_cursor++;
    }
    
    particles.reserve(100000);
    while (p_cursor < p_end) {
        while (p_cursor < p_end && (*p_cursor == ' ' || *p_cursor == '\r' || *p_cursor == '\n')) {
            p_cursor++;
        }
        if (p_cursor >= p_end) break;
        
        Particle p;
        p.ax = p.ay = p.az = 0.0;
        
        auto [ptr1, ec1] = std::from_chars(p_cursor, p_end, p.id);
        if (ec1 != std::errc() || *ptr1 != ',') break;
        p_cursor = ptr1 + 1;
        
        auto [ptr2, ec2] = std::from_chars(p_cursor, p_end, p.mass);
        if (ec2 != std::errc() || *ptr2 != ',') break;
        p_cursor = ptr2 + 1;
        
        auto [ptr3, ec3] = std::from_chars(p_cursor, p_end, p.x);
        if (ec3 != std::errc() || *ptr3 != ',') break;
        p_cursor = ptr3 + 1;
        
        auto [ptr4, ec4] = std::from_chars(p_cursor, p_end, p.y);
        if (ec4 != std::errc() || *ptr4 != ',') break;
        p_cursor = ptr4 + 1;
        
        auto [ptr5, ec5] = std::from_chars(p_cursor, p_end, p.z);
        if (ec5 != std::errc() || *ptr5 != ',') break;
        p_cursor = ptr5 + 1;
        
        auto [ptr6, ec6] = std::from_chars(p_cursor, p_end, p.vx);
        if (ec6 != std::errc() || *ptr6 != ',') break;
        p_cursor = ptr6 + 1;
        
        auto [ptr7, ec7] = std::from_chars(p_cursor, p_end, p.vy);
        if (ec7 != std::errc() || *ptr7 != ',') break;
        p_cursor = ptr7 + 1;
        
        auto [ptr8, ec8] = std::from_chars(p_cursor, p_end, p.vz);
        if (ec8 != std::errc()) break;
        p_cursor = ptr8;
        
        particles.push_back(p);
    }
    
    int n = particles.size();
    
    // Pad n to multiple of 16
    int n_pad = (n + 15) & ~15;
    int stride = n_pad + 64; // pad each array to prevent cache set conflicts
    
    // Allocate a single contiguous block of 64-byte aligned memory for all 10 SoA arrays
    float* data_block = allocate_aligned_float(10 * stride);
    
    float* mass = data_block + 0 * stride;
    float* x    = data_block + 1 * stride;
    float* y    = data_block + 2 * stride;
    float* z    = data_block + 3 * stride;
    float* vx   = data_block + 4 * stride;
    float* vy   = data_block + 5 * stride;
    float* vz   = data_block + 6 * stride;
    float* ax   = data_block + 7 * stride;
    float* ay   = data_block + 8 * stride;
    float* az   = data_block + 9 * stride;

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
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    float dt_f = (float)dt;
    float dt_half = 0.5f * dt_f;
    
    // 1. Initial half-step velocity update + Step 0 position update (fused)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        vx[i] += ax[i] * dt_half;
        vy[i] += ay[i] * dt_half;
        vz[i] += az[i] * dt_half;
        x[i] += vx[i] * dt_f;
        y[i] += vy[i] * dt_f;
        z[i] += vz[i] * dt_f;
    }
    
    // 2. Main simulation loop (Velocity Verlet)
    compute_accelerations_soa(n, n_pad, x, y, z, mass, ax, ay, az);
    
    // Steps 1 to num_steps - 1
    for (int step = 1; step < num_steps; ++step) {
        #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
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
    
    // Write high-precision simulation time
    FILE* time_file = fopen("/app/simulation_time.txt", "w");
    if (!time_file) {
        time_file = fopen("simulation_time.txt", "w");
    }
    if (time_file) {
        fprintf(time_file, "%.9f\n", elapsed.count());
        fclose(time_file);
    }
    
    // Write final state CSV using fast std::to_chars
    FILE* outfile = fopen(output_file.c_str(), "w");
    if (!outfile) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }
    
    // Large buffer to hold the output (48 MB)
    std::vector<char> buffer(48 * 1024 * 1024);
    char* ptr = buffer.data();
    char* end_ptr = ptr + buffer.size();
    
    // Header
    const char* header_out = "id,mass,x,y,z,vx,vy,vz\n";
    size_t header_len = std::strlen(header_out);
    std::memcpy(ptr, header_out, header_len);
    ptr += header_len;
    
    auto write_double = [](char*& p, char* limit, double val) {
        auto [next, ec] = std::to_chars(p, limit, val, std::chars_format::scientific, 17);
        if (ec == std::errc()) {
            p = next;
        }
    };
    
    auto write_int = [](char*& p, char* limit, int val) {
        auto [next, ec] = std::to_chars(p, limit, val);
        if (ec == std::errc()) {
            p = next;
        }
    };
    
    for (const auto& p : particles) {
        write_int(ptr, end_ptr, p.id);
        *ptr++ = ',';
        write_double(ptr, end_ptr, p.mass);
        *ptr++ = ',';
        write_double(ptr, end_ptr, p.x);
        *ptr++ = ',';
        write_double(ptr, end_ptr, p.y);
        *ptr++ = ',';
        write_double(ptr, end_ptr, p.z);
        *ptr++ = ',';
        write_double(ptr, end_ptr, p.vx);
        *ptr++ = ',';
        write_double(ptr, end_ptr, p.vy);
        *ptr++ = ',';
        write_double(ptr, end_ptr, p.vz);
        *ptr++ = '\n';
    }
    
    std::fwrite(buffer.data(), 1, ptr - buffer.data(), outfile);
    std::fclose(outfile);
    
    // Free aligned memory
    free(data_block);
    
    return 0;
}
