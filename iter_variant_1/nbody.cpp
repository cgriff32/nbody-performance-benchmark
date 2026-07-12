#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const double G = 1.0;
const double SOFTENING_SQ = 0.25;

typedef struct {
    int id;
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
} Particle;

// Computes the accelerations using SoA layout
void compute_accelerations_soa(
    const double* restrict x,
    const double* restrict y,
    const double* restrict z,
    const double* restrict mass,
    double* restrict ax,
    double* restrict ay,
    double* restrict az,
    int n) 
{
    const double* restrict x_al = (const double*)__builtin_assume_aligned(x, 64);
    const double* restrict y_al = (const double*)__builtin_assume_aligned(y, 64);
    const double* restrict z_al = (const double*)__builtin_assume_aligned(z, 64);
    const double* restrict mass_al = (const double*)__builtin_assume_aligned(mass, 64);
    double* restrict ax_al = (double*)__builtin_assume_aligned(ax, 64);
    double* restrict ay_al = (double*)__builtin_assume_aligned(ay, 64);
    double* restrict az_al = (double*)__builtin_assume_aligned(az, 64);

    for (int i = 0; i < n; ++i) {
        double px = x_al[i];
        double py = y_al[i];
        double pz = z_al[i];
        double cur_ax = 0.0;
        double cur_ay = 0.0;
        double cur_az = 0.0;
        
        for (int j = 0; j < n; ++j) {
            double dx = x_al[j] - px;
            double dy = y_al[j] - py;
            double dz = z_al[j] - pz;
            
            double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING_SQ;
            double force_factor = mass_al[j] / (dist_sq * sqrt(dist_sq));
            
            cur_ax += force_factor * dx;
            cur_ay += force_factor * dy;
            cur_az += force_factor * dz;
        }
        ax_al[i] = cur_ax;
        ay_al[i] = cur_ay;
        az_al[i] = cur_az;
    }
}

// Calculates the total Hamiltonian energy using SoA layout
double calculate_energy_soa(
    const double* restrict x,
    const double* restrict y,
    const double* restrict z,
    const double* restrict vx,
    const double* restrict vy,
    const double* restrict vz,
    const double* restrict mass,
    int n) 
{
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    
    for (int i = 0; i < n; ++i) {
        double v_sq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        kinetic_energy += 0.5 * mass[i] * v_sq;
        
        double px = x[i];
        double py = y[i];
        double pz = z[i];
        double pmass = mass[i];
        for (int j = i + 1; j < n; ++j) {
            double dx = x[j] - px;
            double dy = y[j] - py;
            double dz = z[j] - pz;
            double dist = sqrt(dx*dx + dy*dy + dz*dz + SOFTENING_SQ);
            potential_energy -= pmass * mass[j] / dist;
        }
    }
    
    return kinetic_energy + potential_energy;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_csv> <output_csv> <num_steps> <dt>\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    int num_steps = atoi(argv[3]);
    double dt = atof(argv[4]);
    
    // Load initial state CSV
    FILE* infile = fopen(input_file, "r");
    if (!infile) {
        fprintf(stderr, "Error: Could not open input file %s\n", input_file);
        return 1;
    }
    
    char line[1024];
    int capacity = 1000;
    Particle* particles = (Particle*)malloc(capacity * sizeof(Particle));
    int n = 0;
    
    // Skip header line
    if (!fgets(line, sizeof(line), infile)) {
        fprintf(stderr, "Error: Empty input file\n");
        fclose(infile);
        free(particles);
        return 1;
    }
    
    while (fgets(line, sizeof(line), infile)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (n >= capacity) {
            capacity *= 2;
            particles = (Particle*)realloc(particles, capacity * sizeof(Particle));
        }
        Particle* p = &particles[n];
        if (sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf", 
                   &p->id, &p->mass, &p->x, &p->y, &p->z, &p->vx, &p->vy, &p->vz) == 8) {
            p->ax = p->ay = p->az = 0.0;
            n++;
        }
    }
    fclose(infile);
    
    printf("Loaded %d particles.\n", n);
    
    // Allocate SoA buffers with 64-byte alignment
    double* x; posix_memalign((void**)&x, 64, n * sizeof(double));
    double* y; posix_memalign((void**)&y, 64, n * sizeof(double));
    double* z; posix_memalign((void**)&z, 64, n * sizeof(double));
    double* vx; posix_memalign((void**)&vx, 64, n * sizeof(double));
    double* vy; posix_memalign((void**)&vy, 64, n * sizeof(double));
    double* vz; posix_memalign((void**)&vz, 64, n * sizeof(double));
    double* ax; posix_memalign((void**)&ax, 64, n * sizeof(double));
    double* ay; posix_memalign((void**)&ay, 64, n * sizeof(double));
    double* az; posix_memalign((void**)&az, 64, n * sizeof(double));
    double* mass; posix_memalign((void**)&mass, 64, n * sizeof(double));
    int* id; posix_memalign((void**)&id, 64, n * sizeof(int));
    
    // Copy from AoS to SoA
    for (int i = 0; i < n; ++i) {
        id[i] = particles[i].id;
        mass[i] = particles[i].mass;
        x[i] = particles[i].x;
        y[i] = particles[i].y;
        z[i] = particles[i].z;
        vx[i] = particles[i].vx;
        vy[i] = particles[i].vy;
        vz[i] = particles[i].vz;
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
    }
    
    // Compute initial accelerations
    compute_accelerations_soa(x, y, z, mass, ax, ay, az, n);
    
    double initial_energy = calculate_energy_soa(x, y, z, vx, vy, vz, mass, n);
    printf("Initial Energy: %.10e\n", initial_energy);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    double dt_sq_half = 0.5 * dt * dt;
    double dt_half = 0.5 * dt;
    
    // Simulation Loop (Velocity Verlet)
    for (int step = 0; step < num_steps; ++step) {
        // 1a. Update positions
        for (int i = 0; i < n; ++i) {
            x[i] += vx[i] * dt + ax[i] * dt_sq_half;
            y[i] += vy[i] * dt + ay[i] * dt_sq_half;
            z[i] += vz[i] * dt + az[i] * dt_sq_half;
        }
        
        // 1b. Half-update velocities
        for (int i = 0; i < n; ++i) {
            vx[i] += ax[i] * dt_half;
            vy[i] += ay[i] * dt_half;
            vz[i] += az[i] * dt_half;
        }
        
        // 2. Compute new accelerations
        compute_accelerations_soa(x, y, z, mass, ax, ay, az, n);
        
        // 3. Complete velocity updates
        for (int i = 0; i < n; ++i) {
            vx[i] += ax[i] * dt_half;
            vy[i] += ay[i] * dt_half;
            vz[i] += az[i] * dt_half;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    double final_energy = calculate_energy_soa(x, y, z, vx, vy, vz, mass, n);
    double energy_drift = fabs(final_energy - initial_energy);
    
    printf("Final Energy:   %.10e\n", final_energy);
    printf("Energy Drift:   %.10e\n", energy_drift);
    printf("Simulation loop time: %f seconds\n", elapsed);
    FILE* time_file = fopen("simulation_time.txt", "w");
    if (time_file) {
        fprintf(time_file, "%.9f\n", elapsed);
        fclose(time_file);
    }
    
    // Copy SoA back to AoS for output
    for (int i = 0; i < n; ++i) {
        particles[i].x = x[i];
        particles[i].y = y[i];
        particles[i].z = z[i];
        particles[i].vx = vx[i];
        particles[i].vy = vy[i];
        particles[i].vz = vz[i];
    }
    
    // Write final state CSV
    FILE* outfile = fopen(output_file, "w");
    if (!outfile) {
        fprintf(stderr, "Error: Could not open output file %s\n", output_file);
        return 1;
    }
    
    fprintf(outfile, "id,mass,x,y,z,vx,vy,vz\n");
    for (int i = 0; i < n; ++i) {
        const Particle* p = &particles[i];
        fprintf(outfile, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                p->id, p->mass, p->x, p->y, p->z, p->vx, p->vy, p->vz);
    }
    fclose(outfile);
    
    free(x); free(y); free(z);
    free(vx); free(vy); free(vz);
    free(ax); free(ay); free(az);
    free(mass); free(id);
    free(particles);
    return 0;
}
