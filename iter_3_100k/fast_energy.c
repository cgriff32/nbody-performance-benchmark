#include <math.h>

double calculate_potential_energy(const double* mass, const double* x, const double* y, const double* z, int n, double softening_sq) {
    double potential = 0.0;
    for (int i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];
        double zi = z[i];
        double mi = mass[i];
        #pragma omp simd reduction(+:potential)
        for (int j = i + 1; j < n; ++j) {
            double dx = x[j] - xi;
            double dy = y[j] - yi;
            double dz = z[j] - zi;
            double dist = sqrt(dx*dx + dy*dy + dz*dz + softening_sq);
            potential -= mi * mass[j] / dist;
        }
    }
    return potential;
}
