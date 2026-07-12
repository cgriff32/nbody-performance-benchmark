#include <math.h>

double calculate_potential_energy(const double* x, const double* y, const double* z, const double* mass, int n) {
    double potential = 0.0;
    const double softening_sq = 0.25;
    for (int i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];
        double zi = z[i];
        double mi = mass[i];
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
