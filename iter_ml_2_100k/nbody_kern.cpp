#include <cmath>
#include <algorithm>

extern "C" {
void compute_accelerations_cpp(
    const double* x, const double* y, const double* z, const double* mass,
    double* ax, double* ay, double* az, int n,
    const double* w) {
    
    double w0 = w[0];
    double w1 = w[1];
    double w2 = w[2];
    
    bool has_w0 = (std::abs(w0) > 1e-6);
    bool has_w2 = (std::abs(w2) > 1e-6);
    
    if (!has_w0 && !has_w2) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double ax_i = 0.0;
            double ay_i = 0.0;
            double az_i = 0.0;
            
            double xi = x[i];
            double yi = y[i];
            double zi = z[i];
            
            #pragma omp simd reduction(+:ax_i,ay_i,az_i)
            for (int j = 0; j < n; ++j) {
                double dx = x[j] - xi;
                double dy = y[j] - yi;
                double dz = z[j] - zi;
                
                double r_sq = dx*dx + dy*dy + dz*dz;
                double soft_sq = r_sq + 0.25;
                double inv_soft = 1.0 / soft_sq;
                double inv_soft_32 = inv_soft * std::sqrt(inv_soft);
                
                double factor = - w1 * inv_soft_32;
                double m_factor = mass[j] * factor;
                
                ax_i += m_factor * dx;
                ay_i += m_factor * dy;
                az_i += m_factor * dz;
            }
            ax[i] = ax_i;
            ay[i] = ay_i;
            az[i] = az_i;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double ax_i = 0.0;
            double ay_i = 0.0;
            double az_i = 0.0;
            
            double xi = x[i];
            double yi = y[i];
            double zi = z[i];
            
            #pragma omp simd reduction(+:ax_i,ay_i,az_i)
            for (int j = 0; j < n; ++j) {
                double dx = x[j] - xi;
                double dy = y[j] - yi;
                double dz = z[j] - zi;
                
                double r_sq = dx*dx + dy*dy + dz*dz;
                double soft_sq = r_sq + 0.25;
                double inv_soft = 1.0 / soft_sq;
                
                double r_dist = std::sqrt(r_sq + 1e-30);
                double inv_soft_32 = inv_soft * std::sqrt(inv_soft);
                
                double factor = w0 / r_dist - w1 * inv_soft_32 - 2.0 * w2 * (inv_soft * inv_soft);
                double m_factor = mass[j] * factor;
                
                ax_i += m_factor * dx;
                ay_i += m_factor * dy;
                az_i += m_factor * dz;
            }
            ax[i] = ax_i;
            ay[i] = ay_i;
            az[i] = az_i;
        }
    }
}
}
