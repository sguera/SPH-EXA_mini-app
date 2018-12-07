#pragma once

inline double compute_3d_k(double n)
{
	//b0, b1, b2 and b3 are defined in "SPHYNX: an accurate density-based SPH method for astrophysical applications", DOI: 10.1051/0004-6361/201630208
	double b0 = 2.7012593e-2;
	double b1 = 2.0410827e-2;
	double b2 = 3.7451957e-3;
	double b3 = 4.7013839e-2;

	return b0 + b1 * sqrt(n) + b2 * n + b3 * sqrt(n*n*n);
}

inline double wharmonic(double v, double h, double K)
{
    double value = (PI/2.0) * v;
    return K/(h*h*h) * pow((sin(value)/value), 5);
}

inline double wharmonic_derivative(double v, double h, double K)
{
    double value = (PI/2.0) * v;
    // r_ih = v * h
    // extra h at the bottom comes from the chain rule of the partial derivative
    double kernel = wharmonic(v, h, K);

    return 5.0 * (PI/2.0) * kernel / (h * h) / v * ((1.0 / tan(value)) - (1.0 / value));
}

template<typename T, typename Tfield = std::vector<T>, typename N, typename Nlist = std::vector<N>>
Momentum : public computeParticleTask {
	
	public:
		
		Momentum(const TField &x, const TField &y, const TField &z, const TField &vx, const TField &vy, const TField &vz, const TField &ro, const TField &p, const TField &h, const TField &c, const TField &m, const N &ngmax, const Nlist &ng, TField &grad_P_x, TField &grad_P_y, TField &grad_P_z){
			&x = &x;
			&y = &y;
			&z = &z;
			&vx = &vx;
			&vy = &vy;
			&vz = &vz;
			&ro = &ro;
			&p = &p;
			&h = &h;
			&c = &c;
			&m = &m;
			&ngmax = &ngmax;
			&ng = &ng;
		}


		compute(N particle_id){

			K = compute_3d_k(5.0);
			T ro_i = ro[particle_id];
		    T p_i = p[particle_id];
		    T x_i = x[particle_id];
		    T y_i = y[particle_id];
		    T z_i = z[particle_id];
		    T vx_i = vx[particle_id];
		    T vy_i = vy[particle_id];
		    T vz_i = vz[particle_id];
		    T h_i = h[particle_id];
			T momentum_x = 0.0;
			T momentum_y = 0.0;
			T momentum_z = 0.0;


			for(int j=0; j<d.nvi[i]; j++) {

				// retrive the id of a neighbor
	        	N neigh_id = ng[particle_id*ngmax+j];
	        	if(neigh_id == particle_id) continue;

	        	T ro_j = ro[neigh_id];
		        T p_j = p[neigh_id];
		        T x_j = x[neigh_id];
		        T y_j = y[neigh_id];
		        T z_j = z[neigh_id];
		        T h_j = h[neigh_id];

		        // calculate the scalar product rv = rij * vij
		        T r_ijx = (x_i - x_j);
		        T r_ijy = (y_i - y_j);
		        T r_ijz = (z_i - z_j);

		        T v_ijx = (vx_i - vx[neigh_id]);
		        T v_ijy = (vy_i - vy[neigh_id]);
		        T v_ijz = (vz_i - vz[neigh_id]);

		        T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

		        T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

		        T viscosity_ij = artificial_viscosity(ro_i, ro_j, h[i], h[neigh_id], c[particle_id], c[neigh_id], rv, r_square);

		        T r_ij = sqrt(r_square);
		        T v_i = r_ij / h_i;
		        T v_j = r_ij / h_j;

		        T derivative_kernel_i = wharmonic_derivative(v_i, h_i, K);
		        T derivative_kernel_j = wharmonic_derivative(v_j, h_j, K);
		        
		        T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
		        T grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
		        T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
		        T grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
		        T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;
		        T grad_v_kernel_z_j = r_ijz * derivative_kernel_j;
				
				momentum_x +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_x_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_x_j) + viscosity_ij * (grad_v_kernel_x_i + grad_v_kernel_x_j)/2.0;
		        momentum_y +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_y_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_y_j) + viscosity_ij * (grad_v_kernel_y_i + grad_v_kernel_y_j)/2.0;
		        momentum_z +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_z_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_z_j) + viscosity_ij * (grad_v_kernel_z_i + grad_v_kernel_z_j)/2.0;
		    }

		    grad_P_x[particle_id] = momentum_x * m[particle_id];
		    grad_P_y[particle_id] = momentum_y * m[particle_id];
		    grad_P_z[particle_id] = momentum_z * m[particle_id];

		}

	private:
		static const T gradh_i = 1.0;
		static const T gradh_j = 1.0;
		static const T K;
		static const N &ngmax;
		static const TField &x, TField &y, TField &z, TField &vx, TField &vy, TField &vz, TField &ro, TField &p, TField &h, TField &c, TField &m;
		static const Nlist &ng;
		TField &grad_P_x, &grad_P_y, &grad_P_z;

}