#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "grad.h"
#include "geom.h"
#include "geom.cu"

__global__ void DA_kernel(double *A, double *B, double *C, int num_rows, int num_cols)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < num_rows)
		for (int col = 0; col < num_cols; col++)
			C[row * num_cols + col] = A[row] * B[row * num_cols + col];
}

__global__ void SA_kernel(double scalar, double *A, int num_rows, int num_cols)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < num_rows)
		for (int col = 0; col < num_cols; col++)
			A[row * num_cols + col] *= scalar;
}

__global__ void PA_kernel(double *A, double *B, int num_rows, int num_cols)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < num_rows)
		for (int col = 0; col < num_cols; col++)
			A[row * num_cols + col] += B[row * num_cols + col];
}

__global__ void gradient_properties_kernel(const float *vertex, int nVertex, const int *face, int nFace, const float *feature, const float *propertySamples, int nSamples, const double *variance, const float *property, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const double *m_bar, const float *u1, const float *u2, const int *fid, double *gradient, double *gradient_raw, double *gradient_diag, double *dEdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nSamples)
	{
		cuCoordinate coord;
		int nCoeff = (degree + 1) * (degree + 1);
		int n = (deg_end + 1) * (deg_end + 1);
		int n0 = deg_beg * deg_beg;
		int size = n - n0;

		// spherical coordinate
		const float *y = &propertySamples[i * 3];
		float cY[3];
		int id1 = face[fid[i] * 3 + 0];
		int id2 = face[fid[i] * 3 + 1];
		int id3 = face[fid[i] * 3 + 2];

		// approximation of Y
		const float *v1 = &vertex[id1 * 3];
		const float *v2 = &vertex[id2 * 3];
		const float *v3 = &vertex[id3 * 3];
		
		cuVector N = cuVector(v1, v2).cross(cuVector(v2, v3));
		float area = N.norm();
		N.unit();
		cuVector Yproj = cuVector(y) * ((cuVector(v1) * N) / (cuVector(y) * N));
		coord.cart2bary((float *)v1, (float *)v2, (float *)v3, (float *)Yproj.fv(), cY);

		const double *Y1 = &Y[nCoeff * id1];
		const double *Y2 = &Y[nCoeff * id2];
		const double *Y3 = &Y[nCoeff * id3];

		double delta[3] = {0, 0, 0};
		for (int j = 0; j < nCoeff; j++)
		{
			double Y = (Y1[j] * cY[0] + Y2[j] * cY[1] + Y3[j] * cY[2]);
			delta[0] += Y * coeff[j];
			delta[1] += Y * coeff[nCoeff + j];
			delta[2] += Y * coeff[nCoeff * 2 + j];
		}
		// m
		double m = feature[i];

		// z
		float z_hat[3];
		coord.sph2cart(pole[0], pole[1], z_hat);
		float z_dot_orth[3]; const float *z_dot_orth_ = (cuVector(u1) * (float)delta[1] + cuVector(u2) * (float)delta[2]).unit().fv();
		for (int j = 0; j < 3; j++) z_dot_orth[j] = z_dot_orth_[j];
		float degree = (float)sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
		cuVector Z_ddot = cuVector(z_hat).cross(cuVector(z_dot_orth));
		float z_ddot[3]; const float *z_ddot_ = Z_ddot.unit().fv();
		for (int j = 0; j < 3; j++) z_ddot[j] = z_ddot_[j];
		float rot[9];
		coord.rotation(z_ddot, degree, rot);
		float z_dot[3];
		coord.rotPoint(z_hat, rot, z_dot);

		// dp/dx
		float nf[3]; const float *nf_ = N.fv();
		for (int j = 0; j < 3; j++) nf[j] = nf_[j];
		cuVector V1(v1), Yv(y);
		float v1N = V1 * N;
		float yN = Yv * N;
		double r1 = v1N / yN;
		double r2 = r1 / yN;
		double dpdx[3][3] = {{r1 - r2 * y[0] * nf[0], -r2 * y[0] * nf[1], -r2 * y[0] * nf[2]},
							{-r2 * y[1] * nf[0], r1 - r2 * y[1] * nf[1], -r2 * y[1] * nf[2]},
							{-r2 * y[2] * nf[0], -r2 * y[2] * nf[1], r1 - r2 * y[2] * nf[2]}};
		// dm/dp
		cuVector YP1(Yproj.fv(), v1), YP2(Yproj.fv(), v2), YP3(Yproj.fv(), v3);
		float m1 = property[id1];
		float m2 = property[id2];
		float m3 = property[id3];

		cuVector V1V2(v1, v2), V2V3(v2, v3), V3V1(v3, v1);
		float v1v2_norm = V1V2.norm();
		float v2v3_norm = V2V3.norm();
		float v3v1_norm = V3V1.norm();
		cuVector DA1DP = V2V3.cross(N).unit() * v2v3_norm * m1;
		cuVector DA2DP = V3V1.cross(N).unit() * v3v1_norm * m2;
		cuVector DA3DP = V1V2.cross(N).unit() * v1v2_norm * m3;
		cuVector DMDP = (DA1DP + DA2DP + DA3DP) / area;
		//cuVector DMDP = (DA1DP + DA2DP + DA3DP);	// canceled out: area
	
		const float *dmdp = DMDP.fv();
	
		// grad_m
		double grad_m[3] = {dpdx[0][0] * dmdp[0] + dpdx[0][1] * dmdp[1] + dpdx[0][2] * dmdp[2],
							dpdx[1][0] * dmdp[0] + dpdx[1][1] * dmdp[1] + dpdx[1][2] * dmdp[2],
							dpdx[2][0] * dmdp[0] + dpdx[2][1] * dmdp[1] + dpdx[2][2] * dmdp[2]};

		//cout << "grad_m: " << grad_m[0] << " " << grad_m[1] << " " << grad_m[2] << endl;
	
		// z_hatXu
		float z_hatXu1[3]; const float *z_hatXu1_ = cuVector(z_hat).cross(cuVector(u1)).fv();
		for (int j = 0; j < 3; j++) z_hatXu1[j] = z_hatXu1_[j];
		float z_hatXu2[3]; const float *z_hatXu2_ = cuVector(z_hat).cross(cuVector(u2)).fv();
		for (int j = 0; j < 3; j++) z_hatXu2[j] = z_hatXu2_[j];

		// [z_dot]x
		double z_dot_x[3][3] = {{0, -z_dot[2], z_dot[1]},
							 {z_dot[2], 0, -z_dot[0]},
							 {-z_dot[1], z_dot[0], 0}};
		// [z_hatXu1]x
		double z_hatXu1_x[3][3] = {{0, -z_hatXu1[2], z_hatXu1[1]},
								 {z_hatXu1[2], 0, -z_hatXu1[0]},
								 {-z_hatXu1[1], z_hatXu1[0], 0}};
		// [z_hatXu2]x
		double z_hatXu2_x[3][3] = {{0, -z_hatXu2[2], z_hatXu2[1]},
								 {z_hatXu2[2], 0, -z_hatXu2[0]},
								 {-z_hatXu2[1], z_hatXu2[0], 0}};

		// dx_hat/dgamma = (z_dot_x * x_hat) * y
		double dxdg = (z_dot_x[0][0] * y[0] + z_dot_x[0][1] * y[1] + z_dot_x[0][2] * y[2]) * grad_m[0] +
					(z_dot_x[1][0] * y[0] + z_dot_x[1][1] * y[1] + z_dot_x[1][2] * y[2]) * grad_m[1] +
					(z_dot_x[2][0] * y[0] + z_dot_x[2][1] * y[1] + z_dot_x[2][2] * y[2]) * grad_m[2];

		// dy/du1 = (z_hatXu1_x * y) * y
		double dxdu1 = (z_hatXu1_x[0][0] * y[0] + z_hatXu1_x[0][1] * y[1] + z_hatXu1_x[0][2] * y[2]) * grad_m[0] +
					(z_hatXu1_x[1][0] * y[0] + z_hatXu1_x[1][1] * y[1] + z_hatXu1_x[1][2] * y[2]) * grad_m[1] +
					(z_hatXu1_x[2][0] * y[0] + z_hatXu1_x[2][1] * y[1] + z_hatXu1_x[2][2] * y[2]) * grad_m[2];

		// dy/du2 = (z_hatXu1_x * y) * y
		double dxdu2 = (z_hatXu2_x[0][0] * y[0] + z_hatXu2_x[0][1] * y[1] + z_hatXu2_x[0][2] * y[2]) * grad_m[0] +
					(z_hatXu2_x[1][0] * y[0] + z_hatXu2_x[1][1] * y[1] + z_hatXu2_x[1][2] * y[2]) * grad_m[1] +
					(z_hatXu2_x[2][0] * y[0] + z_hatXu2_x[2][1] * y[1] + z_hatXu2_x[2][2] * y[2]) * grad_m[2];
		// dE/dx
		dEdx[i] = 2 * (m - m_bar[i]) / variance[i];
		//cout << "dxdg: " << dxdg << " dxdu1: " << dxdu1 << " dxdu2: " << dxdu2 << endl;
	
		gradient_diag[i] = normalization / variance[i];
		//totalArea += area;
		
		for (int j = n0; j < n; j++)
		{
			double nY = (Y1[j] * cY[0] + Y2[j] * cY[1] + Y3[j] * cY[2]);
			/*atomicAdd(&gradient[j], nY * dxdg * dEdx[i]);
			atomicAdd(&gradient[nCoeff + j], nY * dxdu1 * dEdx[i]);
			atomicAdd(&gradient[nCoeff * 2 + j], nY * dxdu2 * dEdx[i]);*/
			gradient_raw[size * 3 * i + j - n0] = nY * dxdg;
			gradient_raw[size * 3 * i + size + j - n0] = nY * dxdu1;
			gradient_raw[size * 3 * i + size * 2 + j - n0] = nY * dxdu2;
		}
	}
}

__global__ void dEdx_kernel(int nVertex, int degree, int deg_beg, int deg_end, double *gradient, double *gradient_raw, double *dEdx)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;

	if (x >= 0 && x < size * 3)
	{
		int k = x / size;
		int j = x % size + n0;
		int nCoeff = (degree + 1) * (degree + 1);
		gradient[nCoeff * k + j] = 0;
		for (int i = 0; i < nVertex; i++)
			gradient[nCoeff * k + j] += gradient_raw[size * 3 * i + size * k + j - n0] * dEdx[i];
	}
}

void Gradient::_ATB(double *d_A, int nr_rows_A, int nr_cols_A, double *d_B, int nr_cols_B, double *d_C)
{
	int m = nr_cols_A, n = nr_cols_B, k = nr_rows_A;
	int lda = m,ldb = n,ldc = m;
	const double alpha = 1;
	const double beta = 0;
	
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	// Do the actual multiplication
	cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
	
	cublasDestroy_v2(handle);
}

void Gradient::_DA(double *d_D, double *d_A, int nr_rows, int nr_cols, double *d_B)
{
	int blocksize = 256; // or any size up to 512
	int nblocks = (nr_rows + blocksize - 1) / blocksize;
	
	DA_kernel<<<nblocks,blocksize>>>(d_D, d_A, d_B, nr_rows, nr_cols);
}

void Gradient::_SA(double scalar, double *d_A, int nr_rows, int nr_cols)
{
	int blocksize = 256; // or any size up to 512
	int nblocks = (nr_rows + blocksize - 1) / blocksize;
	
	SA_kernel<<<nblocks,blocksize>>>(scalar, d_A, nr_rows, nr_cols);
}

void Gradient::_PA(double *d_A, double *d_B, int nr_rows, int nr_cols)
{
	int blocksize = 256; // or any size up to 512
	int nblocks = (nr_rows + blocksize - 1) / blocksize;
	
	PA_kernel<<<nblocks,blocksize>>>(d_A, d_B, nr_rows, nr_cols);
}

void Gradient::_ATDA(double *d_A, double *d_D, int nr_rows_A, int nr_cols_A, double *d_B)
{
	double *d_C;
	cudaMalloc(&d_C, nr_rows_A * nr_cols_A * sizeof(double));
	_DA(d_D, d_A, nr_rows_A, nr_cols_A, d_C);
	_ATB(d_A, nr_rows_A, nr_cols_A, d_C, nr_cols_A, d_B);
	cudaFree(d_C);
}

void Gradient::ATDA(double *h_A, double *h_D, int nr_rows_A, int nr_cols_A, double *h_B)
{
	double *d_A;
	double *d_D;
	double *d_B;
	double *d_C;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(double));
	cudaMalloc(&d_D, nr_rows_A * sizeof(double));
	cudaMalloc(&d_B, nr_cols_A * nr_cols_A * sizeof(double));
	cudaMalloc(&d_C, nr_rows_A * nr_cols_A * sizeof(double));
	cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_D, h_D, nr_rows_A * sizeof(double), cudaMemcpyHostToDevice);

	_DA(d_D, d_A, nr_rows_A, nr_cols_A, d_C);
	_ATB(d_A, nr_rows_A, nr_cols_A, d_C, nr_cols_A, d_B);

	cudaMemcpy(h_B, d_B, nr_cols_A * nr_cols_A * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_D);
	cudaFree(d_B);
	cudaFree(d_C);
}

void Gradient::ATB(double *h_A, int nr_rows_A, int nr_cols_A, double *h_B, int nr_cols_B, double *h_C)
{
	double *d_A;
	double *d_B;
	double *d_C;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(double));
	cudaMalloc(&d_B, nr_cols_A * nr_cols_B * sizeof(double));
	cudaMalloc(&d_C, nr_rows_A * nr_cols_B * sizeof(double));
	cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nr_cols_A * nr_cols_B * sizeof(double), cudaMemcpyHostToDevice);

	_ATB(d_A, nr_rows_A, nr_cols_A, d_B, nr_cols_B, d_C);

	cudaMemcpy(h_C, d_C, nr_rows_A * nr_cols_B * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void Gradient::updateGradientProperties(const float *vertex, int nVertex, const int *face, int nFace, const float *feature, const float *propertySamples, int nSamples, const double *variance, const float *property, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const double *m_bar, const float *u1, const float *u2, const int *fid, double *gradient, double *M, bool hessian)
{
	int blocksize = 256; // or any size up to 512
	int nblocks = (nSamples + blocksize - 1) / blocksize;
	
	float *d_vertex;
	int *d_face;
	float *d_feature;
	double *d_variance;
	float *d_property;
	float *d_pole;
	float *d_propertySamples;
	double *d_Y;
	double *d_coeff;
	float *d_u1;
	float *d_u2;
	double *d_gradient;
	double *d_gradient_new;
	double *d_gradient_raw;
	double *d_gradient_diag;
	double *d_dEdx;
	double *d_m_bar;
	int *d_fid;
	double *d_M;
	double *d_M_new;
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int nblocks2 = (size * 3 + blocksize - 1) / blocksize;
	
	cudaMalloc(&d_vertex, nVertex * 3 * sizeof(float));
	cudaMalloc(&d_face, nFace * 3 * sizeof(int));
	cudaMalloc(&d_feature, nSamples * sizeof(float));
	cudaMalloc(&d_variance, nSamples * sizeof(double));
	cudaMalloc(&d_property, nVertex * sizeof(float));
	cudaMalloc(&d_pole, 2 * sizeof(float));
	cudaMalloc(&d_propertySamples, nSamples * 3 * sizeof(float));
	cudaMalloc(&d_Y, nVertex * (degree + 1) * (degree + 1) * sizeof(double));
	cudaMalloc(&d_coeff, (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_m_bar, nSamples * sizeof(double));
	cudaMalloc(&d_fid, nSamples * sizeof(int));
	cudaMalloc(&d_u1, 3 * sizeof(float));
	cudaMalloc(&d_u2, 3 * sizeof(float));
	cudaMalloc(&d_gradient, (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_gradient_new, (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_gradient_raw, nSamples * (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_gradient_diag, nSamples * sizeof(double));
	cudaMalloc(&d_dEdx, nSamples * sizeof(double));
	cudaMalloc(&d_M, size * 3 * size * 3 * sizeof(double));
	cudaMalloc(&d_M_new, size * 3 * size * 3 * sizeof(double));
	
	cudaMemcpy(d_vertex, vertex, nVertex * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_face, face, nFace * 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_feature, feature, nSamples * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_variance, variance, nSamples * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_property, property, nVertex * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pole, pole, 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_propertySamples, propertySamples, nSamples * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, nVertex * (degree + 1) * (degree + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coeff, coeff, (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m_bar, m_bar, nSamples * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fid, fid, nSamples * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u1, u1, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u2, u2, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gradient, gradient, (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, M, size * 3 * size * 3 * sizeof(double), cudaMemcpyHostToDevice);

	gradient_properties_kernel<<<nblocks,blocksize>>>(d_vertex, nVertex, d_face, nFace, d_feature, d_propertySamples, nSamples, d_variance, d_property, d_pole, d_Y, d_coeff, degree, deg_beg, deg_end, normalization, d_m_bar, d_u1, d_u2, d_fid, d_gradient_new, d_gradient_raw, d_gradient_diag, d_dEdx);
	dEdx_kernel<<<nblocks2,blocksize>>>(nSamples, degree, deg_beg, deg_end, d_gradient_new, d_gradient_raw, d_dEdx);
	_SA(normalization, d_gradient_new, (degree + 1) * (degree + 1) * 3, 1);
	_PA(d_gradient, d_gradient_new, (degree + 1) * (degree + 1) * 3, 1);
	cudaMemcpy(gradient, d_gradient, (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	
	if (hessian)
	{
		_ATDA(d_gradient_raw, d_gradient_diag, nSamples, size * 3, d_M_new);
		_PA(d_M, d_M_new, size * 3, size * 3);
		cudaMemcpy(M, d_M, size * 3 * size * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	}
	//cudaMemcpy(gradient_raw, d_gradient_raw, nSamples * (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(gradient_diag, d_gradient_diag, nSamples * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_vertex);
	cudaFree(d_feature);
	cudaFree(d_variance);
	cudaFree(d_face);
	cudaFree(d_property);
	cudaFree(d_pole);
	cudaFree(d_propertySamples);
	cudaFree(d_Y);
	cudaFree(d_coeff);
	cudaFree(d_u1);
	cudaFree(d_u2);
	cudaFree(d_m_bar);
	cudaFree(d_fid);
	cudaFree(d_gradient);
	cudaFree(d_gradient_new);
	cudaFree(d_gradient_raw);
	cudaFree(d_gradient_diag);
	cudaFree(d_dEdx);
	cudaFree(d_M);
	cudaFree(d_M_new);
}

__global__ void gradient_displacement_kernel(const float *vertex0, const float *vertex1, int nVertex, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const float *u1, const float *u2, double *gradient, double *gradient_raw, double *gradient_diag, double *dEdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nVertex)
	{
		cuCoordinate coord;
		int nCoeff = (degree + 1) * (degree + 1);
		int n = (deg_end + 1) * (deg_end + 1);
		int n0 = deg_beg * deg_beg;
		int size = n - n0;

		const float *x_bar = &vertex0[i * 3];
		double delta[3] = {0, 0, 0};
		for (int j = 0; j < nCoeff; j++)
		{
			delta[0] += Y[nCoeff * i + j] * coeff[j];
			delta[1] += Y[nCoeff * i + j] * coeff[nCoeff + j];
			delta[2] += Y[nCoeff * i + j] * coeff[nCoeff * 2 + j];
		}
		
		// z
		float z_hat[3];
		coord.sph2cart(pole[0], pole[1], z_hat);
		float z_dot_orth[3]; const float *z_dot_orth_ = (cuVector(u1) * (float)delta[1] + cuVector(u2) * (float)delta[2]).unit().fv();
		for (int j = 0; j < 3; j++) z_dot_orth[j] = z_dot_orth_[j];
		float degree = (float)sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
		cuVector Z_ddot = cuVector(z_hat).cross(cuVector(z_dot_orth));
		float z_ddot[3]; const float *z_ddot_ = Z_ddot.unit().fv();
		for (int j = 0; j < 3; j++) z_ddot[j] = z_ddot_[j];
		float rot[9];
		coord.rotation(z_ddot, degree, rot);
		float z_dot[3];
		coord.rotPoint(z_hat, rot, z_dot);

		// x_hat
		const float *x_hat = &vertex1[i * 3];
		
		cuVector X_hat(x_hat), X_bar(x_bar);
		double x_hat_dot_x_bar = X_hat * X_bar;
		
		//cout << "x_hat_dot_x_bar: " << x_hat_dot_x_bar << endl;
		double dxdg = 0, dxdu2 = 0, dxdu1 = 0;
		dEdx[i] = 0;
		gradient_diag[i] = 0;
		if (fabs(x_hat_dot_x_bar) < 0.999999)	// prevent too much divergence: 1 = x_hat and x_bar matched
		{
			// z_hatXu
			/*const float *z_hatXu1 = Vector(z_hat).cross(Vector(u1)).fv();
			const float *z_hatXu2 = Vector(z_hat).cross(Vector(u2)).fv();*/
			float z_hatXu1[3]; const float *z_hatXu1_ = cuVector(z_hat).cross(cuVector(u1)).fv();
			for (int j = 0; j < 3; j++) z_hatXu1[j] = z_hatXu1_[j];
			float z_hatXu2[3]; const float *z_hatXu2_ = cuVector(z_hat).cross(cuVector(u2)).fv();
			for (int j = 0; j < 3; j++) z_hatXu2[j] = z_hatXu2_[j];
		
			// [z_dot]x
			double z_dot_x[3][3] = {{0, -z_dot[2], z_dot[1]},
								 {z_dot[2], 0, -z_dot[0]},
								 {-z_dot[1], z_dot[0], 0}};
			// [z_hatXu1]x
			double z_hatXu1_x[3][3] = {{0, -z_hatXu1[2], z_hatXu1[1]},
									 {z_hatXu1[2], 0, -z_hatXu1[0]},
									 {-z_hatXu1[1], z_hatXu1[0], 0}};
			// [z_hatXu2]x
			double z_hatXu2_x[3][3] = {{0, -z_hatXu2[2], z_hatXu2[1]},
									 {z_hatXu2[2], 0, -z_hatXu2[0]},
									 {-z_hatXu2[1], z_hatXu2[0], 0}};

			// dx_hat/dgamma = (z_dot_x * x_hat) * x_bar
			dxdg = (z_dot_x[0][0] * x_hat[0] + z_dot_x[0][1] * x_hat[1] + z_dot_x[0][2] * x_hat[2]) * x_bar[0] +
					(z_dot_x[1][0] * x_hat[0] + z_dot_x[1][1] * x_hat[1] + z_dot_x[1][2] * x_hat[2]) * x_bar[1] +
					(z_dot_x[2][0] * x_hat[0] + z_dot_x[2][1] * x_hat[1] + z_dot_x[2][2] * x_hat[2]) * x_bar[2];
		
			// dx_hat/du1 = (z_hatXu1_x * x_hat) * x_bar
			dxdu1 = (z_hatXu1_x[0][0] * x_hat[0] + z_hatXu1_x[0][1] * x_hat[1] + z_hatXu1_x[0][2] * x_hat[2]) * x_bar[0] +
					(z_hatXu1_x[1][0] * x_hat[0] + z_hatXu1_x[1][1] * x_hat[1] + z_hatXu1_x[1][2] * x_hat[2]) * x_bar[1] +
					(z_hatXu1_x[2][0] * x_hat[0] + z_hatXu1_x[2][1] * x_hat[1] + z_hatXu1_x[2][2] * x_hat[2]) * x_bar[2];
		
			// dx_hat/du2 = (z_hatXu1_x * x_hat) * x_bar
			dxdu2 = (z_hatXu2_x[0][0] * x_hat[0] + z_hatXu2_x[0][1] * x_hat[1] + z_hatXu2_x[0][2] * x_hat[2]) * x_bar[0] +
					(z_hatXu2_x[1][0] * x_hat[0] + z_hatXu2_x[1][1] * x_hat[1] + z_hatXu2_x[1][2] * x_hat[2]) * x_bar[1] +
					(z_hatXu2_x[2][0] * x_hat[0] + z_hatXu2_x[2][1] * x_hat[1] + z_hatXu2_x[2][2] * x_hat[2]) * x_bar[2];
					
			double one_minus_x_hat_dot_x_bar_sq = 1.0 - x_hat_dot_x_bar * x_hat_dot_x_bar;
			
			// dE/dx
			dEdx[i] = 2 * acos(x_hat_dot_x_bar);
			double drdx = -1.0 / sqrt(one_minus_x_hat_dot_x_bar_sq);
			dxdg *= drdx;
			dxdu1 *= drdx;
			dxdu2 *= drdx;
			gradient_diag[i] = normalization;
		}
		
		for (int j = n0; j < n; j++)
		{
			/*atomicAdd(&gradient[j], Y[nCoeff * i + j] * dxdg * dEdx[i]);
			atomicAdd(&gradient[nCoeff + j], Y[nCoeff * i + j] * dxdu1 * dEdx[i]);
			atomicAdd(&gradient[nCoeff * 2 *  + j], Y[nCoeff * i + j] * dxdu2 * dEdx[i]);*/
			gradient_raw[size * 3 * i + j - n0] = Y[nCoeff * i + j] * dxdg;
			gradient_raw[size * 3 * i + size + j - n0] = Y[nCoeff * i + j] * dxdu1;
			gradient_raw[size * 3 * i + size * 2 + j - n0] = Y[nCoeff * i + j] * dxdu2;
		}
	}
}

void Gradient::updateGradientDsiplacement(const float *vertex0, const float *vertex1, int nVertex, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const float *u1, const float *u2, double *gradient, double *M, bool hessian)
{
	int blocksize = 256; // or any size up to 512
	int nblocks = (nVertex + blocksize - 1) / blocksize;
	
	float *d_vertex0;
	float *d_vertex1;
	float *d_pole;
	double *d_Y;
	double *d_coeff;
	float *d_u1;
	float *d_u2;
	double *d_gradient;
	double *d_gradient_new;
	double *d_gradient_raw;
	double *d_gradient_diag;
	double *d_dEdx;
	double *d_M;
	double *d_M_new;
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int nblocks2 = (size * 3 + blocksize - 1) / blocksize;
	
	cudaMalloc(&d_vertex0, nVertex * 3 * sizeof(float));
	cudaMalloc(&d_vertex1, nVertex * 3 * sizeof(float));
	cudaMalloc(&d_pole, 2 * sizeof(float));
	cudaMalloc(&d_Y, nVertex * (degree + 1) * (degree + 1) * sizeof(double));
	cudaMalloc(&d_coeff, (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_u1, 3 * sizeof(float));
	cudaMalloc(&d_u2, 3 * sizeof(float));
	cudaMalloc(&d_gradient, (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_gradient_new, (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_gradient_raw, nVertex * (degree + 1) * (degree + 1) * 3 * sizeof(double));
	cudaMalloc(&d_gradient_diag, nVertex * sizeof(double));
	cudaMalloc(&d_dEdx, nVertex * sizeof(double));
	cudaMalloc(&d_M, size * 3 * size * 3 * sizeof(double));
	cudaMalloc(&d_M_new, size * 3 * size * 3 * sizeof(double));
	
	cudaMemcpy(d_vertex0, vertex0, nVertex * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertex1, vertex1, nVertex * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pole, pole, 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, nVertex * (degree + 1) * (degree + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coeff, coeff, (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u1, u1, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u2, u2, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gradient, gradient, (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, M, size * 3 * size * 3 * sizeof(double), cudaMemcpyHostToDevice);

	gradient_displacement_kernel<<<nblocks,blocksize>>>(d_vertex0, d_vertex1, nVertex, d_pole, d_Y, d_coeff, degree, deg_beg, deg_end, normalization, d_u1, d_u2, d_gradient_new, d_gradient_raw, d_gradient_diag, d_dEdx);
	dEdx_kernel<<<nblocks2,blocksize>>>(nVertex, degree, deg_beg, deg_end, d_gradient_new, d_gradient_raw, d_dEdx);
	_SA(normalization, d_gradient_new, (degree + 1) * (degree + 1) * 3, 1);
	_PA(d_gradient, d_gradient_new, (degree + 1) * (degree + 1) * 3, 1);
	cudaMemcpy(gradient, d_gradient, (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyDeviceToHost);

	if (hessian)
	{
		_ATDA(d_gradient_raw, d_gradient_diag, nVertex, size * 3, d_M_new);
		_PA(d_M, d_M_new, size * 3, size * 3);
		cudaMemcpy(M, d_M, size * 3 * size * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	}
	//cudaMemcpy(gradient_raw, d_gradient_raw, nVertex * (degree + 1) * (degree + 1) * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(gradient_diag, d_gradient_diag, nVertex * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_vertex0);
	cudaFree(d_vertex1);
	cudaFree(d_pole);
	cudaFree(d_Y);
	cudaFree(d_coeff);
	cudaFree(d_u1);
	cudaFree(d_u2);
	cudaFree(d_gradient);
	cudaFree(d_gradient_new);
	cudaFree(d_gradient_raw);
	cudaFree(d_gradient_diag);
	cudaFree(d_dEdx);
	cudaFree(d_M);
	cudaFree(d_M_new);
}
