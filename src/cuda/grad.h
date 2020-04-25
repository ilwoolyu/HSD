#include <cublas_v2.h>
#include <cusparse_v2.h>

class Gradient
{
public:
	Gradient();
	Gradient(int nVertex, int nFace, int nSamples, int degree);
	~Gradient();
	void updateGradientProperties(const float *vertex, int nVertex, const int *face, int nFace, const float *feature, const float *propertySamples, int nSamples, const double *variance, const float *property, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const double *m_bar, const float *u1, const float *u2, const int *fid, double *gradient, double *M, bool hessian, int sid = 0);
	void updateGradientDsiplacement(const float *vertex0, const float *vertex1, int nVertex, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const float *u1, const float *u2, double *gradient, double *M, bool hessian, int sid = 0);
	static void ATB(double *h_A, int nr_rows_A, int nr_cols_A, double *h_B, int nr_cols_B, double *h_C);
	static void ATDA(double *h_A, double *h_D, int nr_rows_A, int nr_cols_A, double *h_B);
private:
	static void _ATB(double *d_A, int nr_rows_A, int nr_cols_A, double *d_B, int nr_cols_B, double *d_C, cublasHandle_t handle, cudaStream_t stream = 0);
	static void _DA(double *d_D, double *d_A, int nr_rows, int nr_cols, double *d_B, cudaStream_t stream = 0);
	static void _SA(double scalar, double *d_A, int nr_rows, int nr_cols, cudaStream_t stream = 0);
	static void _PA(double *d_A, double *d_B, int nr_rows, int nr_cols, cudaStream_t stream = 0);
	void _ATDA(double *d_A, double *d_D, int nr_rows_A, int nr_cols_A, double *d_B, double *d_C, int sid = 0);
	void allocMemory(int nVertex, int nFace, int nSamples, int degree);
	void freeMemory(void);

private:
	float *d_vertex0;
	float *d_vertex1;
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
	double *d_gradient_work;
	double *d_dEdx;
	double *d_m_bar;
	int *d_fid;
	double *d_M;
	double *d_M_new;
	int nStream;
	int nMaxVertex0;
	int nMaxVertex1;
	int nMaxFace;
	cublasHandle_t *handle;
	cudaStream_t *stream;
};
