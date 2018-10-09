class Gradient
{
public:
	Gradient();
	~Gradient();
	static void updateGradientProperties(const float *vertex, int nVertex, const int *face, int nFace, const float *feature, const float *propertySamples, int nSamples, const double *variance, const float *property, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const double *m_bar, const float *u1, const float *u2, const int *fid, double *gradient, double *M, bool hessian);
	static void updateGradientDsiplacement(const float *vertex0, const float *vertex1, int nVertex, const float *pole, const double *Y, const double *coeff, int degree, int deg_beg, int deg_end, double normalization, const float *u1, const float *u2, double *gradient, double *M, bool hessian);
	static void ATB(double *h_A, int nr_rows_A, int nr_cols_A, double *h_B, int nr_cols_B, double *h_C);
	static void ATDA(double *h_A, double *h_D, int nr_rows_A, int nr_cols_A, double *h_B);
private:
	static void _ATB(double *d_A, int nr_rows_A, int nr_cols_A, double *d_B, int nr_cols_B, double *d_C);
	static void _DA(double *d_D, double *d_A, int nr_rows, int nr_cols, double *d_B);
	static void _SA(double scalar, double *d_A, int nr_rows, int nr_cols);
	static void _PA(double *d_A, double *d_B, int nr_rows, int nr_cols);
	static void _ATDA(double *d_A, double *d_D, int nr_rows_A, int nr_cols_A, double *d_B);
};
