class cuVector
{
public:
	__device__ cuVector(void);
	__device__ cuVector(const float *v);
	__device__ cuVector(const double *v);
	__device__ cuVector(const float *v1, const float *v2);
	__device__ cuVector(const float v1, const float v2, const float v3);
	__device__ cuVector(const int v1, const int v2, const int v3);
	__device__ ~cuVector(void);
	__device__ cuVector cross(const cuVector v);
	__device__ float norm(void);
	__device__ cuVector unit(void);
	__device__ cuVector operator +(const cuVector &v);
	__device__ cuVector operator -(const cuVector &v);
	__device__ cuVector operator -(void);
	__device__ cuVector operator *(const float v);
	__device__ float operator *(const cuVector &v);
	__device__ cuVector operator /(const float v);
	__device__ cuVector operator =(const float *v);
	__device__ cuVector operator =(const float v);
	__device__ bool operator ==(const cuVector &v);
	__device__ bool operator !=(const cuVector &v);
	__device__ bool operator <(const cuVector &v);
	__device__ bool operator >(const cuVector &v);
	__device__ float operator [](const int id);
	__device__ const cuVector &operator +=(const cuVector &v);
	__device__ const cuVector &operator -=(const cuVector &v);
	__device__ const cuVector &operator *=(const float v);
	__device__ const cuVector &operator /=(const float v);
	__device__ const float *fv(void);

private:
	float m_cuVector[3];
};

class cuVectorD
{
public:
	__device__ cuVectorD(void);
	__device__ cuVectorD(const double *v);
	__device__ cuVectorD(const double *v1, const double *v2);
	__device__ cuVectorD(const double v1, const double v2, const double v3);
	__device__ cuVectorD(const int v1, const int v2, const int v3);
	__device__ cuVectorD(const cuVectorD &v);
	__device__ ~cuVectorD(void);
	__device__ cuVectorD cross(const cuVectorD v);
	__device__ double norm(void);
	__device__ cuVectorD unit(void);
	__device__ cuVectorD trunc();
	__device__ const double * fv(void);
	__device__ cuVectorD operator +(const cuVectorD &v);
	__device__ cuVectorD operator -(const cuVectorD &v);
	__device__ cuVectorD operator *(const double v);
	__device__ double operator *(const cuVectorD &v);
	__device__ cuVectorD operator /(const double v);
	__device__ cuVectorD operator =(const double *v);
	__device__ cuVectorD operator =(const int *v);
	__device__ cuVectorD operator =(const double v);
	__device__ const double *operator ()(void);
	__device__ double operator[] (int id);
	__device__ const cuVectorD & operator +=(const cuVectorD &v);
	__device__ const cuVectorD & operator -=(const cuVectorD &v);
	__device__ const cuVectorD & operator *=(const double v);
	__device__ const cuVectorD & operator /=(const double v);
	__device__ bool operator ==(const cuVectorD &v);
	__device__ bool operator !=(const cuVectorD &v);
	__device__ bool operator <(const cuVectorD &v);
	__device__ bool operator >(const cuVectorD &v);
private:
	double m_cuVector[3];
};

class cuCoordinate
{
public:
	__device__ void cart2sph(const double *v, double *phi, double *theta);
	__device__ void cart2sph(const float *v, float *phi, float *theta);
	__device__ void sph2cart(float phi, float theta, float *v);
	__device__ void sph2cart(double phi, double theta, double *v);
	__device__ void cart2bary(float *a, float *b, float *c, float *p, float *coeff, float err = 0);
	__device__ void cart2bary(double *a, double *b, double *c, double *p, double *coeff, double err = 0);
	__device__ void rotPoint(const float *p0, const float *mat, float *p1);
	__device__ void rotPoint(const double *p0, const double *mat, double *p1);
	__device__ void rotPointInv(const float *p0, const float *mat, float *p1);
	__device__ void rotPointInv(const double *p0, const double *mat, double *p1);
	__device__ void rotation(const float *axis, const float theta, float *mat);
	__device__ void rotation(const double *axis, const double theta, double *mat);
};
