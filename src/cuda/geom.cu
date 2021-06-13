__device__ cuVector::cuVector(void)
{
	m_cuVector[0] = m_cuVector[1] = m_cuVector[2] = 0;
}

__device__ cuVector::cuVector(const float *v)
{
	m_cuVector[0] = v[0];
	m_cuVector[1] = v[1];
	m_cuVector[2] = v[2];
}

__device__ cuVector::cuVector(const double *v)
{
	m_cuVector[0] = (float)v[0];
	m_cuVector[1] = (float)v[1];
	m_cuVector[2] = (float)v[2];
}

__device__ cuVector::cuVector(const float *v1, const float *v2)
{
	m_cuVector[0] = v2[0] - v1[0];
	m_cuVector[1] = v2[1] - v1[1];
	m_cuVector[2] = v2[2] - v1[2];
}

__device__ cuVector::cuVector(const float v1, const float v2, const float v3)
{
	m_cuVector[0] = v1;
	m_cuVector[1] = v2;
	m_cuVector[2] = v3;
}

__device__ cuVector::cuVector(const int v1, const int v2, const int v3)
{
	m_cuVector[0] = (float)v1;
	m_cuVector[1] = (float)v2;
	m_cuVector[2] = (float)v3;
}

__device__ cuVector::~cuVector(void)
{
}

__device__ cuVector cuVector::cross(const cuVector v)
{
	cuVector p;
	p.m_cuVector[0] = m_cuVector[1] * v.m_cuVector[2] - m_cuVector[2] * v.m_cuVector[1];
	p.m_cuVector[1] = m_cuVector[2] * v.m_cuVector[0] - m_cuVector[0] * v.m_cuVector[2];
	p.m_cuVector[2] = m_cuVector[0] * v.m_cuVector[1] - m_cuVector[1] * v.m_cuVector[0];
	return p;
}

__device__ float cuVector::norm(void)
{
	return sqrt(m_cuVector[0] * m_cuVector[0] + m_cuVector[1] * m_cuVector[1] + m_cuVector[2] * m_cuVector[2]);
}

__device__ cuVector cuVector::unit(void)
{
	float s = norm();
	if (s > 0)
	{
		m_cuVector[0] /= s;
		m_cuVector[1] /= s;
		m_cuVector[2] /= s;
	}
	return *this;
}

__device__ const float * cuVector::fv(void)
{
	return m_cuVector;
}

__device__ cuVector cuVector::operator +(const cuVector &v)
{
	return cuVector(m_cuVector[0] + v.m_cuVector[0], m_cuVector[1] + v.m_cuVector[1], m_cuVector[2] + v.m_cuVector[2]);
}

__device__ cuVector cuVector::operator -(const cuVector &v)
{
	return cuVector(m_cuVector[0] - v.m_cuVector[0], m_cuVector[1] - v.m_cuVector[1], m_cuVector[2] - v.m_cuVector[2]);
}

__device__ cuVector cuVector::operator -(void)
{
	return cuVector(-m_cuVector[0], -m_cuVector[1], -m_cuVector[2]);
}

__device__ cuVector cuVector::operator *(const float v)
{
	return cuVector(m_cuVector[0] * v, m_cuVector[1] * v, m_cuVector[2] * v);
}

__device__ float cuVector::operator *(const cuVector &v)
{
	return m_cuVector[0] * v.m_cuVector[0] + m_cuVector[1] * v.m_cuVector[1] + m_cuVector[2] * v.m_cuVector[2];
}

__device__ cuVector cuVector::operator /(const float v)
{
	return cuVector(m_cuVector[0] / v, m_cuVector[1] / v, m_cuVector[2] / v);
}

__device__ cuVector cuVector::operator =(const float *v)
{
	m_cuVector[0] = v[0];
	m_cuVector[1] = v[1];
	m_cuVector[2] = v[2];
	return *this;
}

__device__ cuVector cuVector::operator =(const float v)
{
	m_cuVector[0] = v;
	m_cuVector[1] = 0;
	m_cuVector[2] = 0;
	return *this;
}

__device__ const cuVector & cuVector::operator +=(const cuVector &v)
{
	m_cuVector[0] += v.m_cuVector[0];
	m_cuVector[1] += v.m_cuVector[1];
	m_cuVector[2] += v.m_cuVector[2];
	return *this;
}

__device__ const cuVector & cuVector::operator -=(const cuVector &v)
{
	m_cuVector[0] -= v.m_cuVector[0];
	m_cuVector[1] -= v.m_cuVector[1];
	m_cuVector[2] -= v.m_cuVector[2];
	return *this;
}

__device__ const cuVector & cuVector::operator *=(const float v)
{
	m_cuVector[0] *= v; m_cuVector[1] *= v; m_cuVector[2] *= v;
	return *this;
}

__device__ const cuVector & cuVector::operator /=(const float v)
{
	m_cuVector[0] /= v; m_cuVector[1] /= v; m_cuVector[2] /= v;
	return *this;
}

__device__ bool cuVector::operator ==(const cuVector &v)
{
	return (m_cuVector[0] == v.m_cuVector[0]) && (m_cuVector[1] == v.m_cuVector[1]) && (m_cuVector[2] == v.m_cuVector[2]);
}

__device__ float cuVector::operator[] (const int id)
{
	return m_cuVector[id];
}

__device__ bool cuVector::operator !=(const cuVector &v)
{
	return !(m_cuVector[0] == v.m_cuVector[0]) && (m_cuVector[1] == v.m_cuVector[1]) && (m_cuVector[2] == v.m_cuVector[2]);
}

__device__ bool cuVector::operator <(const cuVector &v)
{
	if (m_cuVector[0] == v.m_cuVector[0])
		if (m_cuVector[1] == v.m_cuVector[1])
			return m_cuVector[2] < v.m_cuVector[2];
		else
			return m_cuVector[1] < v.m_cuVector[1];
	else
		return m_cuVector[0] < v.m_cuVector[0];
}

__device__ bool cuVector::operator >(const cuVector &v)
{
	if (m_cuVector[0] == v.m_cuVector[0])
		if (m_cuVector[1] == v.m_cuVector[1])
			return m_cuVector[2] > v.m_cuVector[2];
		else
			return m_cuVector[1] > v.m_cuVector[1];
	else
		return m_cuVector[0] > v.m_cuVector[0];
}

__device__ cuVectorD::cuVectorD(void)
{
	m_cuVector[0] = m_cuVector[1] = m_cuVector[2] = 0;
}

__device__ cuVectorD::cuVectorD(const double *v)
{
	m_cuVector[0] = v[0];
	m_cuVector[1] = v[1];
	m_cuVector[2] = v[2];
}

__device__ cuVectorD::cuVectorD(const double *v1, const double *v2)
{
	m_cuVector[0] = v2[0] - v1[0];
	m_cuVector[1] = v2[1] - v1[1];
	m_cuVector[2] = v2[2] - v1[2];
}

__device__ cuVectorD::cuVectorD(const double v1, const double v2, const double v3)
{
	m_cuVector[0] = v1;
	m_cuVector[1] = v2;
	m_cuVector[2] = v3;
}

__device__ cuVectorD::cuVectorD(const int v1, const int v2, const int v3)
{
	m_cuVector[0] = (double)v1;
	m_cuVector[1] = (double)v2;
	m_cuVector[2] = (double)v3;
}

__device__ cuVectorD::cuVectorD(const cuVectorD &v)
{
	m_cuVector[0] = v.m_cuVector[0];
	m_cuVector[1] = v.m_cuVector[1];
	m_cuVector[2] = v.m_cuVector[2];
}

__device__ cuVectorD::~cuVectorD(void)
{
}

__device__ cuVectorD cuVectorD::cross(const cuVectorD v)
{
	cuVectorD p;
	p.m_cuVector[0] = m_cuVector[1] * v.m_cuVector[2] - m_cuVector[2] * v.m_cuVector[1];
	p.m_cuVector[1] = m_cuVector[2] * v.m_cuVector[0] - m_cuVector[0] * v.m_cuVector[2];
	p.m_cuVector[2] = m_cuVector[0] * v.m_cuVector[1] - m_cuVector[1] * v.m_cuVector[0];
	return p;
}

__device__ double cuVectorD::norm(void)
{
	return sqrt(m_cuVector[0] * m_cuVector[0] + m_cuVector[1] * m_cuVector[1] + m_cuVector[2] * m_cuVector[2]);
}

__device__ cuVectorD cuVectorD::unit(void)
{
	double s = norm();
	if (s > 0)
	{
		m_cuVector[0] /= s;
		m_cuVector[1] /= s;
		m_cuVector[2] /= s;
	}
	return *this;
}

__device__ cuVectorD cuVectorD::trunc()
{
	m_cuVector[0] = floor(m_cuVector[0]);
	m_cuVector[1] = floor(m_cuVector[1]);
	m_cuVector[2] = floor(m_cuVector[2]);

	return *this;
}

__device__ const double * cuVectorD::fv(void)
{
	return m_cuVector;
}

__device__ cuVectorD cuVectorD::operator +(const cuVectorD &v)
{
	return cuVectorD(m_cuVector[0] + v.m_cuVector[0], m_cuVector[1] + v.m_cuVector[1], m_cuVector[2] + v.m_cuVector[2]);
}

__device__ cuVectorD cuVectorD::operator -(const cuVectorD &v)
{
	return cuVectorD(m_cuVector[0] - v.m_cuVector[0], m_cuVector[1] - v.m_cuVector[1], m_cuVector[2] - v.m_cuVector[2]);
}

__device__ cuVectorD cuVectorD::operator *(const double v)
{
	return cuVectorD(m_cuVector[0] * v, m_cuVector[1] * v, m_cuVector[2] * v);
}

__device__ double cuVectorD::operator *(const cuVectorD &v)
{
	return m_cuVector[0] * v.m_cuVector[0] + m_cuVector[1] * v.m_cuVector[1] + m_cuVector[2] * v.m_cuVector[2];
}

__device__ cuVectorD cuVectorD::operator /(const double v)
{
	return cuVectorD(m_cuVector[0] / v, m_cuVector[1] / v, m_cuVector[2] / v);
}

__device__ cuVectorD cuVectorD::operator =(const double *v)
{
	m_cuVector[0] = v[0];
	m_cuVector[1] = v[1];
	m_cuVector[2] = v[2];
	return *this;
}

__device__ cuVectorD cuVectorD::operator =(const int *v)
{
	m_cuVector[0] = (double)v[0];
	m_cuVector[1] = (double)v[1];
	m_cuVector[2] = (double)v[2];
	return *this;
}

__device__ cuVectorD cuVectorD::operator =(const double v)
{
	m_cuVector[0] = v;
	m_cuVector[1] = v;
	m_cuVector[2] = v;
	return *this;
}

__device__ const double * cuVectorD::operator ()(void)
{
	return m_cuVector;
}

__device__ double cuVectorD::operator[] (int id)
{
	return m_cuVector[id];
}

__device__ const cuVectorD & cuVectorD::operator +=(const cuVectorD &v)
{
	m_cuVector[0] += v.m_cuVector[0];
	m_cuVector[1] += v.m_cuVector[1];
	m_cuVector[2] += v.m_cuVector[2];
	return *this;
}

__device__ const cuVectorD & cuVectorD::operator -=(const cuVectorD &v)
{
	m_cuVector[0] -= v.m_cuVector[0];
	m_cuVector[1] -= v.m_cuVector[1];
	m_cuVector[2] -= v.m_cuVector[2];
	return *this;
}

__device__ const cuVectorD & cuVectorD::operator *=(const double v)
{
	m_cuVector[0] *= v; m_cuVector[1] *= v; m_cuVector[2] *= v;
	return *this;
}

__device__ const cuVectorD & cuVectorD::operator /=(const double v)
{
	m_cuVector[0] /= v; m_cuVector[1] /= v; m_cuVector[2] /= v;
	return *this;
}

__device__ bool cuVectorD::operator ==(const cuVectorD &v)
{
	return (m_cuVector[0] == v.m_cuVector[0]) && (m_cuVector[1] == v.m_cuVector[1]) && (m_cuVector[2] == v.m_cuVector[2]);
}

__device__ bool cuVectorD::operator !=(const cuVectorD &v)
{
	return !(m_cuVector[0] == v.m_cuVector[0]) && (m_cuVector[1] == v.m_cuVector[1]) && (m_cuVector[2] == v.m_cuVector[2]);
}

__device__ bool cuVectorD::operator <(const cuVectorD &v)
{
	if (m_cuVector[0] == v.m_cuVector[0])
		if (m_cuVector[1] == v.m_cuVector[1])
			return m_cuVector[2] < v.m_cuVector[2];
		else
			return m_cuVector[1] < v.m_cuVector[1];
	else
		return m_cuVector[0] < v.m_cuVector[0];
}

__device__ bool cuVectorD::operator >(const cuVectorD &v)
{
	if (m_cuVector[0] == v.m_cuVector[0])
		if (m_cuVector[1] == v.m_cuVector[1])
			return m_cuVector[2] > v.m_cuVector[2];
		else
			return m_cuVector[1] > v.m_cuVector[1];
	else
		return m_cuVector[0] > v.m_cuVector[0];
}

__device__ void cuCoordinate::cart2sph(const double *v, double *phi, double *theta)
{
	// phi: azimuth, theta: elevation
	double d = v[0] * v[0] + v[1] * v[1];
	*phi = (d == 0) ? 0: atan2(v[1], v[0]);
	*theta = (v[2] == 0) ? 0: atan2(v[2], sqrt(d));
}

__device__ void cuCoordinate::cart2sph(const float *v, float *phi, float *theta)
{
	// phi: azimuth, theta: elevation
	float d = v[0] * v[0] + v[1] * v[1];
	*phi = (d == 0) ? 0: atan2(v[1], v[0]);
	*theta = (v[2] == 0) ? 0: atan2(v[2], sqrt(d));
}

__device__ void cuCoordinate::sph2cart(float phi, float theta, float *v)
{
	// phi: azimuth, theta: elevation
	v[2] = sin(theta);
	float coselev = cos(theta);
	v[0] = coselev * cos(phi);
	v[1] = coselev * sin(phi);
}

__device__ void cuCoordinate::sph2cart(double phi, double theta, double *v)
{
	// phi: azimuth, theta: elevation
	v[2] = sin(theta);
	double coselev = cos(theta);
	v[0] = coselev * cos(phi);
	v[1] = coselev * sin(phi);
}

__device__ void cuCoordinate::cart2bary(const float *a, const float *b, const float *c, const float *p, float *coeff, float err)
{
	// test dataset for debug
	/*float a[3] = {-0.6498,0.3743,0.6616};
	float b[3] = {-0.6571,0.3837,0.6488};
	float c[3] = {-0.6646,0.3675,0.6506};
	float p[3] = {-0.6572,0.3752,0.6537};
	float coeff[3];*/

	// a counter clockwise order
	cuVector A(a), B(b), C(c), P(p);
	cuVector N((B-A).cross(C-A));

	float ABC = N * N / N.norm();
	N.unit();
	coeff[0] = (B-P).cross(C-P) * N / ABC;
	coeff[1] = (C-P).cross(A-P) * N / ABC;
	//coeff[2] = (A-P).cross(B-P) * N / ABC;
	coeff[2] = 1 - coeff[0] - coeff[1];

	if (fabs(coeff[0]) < err)
	{
		coeff[0] = 0;
		coeff[1] = 1 - (P - B).norm() / (C - B).norm();
		coeff[2] = 1 - coeff[1];
		if (fabs(coeff[1]) < err)
		{
			coeff[1] = 0;
			coeff[2] = 1;
		}
		else if (fabs(coeff[2]) < err)
		{
			coeff[1] = 1;
			coeff[2] = 0;
		}
	}
	else if (fabs(coeff[1]) < err)
	{
		coeff[1] = 0;
		coeff[2] = 1 - (P - C).norm() / (A - C).norm();
		coeff[0] = 1 - coeff[2];
		if (fabs(coeff[2]) < err)
		{
			coeff[2] = 0;
			coeff[0] = 1;
		}
		else if (fabs(coeff[0]) < err)
		{
			coeff[2] = 1;
			coeff[0] = 0;
		}
	}
	else if (fabs(coeff[2]) < err)
	{
		coeff[2] = 0;
		coeff[0] = 1 - (P - A).norm() / (B - A).norm();
		coeff[1] = 1 - coeff[0];
		if (fabs(coeff[0]) < err)
		{
			coeff[0] = 0;
			coeff[1] = 1;
		}
		else if (fabs(coeff[1]) < err)
		{
			coeff[0] = 1;
			coeff[1] = 0;
		}
	}
	// debug
	/*printf("coeff: %f %f %f\n",coeff[0],coeff[1],coeff[2]);
	cuVector PP = A * coeff[0] + B * coeff[1] + C * coeff[2];
	printf("recons: %f %f %f\n", PP.fv()[0],PP.fv()[1],PP.fv()[2]);*/
}
__device__ void cuCoordinate::cart2bary(const double *a, const double *b, const double *c, const double *p, double *coeff, double err)
{
	// a counter clockwise order
	cuVectorD A(a), B(b), C(c), P(p);
	cuVectorD N((B-A).cross(C-A));

	double ABC = N * N / N.norm();
	N.unit();
	coeff[0] = (B-P).cross(C-P) * N / ABC;
	coeff[1] = (C-P).cross(A-P) * N / ABC;
	coeff[2] = 1 - coeff[0] - coeff[1];
	
	if (fabs(coeff[0]) < err)
	{
		coeff[0] = 0;
		coeff[1] = 1 - (P - B).norm() / (C - B).norm();
		coeff[2] = 1 - coeff[1];
		if (fabs(coeff[1]) < err)
		{
			coeff[1] = 0;
			coeff[2] = 1;
		}
		else if (fabs(coeff[2]) < err)
		{
			coeff[1] = 1;
			coeff[2] = 0;
		}
	}
	else if (fabs(coeff[1]) < err)
	{
		coeff[1] = 0;
		coeff[2] = 1 - (P - C).norm() / (A - C).norm();
		coeff[0] = 1 - coeff[2];
		if (fabs(coeff[2]) < err)
		{
			coeff[2] = 0;
			coeff[0] = 1;
		}
		else if (fabs(coeff[0]) < err)
		{
			coeff[2] = 1;
			coeff[0] = 0;
		}
	}
	else if (fabs(coeff[2]) < err)
	{
		coeff[2] = 0;
		coeff[0] = 1 - (P - A).norm() / (B - A).norm();
		coeff[1] = 1 - coeff[0];
		if (fabs(coeff[0]) < err)
		{
			coeff[0] = 0;
			coeff[1] = 1;
		}
		else if (fabs(coeff[1]) < err)
		{
			coeff[0] = 1;
			coeff[1] = 0;
		}
	}
}

__device__ void cuCoordinate::rotPoint(const float *p0, const float *mat, float *p1)
{
	for (int i = 0; i < 3; i++)
		p1[i] = mat[i * 3 + 0] * p0[0] + mat[i * 3 + 1] * p0[1] + mat[i * 3 + 2] * p0[2];
}
__device__ void cuCoordinate::rotPointInv(const float *p0, const float *mat, float *p1)
{
	for (int i = 0; i < 3; i++)
		p1[i] = mat[i] * p0[0] + mat[i + 3] * p0[1] + mat[i + 6] * p0[2];
}
__device__ void cuCoordinate::rotPoint(const double *p0, const double *mat, double *p1)
{
	for (int i = 0; i < 3; i++)
		p1[i] = mat[i * 3 + 0] * p0[0] + mat[i * 3 + 1] * p0[1] + mat[i * 3 + 2] * p0[2];
}
__device__ void cuCoordinate::rotPointInv(const double *p0, const double *mat, double *p1)
{
	for (int i = 0; i < 3; i++)
		p1[i] = mat[i] * p0[0] + mat[i + 3] * p0[1] + mat[i + 6] * p0[2];
}
__device__ void cuCoordinate::rotation(const float *axis, const float theta, float *mat)
{
	memset(mat, 0, sizeof(float) * 9);

	mat[0] = 1; mat[4] = 1; mat[8] = 1;

	cuVector RAxis(axis);
	if (RAxis.norm() == 0) return;
	RAxis.unit();

	float A[9] = {0, -RAxis[2], RAxis[1], RAxis[2], 0, -RAxis[0], -RAxis[1], RAxis[0], 0};

	// A * sin(-theta)
	for (int i = 0; i < 9; i++)
		mat[i] += A[i] * sin(theta);
	// A * A * (1 - cos(-theta))
	for (int i = 0; i < 9; i++)
		for (int j = 0; j < 3; j++)
			mat[i] += A[j + (i / 3) * 3] * A[j * 3 + i % 3] * (1 - cos(theta));
}
__device__ void cuCoordinate::rotation(const double *axis, const double theta, double *mat)
{
	memset(mat, 0, sizeof(double) * 9);

	mat[0] = 1; mat[4] = 1; mat[8] = 1;

	cuVectorD RAxis(axis);
	if (RAxis.norm() == 0) return;
	RAxis.unit();

	double A[9] = {0, -RAxis[2], RAxis[1], RAxis[2], 0, -RAxis[0], -RAxis[1], RAxis[0], 0};

	// A * sin(-theta)
	for (int i = 0; i < 9; i++)
		mat[i] += A[i] * sin(theta);
	// A * A * (1 - cos(-theta))
	for (int i = 0; i < 9; i++)
		for (int j = 0; j < 3; j++)
			mat[i] += A[j + (i / 3) * 3] * A[j * 3 + i % 3] * (1 - cos(theta));
}
