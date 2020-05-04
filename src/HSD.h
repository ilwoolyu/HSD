/*************************************************
*	HSD.h
*
*	Release: Sep 2016
*	Update: Apr 2020
*
*	University of North Carolina at Chapel Hill
*	Department of Computer Science
*	
*	Ilwoo Lyu, ilwoolyu@cs.unc.edu
*************************************************/

#pragma once
#include <algorithm>
#include <vector>
#include "Mesh.h"
#include "AABB_Sphere.h"
#include "SurfaceUtil.h"

#ifdef _USE_CUDA_BLAS
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda/grad.h"
#endif

using namespace std;

class HSD
{
public:
	HSD(void);
	HSD(const char **sphere, int nSubj, const char **property, int nProperties, const char **output, const char **outputcoeff, const float *weight, int deg = 5, const char **landmark = NULL, float weightMap = 0.1, float weightLoc = 0, float idprior = 200, const char **coeff = NULL, const char **surf = NULL, int maxIter = 50, const bool *fixedSubj = NULL, int icosahedron = 7, bool realtimeCoeff = false, const char *tmpVariance = NULL, bool guess = true, const char *ico_mesh = NULL, int nCThreads = 1, bool resampling = false);
	~HSD(void);
	void run(void);
	void saveCoeff(const char *filename, int id);
	void saveSphere(const char *filename, int id);
	double cost(int subj = -1);
	double optimalMeanCost(double *coeff, int subj);
	
private:
	// class members for initilaization
	void init(const char **sphere, const char **property, const float *weight, const char **landmark, float weightLoc, const char **coeff, const char **surf, int samplingDegree = 3, const bool *fixedSubj = NULL, const char *tmpVariance = NULL, const char *ico_mesh = NULL, int nCThreads = 1);
	void initSphericalHarmonics(int subj, const char **coeff, double *Y = NULL);
	string initTriangleFlipping(int subj);
	string initProperties(int subj, const char **property, int nLines, AABB_Sphere *tree = NULL, Mesh *sphere = NULL, int nHeaderLines = 0);
	string initLandmarks(int subj, const char **landmark);
	void initTangentPlane(int subj);
	string initArea(int subj);
	void initPairwise(const char *tmpVariance);
	void guessInitCoeff(void);
	int icosahedron(int degree, Mesh *mesh);
	int sphericalCoord(int degree);

	void optimization(void);
	void updateGradient(int deg_beg, int deg_end, double lambda, int subj = -1);
	void updateGradientLandmark(int deg_beg, int deg_end, int subj = -1);
	void updateGradientProperties(int deg_beg, int deg_end, int subj = -1);
	void updateGradientProperties_cuda(int deg_beg, int deg_end, int subj = -1, int sid = 0, int ssid = 0);
	void updateGradientDisplacement(int deg_beg, int deg_end, int subj = -1);
	void updateGradientDisplacement_cuda(int deg_beg, int deg_end, int subj = -1, int sid = 0, int ssid = 0);
	void updateHessian(int deg_beg, int deg_end, int nSamples, int subj);
	void updateNewGradient(int deg_beg, int deg_end, double lambda, int subj);
	void updateLandmark(int subj = -1);
	void updateProperties(int subj = -1);
	void updatePropertyStats(void);
	void updateArea(int subj);
	void updateDisplacement(int subj_id = -1, int degree = 0);
	void inverse(double *M, int dim, int *ipiv, double *work);
	void linear(double *A, double *b, int dim, int *ipiv, double *work);
	void ATDA(double *A, double *D, int nr_rows_A, int nr_cols_A, double *B);
	void ATB(double *A, int nr_rows_A, int nr_cols_A, double *B, int nr_cols_B, double *C);
	void minGradientDescent(int deg_beg, int deg_end, int subj = -1);
	double varLandmarks(int subj = -1);
	double varProperties(int subj = -1);
	double varArea(int subj = -1);
	double varDisplacement(int subj = -1);
	double trace(int subj = -1);
	float propertyInterpolation(float *refMap, int index, float *coeff, Mesh *mesh);
	double propertyInterpolation(double *refMap, int index, float *coeff, Mesh *mesh);
	bool testTriangleFlip(Mesh *mesh, const bool *flip, float neg_area_threshold);

	// deformation field reconstruction
	void updateDeformation(int subject, bool enforce = false);
	bool updateCoordinate(const float *v0, float *v1, const double *Y, const double *coeff, float degree, const float *pole, const float *tan1, const float *tan2);
	
private:
	struct point
	{
		float p[3];
		float p0[3];	// new position after rigid
		float id;
		double *Y;
		int subject;
	};
	struct vert
	{
		Vector v;
		int id;
	};
	struct spharm
	{
		int degree;
		double *coeff;
		double *coeff_prev_step;
		double *gradient;
		double *Y;
		float *pole;
		float *tan1, *tan2;
		vector<point *> vertex;
		AABB_Sphere *tree;
		Mesh *sphere0;
		Mesh *sphere;
		Mesh *surf;
		float *property;
		int *tree_cache;
		float *meanProperty;
		float *medianProperty;
		float *maxProperty;
		float *minProperty;
		float *sdevProperty;
		float *area0;
		float *area1;
		vector<point *> landmark;
		bool *flip;
		bool fixed;
		bool isFlip;
		bool step_adjusted;
		double step;
		float neg_area_threshold;
#ifdef _USE_CUDA_BLAS
		float *vertex0;
		float *vertex1;
		int *face;
		int *neighbor;
		int *nNeighbor;
#endif
	};

	int m_nSubj;
	int m_csize;
	int m_nProperties;
	int m_nSurfaceProperties;
	int m_maxIter;
	int m_degree;
	int m_degree_inc;	// incremental degree
	int m_nDeformableSubj;
	int m_icosahedron;
	int m_nSamples;
	int m_nQuerySamples;
	int m_fine_res;
	int m_nMaxVertex;
	int m_nThreads;
	int m_nCThreads;
	
	double *m_coeff;
	double *m_coeff_prev_step;	// previous coefficients
	double *m_gradient;
	double *m_gradient_raw;
	double *m_gradient_diag;
	double *m_mean;
	double *m_variance;
	double *m_variance_area;
	double *m_ico_Y;
	bool *m_updated;
	bool m_realtime_coeff;
	bool m_pairwise;
	bool m_multi_res;
	bool m_guess;
	bool m_resampling;
	spharm *m_spharm;
	vector<float> m_propertySamples;
	
#ifdef _USE_CUDA_BLAS
	Gradient **m_cuda_grad;
	float *m_propertySamples_pinned;
#endif

	float m_mincost;
	float m_eta;
	float m_lambda1;
	float m_lambda2;

	double m_cost_lm;
	double m_cost_prop;
	double m_cost_disp;
	double m_cost_area;
	double m_cost_var;

	// work space for the entire procedure
	float *m_cov;
	float *m_feature;
	float *m_feature_weight;
	float *m_pole;
	float *m_Tbasis;
	int *m_ipiv;	// for lapack work space
	double *m_work;	// for lapack work space
	double *m_Hessian;
	double *m_Hessian_work;
	double *m_gradient_new;
	double *m_gradient_work;

	// tic
	int nIter;
	int nSuccessIter;

	// output list
	const char **m_output;
	const char **m_outputcoeff;
	
	Mesh *m_ico_mesh;
};

class cost_function_subj
{
public:
    cost_function_subj (HSD *instance, int _subj)
    {
        m_instance = instance;
        subj = _subj;
    }

    double operator () (double *arg)
    {
		double cost = m_instance->optimalMeanCost(arg, subj);
        return cost;
    }

private:
	HSD *m_instance;
	int subj;
};
