/*************************************************
*	HSD.cpp
*
*	Release: Sep 2016
*	Update: May 2020
*
*	University of North Carolina at Chapel Hill
*	Department of Computer Science
*	
*	Ilwoo Lyu, ilwoolyu@cs.unc.edu
*************************************************/

#include <cstring>
#include <lapacke.h>
#include <cblas.h>
#include <map>
#include "HSD.h"
#include "SphericalHarmonics.h"
#include "bobyqa.h"

HSD::HSD(void)
{
	m_maxIter = 0;
	m_nSubj = 0;
	m_mincost = FLT_MAX;
	m_nProperties = 0;
	m_nSurfaceProperties = 0;
	m_output = NULL;
	m_outputcoeff = NULL;
	m_pole = NULL;
	m_ico_Y = NULL;
	m_degree = 0;
	m_eta = 1;
	m_lambda1 = 1;
	m_lambda2 = 200;
	m_degree_inc = 0;	// starting degree for the incremental optimization
	m_icosahedron = 1;
	m_fine_res = 6;
	m_guess = true;
	m_realtime_coeff = false;
	m_pairwise = false;
	m_resampling = false;
	m_multi_res = true;
}

HSD::HSD(const char **sphere, int nSubj, const char **property, int nProperties, const char **output, const char **outputcoeff, const float *weight, int deg, const char **landmark, float weightMap, float weightLoc, float idprior, const char **coeff, const char **surf, int maxIter, const bool *fixedSubj, int icosahedron, bool realtimeCoeff, const char *tmpVariance, bool guess, const char *ico_mesh, int nCThreads, bool resampling)
{
	m_maxIter = maxIter;
	m_nSubj = nSubj;
	m_mincost = FLT_MAX;
	m_nProperties = nProperties;
	m_nSurfaceProperties = (weightLoc > 0)? 3: 0;
	m_output = output;
	m_outputcoeff = outputcoeff;
	m_degree = deg;
	m_lambda1 = 1;
	m_lambda2 = idprior;
	m_fine_res = 6;
	m_guess = guess;
	m_pole = NULL;
	m_ico_Y = NULL;
	m_eta = (landmark == NULL)? 1: weightMap;
	m_degree_inc = 0;	// starting degree for the incremental optimization
	m_realtime_coeff = realtimeCoeff;
	m_icosahedron = icosahedron;
	m_pairwise = false;
	m_resampling = resampling;
	m_multi_res = true;
	init(sphere, property, weight, landmark, weightLoc, coeff, surf, icosahedron, fixedSubj, tmpVariance, ico_mesh, nCThreads);
}

HSD::~HSD(void)
{
	delete [] m_cov;
	delete [] m_feature_weight;
	delete [] m_updated;
	delete [] m_work;
	delete [] m_ipiv;
	delete [] m_Hessian_work;
	delete [] m_gradient_new;
#ifdef _USE_CUDA_BLAS
	for (int i = 0; i < m_nCThreads; i++)
		delete m_cuda_grad[i];
	delete [] m_cuda_grad;
	cudaFreeHost(m_feature);
	cudaFreeHost(m_mean);
	cudaFreeHost(m_variance);
	cudaFreeHost(m_coeff);
	cudaFreeHost(m_coeff_prev_step);
	cudaFreeHost(m_gradient);
	cudaFreeHost(m_pole);
	cudaFreeHost(m_Tbasis);
	cudaFreeHost(m_Hessian);
	cudaFreeHost(m_propertySamples_pinned);
#else
	delete [] m_feature;
	delete [] m_gradient_raw;
	delete [] m_gradient_diag;
	delete [] m_gradient_work;
	delete [] m_mean;
	delete [] m_variance;
	delete [] m_coeff;
	delete [] m_coeff_prev_step;
	delete [] m_gradient;
	delete [] m_pole;
	delete [] m_Tbasis;
	delete [] m_Hessian;
#endif
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		delete m_spharm[subj].tree;
		delete m_spharm[subj].surf;
		delete m_spharm[subj].sphere;
		delete [] m_spharm[subj].meanProperty;
		delete [] m_spharm[subj].medianProperty;
		delete [] m_spharm[subj].maxProperty;
		delete [] m_spharm[subj].minProperty;
		delete [] m_spharm[subj].sdevProperty;
		delete [] m_spharm[subj].flip;
		delete [] m_spharm[subj].area0;
		delete [] m_spharm[subj].area1;
		if (m_resampling && ((m_pairwise && !m_spharm[subj].fixed) || !m_pairwise))
			delete m_spharm[subj].sphere0;
#ifdef _USE_CUDA_BLAS
		cudaFreeHost(m_spharm[subj].tree_cache);
		cudaFreeHost(m_spharm[subj].vertex0);
		cudaFreeHost(m_spharm[subj].vertex1);
		cudaFreeHost(m_spharm[subj].property);
		if (!m_resampling)
		{
			cudaFreeHost(m_spharm[subj].face);
			cudaFreeHost(m_spharm[subj].neighbor);
			cudaFreeHost(m_spharm[subj].nNeighbor);
		}
		if ((m_resampling && m_pairwise && m_spharm[subj].fixed) || !m_resampling)
			cudaFreeHost(m_spharm[subj].Y);
#else
		delete [] m_spharm[subj].tree_cache;
		delete [] m_spharm[subj].property;
		if ((m_resampling && m_pairwise && m_spharm[subj].fixed) || !m_resampling)
			delete [] m_spharm[subj].Y;
#endif
	}
	delete [] m_spharm;
	if (m_resampling) delete [] m_ico_Y;
}

void HSD::run(void)
{
	cout << "Optimization\n";
	optimization();

	// write the solutions
	cout << "Saving outputs\n";
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		cout << "-Subject " << subj << endl;
		if (m_output != NULL && !m_spharm[subj].fixed)
		{
			updateDeformation(subj, true);	// update defomation fields
			saveSphere(m_output[subj], subj);
		}
		if (m_outputcoeff != NULL && !m_spharm[subj].fixed) saveCoeff(m_outputcoeff[subj], subj);
	}
	cout << "All done!\n";
}

void HSD::init(const char **sphere, const char **property, const float *weight, const char **landmark, float weightLoc, const char **coeff, const char **surf, int samplingDegree, const bool *fixedSubj, const char *tmpVariance, const char *ico_mesh, int nCThreads)
{
	int csize = (m_degree + 1) * (m_degree + 1);
	m_spharm = new spharm[m_nSubj];	// spharm info
	m_updated = new bool[m_nSubj];	// AABB tree cache
	//m_work = new float[m_nSubj * 3 - 1];	// workspace for eigenvalue computation
#ifdef _USE_SYSV
	m_work = new double[64 * csize * 3 * m_nSubj];	// workspace
	m_ipiv = new int[csize * 3 * m_nSubj];	// workspace
#else
	m_work = new double[csize * 3 * csize * 3 * m_nSubj];	// workspace
	m_ipiv = new int[(csize * 3 + 1) * m_nSubj];	// workspace
#endif
#ifdef _USE_CUDA_BLAS
	cudaError_t status;
#endif
#ifdef _USE_CUDA_BLAS
	status = cudaMallocHost((void**)&m_Hessian, csize * 3 * csize * 3 * m_nSubj * sizeof(double));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
#else
	m_Hessian = new double[csize * 3 * csize * 3 * m_nSubj];
#endif
	m_Hessian_work = new double[csize * 3 * csize * 3];
#ifdef _USE_SYSV
	m_gradient_new = new double[csize * 3 * m_nSubj];
#else
	m_gradient_new = new double[csize * 3 * 2 * m_nSubj];
#endif
	m_csize = (m_degree + 1) * (m_degree + 1) * m_nSubj; // total # of coefficients
#ifdef _USE_CUDA_BLAS
	status = cudaMallocHost((void**)&m_coeff, m_csize * 3 * sizeof(double));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
	status = cudaMallocHost((void**)&m_coeff_prev_step, m_csize * 3 * sizeof(double));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
	status = cudaMallocHost((void**)&m_gradient, m_csize * 3 * sizeof(double));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
	status = cudaMallocHost((void**)&m_pole, m_nSubj * 3 * 2 * sizeof(float));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
	status = cudaMallocHost((void**)&m_Tbasis, m_nSubj * 3 * 2 * sizeof(float));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
#else
	m_coeff = new double[m_csize * 3];	// how many coefficients are required: the sum of all possible coefficients
	m_coeff_prev_step = new double[m_csize * 3];	// the previous coefficients
	m_gradient = new double[m_csize * 3];
	m_pole = new float[m_nSubj * 3 * 2];	// pole information
	m_Tbasis = new float[m_nSubj * 3 * 2];	// tangent plane for the exponential map
#endif

	// set all the coefficient to zeros
	memset(m_coeff, 0, sizeof(double) * m_csize * 3);
	memset(m_coeff_prev_step, 0, sizeof(double) * m_csize * 3);
	memset(m_gradient, 0, sizeof(double) * m_csize * 3);
	memset(m_updated, 0, sizeof(bool) * m_nSubj);

	// copy fixed subject information if any
	m_nDeformableSubj = m_nSubj;		// # of subjects will be deformed
	if (fixedSubj != NULL)
	{
		m_nDeformableSubj = 0;
		for (int i = 0; i < m_nSubj; i++)
		{
			if (!fixedSubj[i])
			{
				m_spharm[i].fixed = false;
				m_nDeformableSubj++;
			}
			else
			{
				m_spharm[i].fixed = true;
			}
		}
	}
	else
	{
		for (int i = 0; i < m_nSubj; i++) m_spharm[i].fixed = false;
	}

	// icosahedron subdivision for evaluation on properties: this generates uniform sampling points over the sphere - m_propertySamples
	//if (m_nProperties + m_nSurfaceProperties > 0) icosahedron(7, m_ico_mesh);
	if (m_nProperties + m_nSurfaceProperties > 0)
	{
		m_ico_mesh = new Mesh();
		if (ico_mesh == NULL)
		{
			icosahedron(m_icosahedron, m_ico_mesh);
			switch (m_icosahedron)
			{
				case 3: m_nSamples = 642; break;
				case 4: m_nSamples = 2562; break;
				case 5: m_nSamples = 10242; break;
				case 6: m_nSamples = 40962; break;
				case 7: m_nSamples = 163842; break;
			}
		}
		else
		{
			m_ico_mesh->openFile(ico_mesh);
			m_nSamples = m_ico_mesh->nVertex();
			switch (m_nSamples)
			{
				case 642: m_icosahedron = 3; break;
				case 2562: m_icosahedron = 4; break;
				case 10242: m_icosahedron = 5; break;
				case 40962: m_icosahedron = 6; break;
				case 163842: m_icosahedron = 7; break;
				default: cout << "Fatal error: icosahedron mesh is invalid (vert#=" << m_nSamples << ")" << endl; exit(1); break;
			}
			for (int i = 0; i < m_ico_mesh->nVertex(); i++)
			{
				Vertex *v = (Vertex *)m_ico_mesh->vertex(i);	// vertex information on the sphere
				const float *v0 = v->fv();
				Vector V(v0);
				const float *p = V.unit().fv();
				m_propertySamples.push_back(p[0]);
				m_propertySamples.push_back(p[1]);
				m_propertySamples.push_back(p[2]);
			}
			cout << "Icosahedron level: " << m_icosahedron << endl;
		}
	}
	if (!m_multi_res) m_fine_res = m_icosahedron + 1;

	if (m_nDeformableSubj == 1 && m_nSubj == 2) m_pairwise = true;
	if (m_pairwise)
	{
		cout << "Running on pair-wise registration" << endl;
		m_multi_res = true;
		//m_lambda2 *= 0.32;
	}
	
	float *ico_vertex = NULL;
	int *ico_face = NULL;
	int *ico_neighbor = NULL;
	int *ico_nNeighbor = NULL;
	if (m_resampling)
	{
		int nVertex = m_ico_mesh->nVertex();
		int nFace = m_ico_mesh->nFace();
		ico_vertex = new float[nVertex * 3];
		ico_face = new int[nFace * 3];
		for (int i = 0; i < nVertex; i++)
		{
			Vertex *v = (Vertex *)m_ico_mesh->vertex(i);
			const float *v0 = v->fv();
			Vector V(v0); V.unit();
			v->setVertex(V.fv());

			memcpy(&ico_vertex[i * 3], v->fv(), 3 * sizeof(float));
		}
		for (int i = 0; i < nFace; i++)
		{
			Face *f = (Face *)m_ico_mesh->face(i);
			memcpy(&ico_face[i * 3], f->list(), 3 * sizeof(int));
		}
		m_ico_Y = new double[(m_degree + 1) * (m_degree + 1) * nVertex];
		#pragma omp parallel for
		for (int i = 0; i < nVertex; i++)
		{
			Vertex *v = (Vertex *)m_ico_mesh->vertex(i);
			const float *v0 = v->fv();
			double vd[3] = {v0[0], v0[1], v0[2]};
			SphericalHarmonics::basis(m_degree, vd, &m_ico_Y[i * (m_degree + 1) * (m_degree + 1)]);
		}
#ifdef _USE_CUDA_BLAS
		int nNeighbor = (nVertex + nFace - 2) * 2;	// use Euler number
		ico_neighbor = new int[nNeighbor * sizeof(int)];
		ico_nNeighbor = new int[nVertex * 2 * sizeof(int)];
		int c = 0;
		for (int i = 0; i < nVertex; i++)
		{
			ico_nNeighbor[i * 2] = c;
			ico_nNeighbor[i * 2 + 1] = m_ico_mesh->vertex(i)->nNeighbor();
			memcpy(&ico_neighbor[ico_nNeighbor[i * 2]], m_ico_mesh->vertex(i)->list(), sizeof(int) * ico_nNeighbor[i * 2 + 1]);
			c += ico_nNeighbor[i * 2 + 1];
		}
#endif
	}

	cout << "Initialzation of subject information\n";

	cout << "-Loading spherical mesh\n";
	int dot = m_nSubj / 20 + (m_nSubj % 20 != 0);
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		// spehre and surface information
		m_spharm[subj].sphere = new Mesh();
		if (sphere != NULL)
		{
			m_spharm[subj].sphere->openFile(sphere[subj]);
			m_spharm[subj].sphere0 = m_spharm[subj].sphere;
			// make sure a unit sphere
			//m_spharm[subj].sphere->centering();
		}
		else
		{
			cout << " Fatal error: No sphere mapping is provided!\n";
			exit(1);
		}
		if (subj % dot == 0)
		{
			cout << ".";
			fflush(stdout);
		}
	}
	cout << endl;

	#pragma omp parallel for
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		string log;
		log += "Subject " + to_string(subj) + " - " + sphere[subj] + "\n";
		log += "-Sphere information\n";
		for (int i = 0; i < m_spharm[subj].sphere->nVertex(); i++)
		{
			Vertex *v = (Vertex *)m_spharm[subj].sphere->vertex(i);	// vertex information on the sphere
			const float *v0 = v->fv();
			Vector V(v0); V.unit();
			v->setVertex(V.fv());
		}

		// landmarks
		if (landmark != NULL)
		{
			log += "-Landmark information\n";
			log += initLandmarks(subj, landmark);
		}

		if (m_nProperties + m_nSurfaceProperties > 0)
		{
			// AABB tree construction for speedup computation
			m_spharm[subj].tree = new AABB_Sphere(m_spharm[subj].sphere);
			AABB_Sphere *tree = m_spharm[subj].tree;
			Mesh *sphere = m_spharm[subj].sphere;

			int nVertex = m_spharm[subj].sphere->nVertex();
			if (m_resampling && ((m_pairwise && !m_spharm[subj].fixed) || !m_pairwise))
			{
				m_spharm[subj].sphere = new Mesh();
				m_spharm[subj].sphere->setMesh(ico_vertex, ico_face, NULL, m_ico_mesh->nVertex(), m_ico_mesh->nFace(), 0, false);
				m_spharm[subj].tree = new AABB_Sphere(m_spharm[subj].sphere);
			}
			log += "--Total vertices: " + to_string(m_spharm[subj].sphere->nVertex()) + "\n";
			log += "-AABB tree reconstruction\n";

			// property information
			if (m_nSurfaceProperties > 0)
			{
				log += "-Location information\n";
				m_spharm[subj].surf = new Mesh();
				m_spharm[subj].surf->openFile(surf[subj]);
			}
			else m_spharm[subj].surf = NULL;

			log += "-Property information\n";
			log += initProperties(subj, property, nVertex, tree, sphere);
			if (m_resampling && ((m_pairwise && !m_spharm[subj].fixed) || !m_pairwise)) delete tree;
		}
		else m_spharm[subj].tree = NULL;

		// previous spherical harmonic deformation fields
		log += "-Spherical harmonics information\n";
		initSphericalHarmonics(subj, coeff, m_ico_Y);

#ifdef _USE_CUDA_BLAS
		int nVertex = m_spharm[subj].sphere->nVertex();
		int nFace = m_spharm[subj].sphere->nFace();
		status = cudaMallocHost((void**)&m_spharm[subj].vertex0, nVertex * 3 * sizeof(float));
		if (status != cudaSuccess)
		{
			cout << "Fatal error: allocating pinned host memory" << endl;
			exit(1);
		}
		status = cudaMallocHost((void**)&m_spharm[subj].vertex1, nVertex * 3 * sizeof(float));
		if (status != cudaSuccess)
		{
			cout << "Fatal error: allocating pinned host memory" << endl;
			exit(1);
		}
		for (int i = 0; i < nVertex; i++) memcpy(&m_spharm[subj].vertex0[i * 3], m_spharm[subj].vertex[i]->p0, sizeof(float) * 3);
		if (!m_resampling)
		{
			status = cudaMallocHost((void**)&m_spharm[subj].face, nFace * 3 * sizeof(int));
			if (status != cudaSuccess)
			{
				cout << "Fatal error: allocating pinned host memory" << endl;
				exit(1);
			}
			for (int i = 0; i < nFace; i++) memcpy(&m_spharm[subj].face[i * 3], m_spharm[subj].sphere->face(i)->list(), sizeof(int) * 3);
			int nNeighbor = (nFace + nVertex - 2) * 2;	// use Euler number
			status = cudaMallocHost((void**)&m_spharm[subj].neighbor, nNeighbor * sizeof(int));
			if (status != cudaSuccess)
			{
				cout << "Fatal error: allocating pinned host memory" << endl;
				exit(1);
			}
			status = cudaMallocHost((void**)&m_spharm[subj].nNeighbor, nVertex * 2 * sizeof(int));
			if (status != cudaSuccess)
			{
				cout << "Fatal error: allocating pinned host memory" << endl;
				exit(1);
			}
			int c = 0;
			for (int i = 0; i < nVertex; i++)
			{
				m_spharm[subj].nNeighbor[i * 2] = c;
				m_spharm[subj].nNeighbor[i * 2 + 1] = m_spharm[subj].sphere->vertex(i)->nNeighbor();
				memcpy(&m_spharm[subj].neighbor[m_spharm[subj].nNeighbor[i * 2]], m_spharm[subj].sphere->vertex(i)->list(), sizeof(int) * m_spharm[subj].nNeighbor[i * 2 + 1]);
				c += m_spharm[subj].nNeighbor[i * 2 + 1];
			}
		}
		else
		{
			m_spharm[subj].face = NULL;
			m_spharm[subj].neighbor = NULL;
			m_spharm[subj].nNeighbor = NULL;
		}
#endif

		// optimal pole and tangent plane
		initTangentPlane(subj);

		// triangle flipping
		m_spharm[subj].flip = NULL;
		log += "-Triangle flipping\n";
		log += initTriangleFlipping(subj);
		
		// area
		m_spharm[subj].area0 = NULL;
		m_spharm[subj].area1 = NULL;
		log += "-Surface area\n";
		log += initArea(subj);
		
		log += "----------\n";
		
		cout << log;
	}

	cout << "Computing weight terms" << endl;
	int nLandmark = m_spharm[0].landmark.size() * 3;	// # of landmarks: we assume all the subject has the same number, which already is checked above.
	int nSamples = m_propertySamples.size() / 3;	// # of sampling points for property map agreement
#ifdef _USE_CUDA_BLAS
	status = cudaMallocHost((void**)&m_propertySamples_pinned, nSamples * 3 * sizeof(float));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
	memcpy(m_propertySamples_pinned, (float *)&m_propertySamples[0], nSamples * 3 * sizeof(float));
#endif
	m_nQuerySamples = nSamples;
	
	// weights for covariance matrix computation
	int nTotalProperties = m_nProperties + m_nSurfaceProperties;	// if location information is provided, total number = # of property + 3 -> (x, y, z location)
	m_feature_weight = new float[nLandmark + nSamples * nTotalProperties];
	float landmarkWeight = (nLandmark > 0) ? (float)(nSamples * nTotalProperties) / (float)nLandmark: 0;	// based on the number ratio (balance between landmark and property)
	float totalWeight = weightLoc;
	for (int n = 0; n < m_nProperties; n++) totalWeight += weight[n];
	landmarkWeight *= totalWeight;
	if (landmarkWeight == 0) landmarkWeight = 1;
	cout << "Total properties: " << nTotalProperties << endl;
	cout << "Sampling points: " << m_nSamples << " " << nSamples << endl;

	// assign the weighting factors
	for (int i = 0; i < nLandmark; i++) m_feature_weight[i] = landmarkWeight;
	for (int n = 0; n < m_nProperties; n++)
		for (int i = 0; i < nSamples; i++)
			m_feature_weight[nLandmark + nSamples * n + i] = weight[n];
	// weight for location information
	for (int n = 0; n < m_nSurfaceProperties; n++)
		for (int i = 0; i < nSamples; i++)
			m_feature_weight[nLandmark + nSamples * (m_nProperties + n) + i] = weightLoc;
	
	if (nLandmark > 0) cout << "Landmark weight: " << landmarkWeight << endl;
	if (m_nProperties > 0)
	{
		cout << "Property weight: ";
		for (int i = 0; i < m_nProperties; i++) cout << weight[i] << " ";
		cout << endl;
	}
	if (weightLoc > 0) cout << "Location weight: " << weightLoc << endl;
	
	cout << "Initialization of work space\n";
	m_cov = new float[m_nSubj * m_nSubj];	// convariance matrix defined in the duel space with dimensions: nSubj x nSubj
#ifdef _USE_CUDA_BLAS
	status = cudaMallocHost((void**)&m_feature, m_nSubj * (nLandmark + nSamples * nTotalProperties) * sizeof(float));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
#else
	m_feature = new float[m_nSubj * (nLandmark + nSamples * nTotalProperties)];	// the entire feature vector map for optimization
#endif

	// AABB tree cache for each subject: this stores the closest face of the sampling point to the corresponding face on the input sphere model
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (nTotalProperties > 0)
		{
#ifdef _USE_CUDA_BLAS
			status = cudaMallocHost((void**)&m_spharm[subj].tree_cache, nSamples * sizeof(int));
			if (status != cudaSuccess)
			{
				cout << "Fatal error: allocating pinned host memory" << endl;
				exit(1);
			}
#else
			m_spharm[subj].tree_cache = new int[nSamples];
#endif
			for (int i = 0; i < nSamples; i++)
				m_spharm[subj].tree_cache[i] = -1;	// initially, set to -1 (invalid index)
			int nVertex = m_spharm[subj].sphere->nVertex();
		}
		else
		{
			m_spharm[subj].tree_cache = NULL;
		}
	}

	// Find the maximum buffer size for gradients
	int nVertex = 0;
	int nFace = 0;
	m_nMaxVertex = m_nQuerySamples;
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		m_nMaxVertex = (m_nMaxVertex < m_spharm[subj].sphere->nVertex()) ? m_spharm[subj].sphere->nVertex(): m_nMaxVertex;
		nVertex = (nVertex < m_spharm[subj].sphere->nVertex()) ? m_spharm[subj].sphere->nVertex(): nVertex;
		nFace = (nFace < m_spharm[subj].sphere->nFace()) ? m_spharm[subj].sphere->nFace(): nFace;
	}

#ifdef _USE_CUDA_BLAS
	m_nCThreads = nCThreads;
	if (m_nCThreads == 0)
	{
		size_t free, total;
		Gradient *gsize = new Gradient(nVertex, nFace, m_nQuerySamples, m_degree);
		cudaMemGetInfo(&free, &total);
		m_nCThreads = (total / (total - free));
		delete gsize;
	}
	m_nCThreads = (m_nCThreads < m_nDeformableSubj) ? m_nCThreads: m_nDeformableSubj;
	cout << "# of CUDA streams: " << m_nCThreads << endl;
	m_cuda_grad = new Gradient*[m_nCThreads];
	for (int i = 0; i < m_nCThreads; i++)
	{
		m_cuda_grad[i] = new Gradient(nVertex, nFace, m_nQuerySamples, m_degree, m_ico_Y, ico_face, ico_neighbor, ico_nNeighbor);
	}
#else
	m_nCThreads = 1;
	m_gradient_raw = new double[m_nMaxVertex * 3 * csize * 2];	// work space for jacobian
	m_gradient_diag = new double[m_nMaxVertex];	// work space for jacobian
	m_gradient_work = new double[csize * 3 + m_nMaxVertex];
	for (int i = 0; i < m_nMaxVertex; i++) m_gradient_work[csize * 3 + i] = 1;
#endif
	if (m_resampling)
	{
		delete [] ico_vertex;
		delete [] ico_face;
#ifdef _USE_CUDA_BLAS
		delete [] ico_neighbor;
		delete [] ico_nNeighbor;
#endif
	}

	cout << "Feature vector creation\n";

	// feature update
	#pragma omp parallel for
	for (int subj = 0; subj < m_nSubj; subj++) updateDeformation(subj, true);
	
	if (coeff != NULL)
	{
		#pragma omp parallel for
		for (int subj = 0; subj < m_nSubj; subj++) initArea(subj);
	}
#ifdef _USE_CUDA_BLAS
	status = cudaMallocHost((void**)&m_mean, nSamples * nTotalProperties * sizeof(double));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
	status = cudaMallocHost((void**)&m_variance, nSamples * nTotalProperties * sizeof(double));
	if (status != cudaSuccess)
	{
		cout << "Fatal error: allocating pinned host memory" << endl;
		exit(1);
	}
#else
	m_mean = new double[nSamples * nTotalProperties];
	m_variance = new double[nSamples * nTotalProperties];
#endif

	bool guess = m_guess;
	m_guess = false;
	// set all the tree needs to be updated
	memset(m_updated, 0, sizeof(bool) * m_nSubj);
	if (nLandmark > 0) updateLandmark(); // update landmark
	if (nSamples > 0)	// update properties
	{
		updateDisplacement();
		updateProperties();
		if (m_pairwise) initPairwise(tmpVariance);
	}
	m_guess = guess;

	// inital coefficients for the previous step
	memcpy(m_coeff_prev_step, m_coeff, sizeof(double) * m_csize * 3);
	
	cout << "Initialization done!" << endl;
}

void HSD::initSphericalHarmonics(int subj, const char **coeff, double *Y)
{
	// spherical harmonics information
	int n = (m_degree + 1) * (m_degree + 1);	// total number of coefficients (this must be the same across all the subjects at the end of this program)
	// new memory allocation for coefficients
	
	m_spharm[subj].coeff = &m_coeff[subj * n * 3];	// latitudes
	m_spharm[subj].coeff_prev_step = &m_coeff_prev_step[subj * n * 3];	// latitudes
	m_spharm[subj].gradient = &m_gradient[subj * n * 3];	// latitudes

	// pole configuration
	m_spharm[subj].pole = &m_pole[subj * 2];
	m_spharm[subj].tan1 = &m_Tbasis[subj * 3 * 2];
	m_spharm[subj].tan2 = &m_Tbasis[subj * 3 * 2 + 3];

	if (coeff != NULL && !m_spharm[subj].fixed)	// previous spherical harmonics information
	{
		FILE *fp = fopen(coeff[subj],"r");
		if (fscanf(fp, "%d", &m_spharm[subj].degree) == -1)	// previous deformation field degree
		{
			cout << "Fatal error: something goes wrong during I/O processing" << endl;
			fclose(fp);
			exit(1);
		}
		if (fscanf(fp, "%f %f", &m_spharm[subj].pole[0], &m_spharm[subj].pole[1]) == -1)	// optimal pole information
		{
			cout << "Fatal error: something goes wrong during I/O processing" << endl;
			fclose(fp);
			exit(1);
		}

		if (m_spharm[subj].degree > m_degree) m_spharm[subj].degree = m_degree;	// if the previous degree is larger than desired one, just crop it.

		// load previous coefficient information
		for (int i = 0; i < (m_spharm[subj].degree + 1) * (m_spharm[subj].degree + 1); i++)
		{
			if (fscanf(fp, "%lf %lf %lf", &m_spharm[subj].coeff[i], &m_spharm[subj].coeff[n + i], &m_spharm[subj].coeff[2 * n + i]) == -1)
			{
				cout << "Fatal error: something goes wrong during I/O processing" << endl;
				fclose(fp);
				exit(1);
			}
		}
		fclose(fp);
	}
	else	// no spherical harmonic information is provided
	{
		float p[3] = {0, 0, 1};
		Coordinate::cart2sph(p, &m_spharm[subj].pole[0], &m_spharm[subj].pole[1]);
	}
	m_spharm[subj].degree = m_degree;
	
	int nVertex = m_spharm[subj].sphere->nVertex();
	if (Y == NULL || m_spharm[subj].fixed)
	{
#ifdef _USE_CUDA_BLAS
		cudaError_t status = cudaMallocHost((void**)&m_spharm[subj].Y, (m_degree + 1) * (m_degree + 1) * nVertex * sizeof(double));
		if (status != cudaSuccess)
		{
			cout << "Fatal error: allocating pinned host memory" << endl;
			exit(1);
		}
#else
		m_spharm[subj].Y = new double[(m_degree + 1) * (m_degree + 1) * nVertex];
#endif
	}
	else m_spharm[subj].Y = Y;
	// build spherical harmonic basis functions for each vertex
	for (int i = 0; i < nVertex; i++)
	{
		// vertex information
		point *p = new point();	// new spherical information allocation
		Vertex *v = (Vertex *)m_spharm[subj].sphere->vertex(i);	// vertex information on the sphere
		const float *v0 = v->fv();
		p->Y = &m_spharm[subj].Y[(m_degree + 1) * (m_degree + 1) * i];
		p->p[0] = v0[0]; p->p[1] = v0[1]; p->p[2] = v0[2];
		p->id = i;
		p->subject = subj;
		//SphericalHarmonics::basis(m_degree, p->p, p->Y);
		double vd[3] = {v0[0], v0[1], v0[2]};
		if (Y == NULL || m_spharm[subj].fixed) SphericalHarmonics::basis(m_degree, vd, p->Y);
		m_spharm[subj].vertex.push_back(p);
	}
}

void HSD::initTangentPlane(int subj)
{
	float fmean[3] = {0, 0, 1};
	
	int nLandmark = m_spharm[subj].landmark.size();
	if (nLandmark > 0)
	{
		// frechet mean of landmarks
		Coordinate::sph2cart(m_spharm[subj].pole[0], m_spharm[subj].pole[1], fmean);
		Vector optMean(fmean);
		for (int landmark = 0; landmark < m_spharm[subj].landmark.size(); landmark++)
			optMean += Vector(m_spharm[subj].landmark[landmark]->p);
		optMean.unit();
		Coordinate::cart2sph(optMean.fv(), &m_spharm[subj].pole[0], &m_spharm[subj].pole[1]);
		cost_function_subj costFunc(this, subj);
		double delta[2] = {0, 0};
		double xl[2] = {-PI, -PI};
		double xu[2] = {PI, PI};
		//min_newuoa(2, delta, costFunc, PI, 1e-6, m_maxIter);
		min_bobyqa(2, delta, costFunc, xl, xu, 1e-1, 1e-6, m_maxIter);
		//cout << delta[0] << " " << delta[1] << endl;
		m_spharm[subj].pole[0] += (float)delta[0];
		m_spharm[subj].pole[1] += (float)delta[1];
	}
	
	// setup tangent plane
	Coordinate::sph2cart(m_spharm[subj].pole[0], m_spharm[subj].pole[1], fmean);
	Coordinate::sph2cart(m_spharm[subj].pole[0] + 0.1, m_spharm[subj].pole[1] + 0.1, m_spharm[subj].tan1);	// arbitrary tangent basis
	Coordinate::proj2plane(fmean[0], fmean[1], fmean[2], -1, m_spharm[subj].tan1, m_spharm[subj].tan1);
	const float *tan1 = Vector(fmean, m_spharm[subj].tan1).unit().fv();
	m_spharm[subj].tan1[0] = tan1[0]; m_spharm[subj].tan1[1] = tan1[1]; m_spharm[subj].tan1[2] = tan1[2];
	const float *tan2 = (Vector(fmean).cross(Vector(m_spharm[subj].tan1))).unit().fv();
	m_spharm[subj].tan2[0] = tan2[0]; m_spharm[subj].tan2[1] = tan2[1]; m_spharm[subj].tan2[2] = tan2[2];
}

string HSD::initProperties(int subj, const char **property, int nLines, AABB_Sphere *tree, Mesh *sphere, int nHeaderLines)
{
	string log;
	int nVertex = m_spharm[subj].sphere->nVertex();	// this is the same as the number of properties
	int nFace = m_spharm[subj].sphere->nFace();
	bool resampling = m_resampling;
	float *property_raw;

	if (m_nProperties + m_nSurfaceProperties > 0)
	{
		m_spharm[subj].meanProperty = new float[m_nProperties + m_nSurfaceProperties];
		m_spharm[subj].medianProperty = new float[m_nProperties + m_nSurfaceProperties];
		m_spharm[subj].maxProperty = new float[m_nProperties + m_nSurfaceProperties];
		m_spharm[subj].minProperty = new float[m_nProperties + m_nSurfaceProperties];
		m_spharm[subj].sdevProperty = new float[m_nProperties + m_nSurfaceProperties];
#ifdef _USE_CUDA_BLAS
		cudaError_t status = cudaMallocHost((void**)&m_spharm[subj].property, (m_nProperties + m_nSurfaceProperties) * nVertex * sizeof(float));
		if (status != cudaSuccess)
		{
			cout << "Fatal error: allocating pinned host memory" << endl;
			exit(1);
		}
#else
		m_spharm[subj].property = new float[(m_nProperties + m_nSurfaceProperties) * nVertex];
#endif
		if (resampling)
			property_raw = new float[(m_nProperties + m_nSurfaceProperties) * nLines];
		else
			property_raw = m_spharm[subj].property;
		}
	else
	{
		m_spharm[subj].meanProperty = NULL;
		m_spharm[subj].medianProperty = NULL;
		m_spharm[subj].maxProperty = NULL;
		m_spharm[subj].minProperty = NULL;
		m_spharm[subj].sdevProperty = NULL;
		m_spharm[subj].property = NULL;
	}
	for (int i = 0; i < m_nProperties; i++)	// property information
	{
		int index = subj * m_nProperties + i;
		log += "\t" + string(property[index]) + "\n";
		FILE *fp = fopen(property[index], "r");
		
		// remove header lines
		char line[1024];
		for (int j = 0; j < nHeaderLines; j++)
		{
			if (fgets(line, sizeof(line), fp) == NULL)
			{
				cout << "Fatal error: something goes wrong during I/O processing" << endl;
				fclose(fp);
				exit(1);
			}
		}
		// load property information
		for (int j = 0; j < nLines; j++)
		{
			if (fscanf(fp, "%f", &property_raw[nLines * i + j]) == -1)
			{
				cout << "Fatal error: something goes wrong during I/O processing" << endl;
				fclose(fp);
				exit(1);
			}
		}
		fclose(fp);
	}
	for (int i = 0; i < m_nSurfaceProperties; i++)	// x, y, z dimensions
	{
		for (int j = 0; j < nLines; j++)
		{
			Vertex *v = (Vertex *)m_spharm[subj].surf->vertex(j);
			const float *v0 = v->fv();
			property_raw[nLines * (m_nProperties + i) + j] = v0[i];
		}
	}
	if (resampling)
	{
		for (int j = 0; j < nVertex; j++)
		{
			float coeff[3];
			Vertex *v = (Vertex *)m_spharm[subj].sphere->vertex(j);
			const float *v0 = v->fv();
			int fid = tree->closestFace((float *)v0, coeff);
			for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
				m_spharm[subj].property[nVertex * i + j] = propertyInterpolation(&property_raw[nLines * i], fid, coeff, sphere);
		}
		delete [] property_raw;
	}

	// find the best statistics across subjects
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		log += "--Property " + to_string(i) + "\n";
		m_spharm[subj].meanProperty[i] = Statistics::mean(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].medianProperty[i] = Statistics::median(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].maxProperty[i] = Statistics::max(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].minProperty[i] = Statistics::min(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].sdevProperty[i] = sqrt(Statistics::var(&m_spharm[subj].property[nVertex * i], nVertex));
		log += "---Min/Max: " + to_string(m_spharm[subj].minProperty[i]) + ", " + to_string(m_spharm[subj].maxProperty[i]) + "\n";
		log += "---Mean/Stdev: " + to_string(m_spharm[subj].meanProperty[i]) + ", " + to_string(m_spharm[subj].sdevProperty[i]) + "\n";
		log += "---Median: " + to_string(m_spharm[subj].medianProperty[i]) + "\n";
	}
	
	// normalization - median z-score
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		for (int j = 0; j < nVertex; j++)
		{
			m_spharm[subj].property[nVertex * i + j] -= m_spharm[subj].medianProperty[i];
		}
	}
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		m_spharm[subj].sdevProperty[i] = sqrt(Statistics::var(&m_spharm[subj].property[nVertex * i], nVertex));
	}
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		for (int j = 0; j < nVertex; j++)
		{
			m_spharm[subj].property[nVertex * i + j] /= m_spharm[subj].sdevProperty[i];
		}
	}
	// over 3 sigma -> this bounds -4s to 4s
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		for (int j = 0; j < nVertex; j++)
		{
			if (m_spharm[subj].property[nVertex * i + j] < -3)
				m_spharm[subj].property[nVertex * i + j] = -3 - (1 - exp(3 + m_spharm[subj].property[nVertex * i + j]));
			if (m_spharm[subj].property[nVertex * i + j] > 3)
				m_spharm[subj].property[nVertex * i + j] = 3 + (1 - exp(3 - m_spharm[subj].property[nVertex * i + j]));
		}
	}
	// adjusted stdev
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		m_spharm[subj].sdevProperty[i] = sqrt(Statistics::var(&m_spharm[subj].property[nVertex * i], nVertex));
	}
	// refinement
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		for (int j = 0; j < nVertex; j++)
		{
			m_spharm[subj].property[nVertex * i + j] /= m_spharm[subj].sdevProperty[i];
		}
	}
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		for (int j = 0; j < nVertex; j++)
		{
			if (m_spharm[subj].property[nVertex * i + j] < -3)
				m_spharm[subj].property[nVertex * i + j] = -3 - (1 - exp(3 + m_spharm[subj].property[nVertex * i + j]));
			if (m_spharm[subj].property[nVertex * i + j] > 3)
				m_spharm[subj].property[nVertex * i + j] = 3 + (1 - exp(3 - m_spharm[subj].property[nVertex * i + j]));
		}
	}
	// adjusted stats
	for (int i = 0; i < m_nProperties + m_nSurfaceProperties; i++)
	{
		m_spharm[subj].meanProperty[i] = Statistics::mean(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].medianProperty[i] = Statistics::median(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].maxProperty[i] = Statistics::max(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].minProperty[i] = Statistics::min(&m_spharm[subj].property[nVertex * i], nVertex);
		m_spharm[subj].sdevProperty[i] = sqrt(Statistics::var(&m_spharm[subj].property[nVertex * i], nVertex));
		log += "---Adjusted Min/Max: " + to_string(m_spharm[subj].minProperty[i]) + ", " + to_string(m_spharm[subj].maxProperty[i]) + "\n";
		log += "---Adjusted Mean/Stdev: " + to_string(m_spharm[subj].meanProperty[i]) + ", " + to_string(m_spharm[subj].sdevProperty[i]) + "\n";
		log += "---Adjusted Median: " + to_string(m_spharm[subj].medianProperty[i]) + "\n";
	}

	return log;
}

string HSD::initTriangleFlipping(int subj)
{
	string log;
	int nFace = m_spharm[subj].sphere->nFace();
	if (m_spharm[subj].flip == NULL) m_spharm[subj].flip = new bool[nFace];
	int nFlips = 0;
	int nInitFlips = 0;

	double neg_area = 0;
	// check triangle flips
	for (int i = 0; i < nFace; i++)
	{
		const float *v1 = m_spharm[subj].sphere->vertex(m_spharm[subj].sphere->face(i)->list(0))->fv();
		const float *v2 = m_spharm[subj].sphere->vertex(m_spharm[subj].sphere->face(i)->list(1))->fv();
		const float *v3 = m_spharm[subj].sphere->vertex(m_spharm[subj].sphere->face(i)->list(2))->fv();

		Vector V1(v1), V2(v2), V3(v3);
		Vector V = (V1 + v2 + V3) / 3;

		Vector U = (V2 - V1).cross(V3 - V1);

		m_spharm[subj].flip[i] = false;
		if (V * U < 0)
		{
			double neg = U.norm();
			neg_area += neg;
			nInitFlips++;
			if (neg > 1e-4) m_spharm[subj].flip[i] = true;
		}
		if (m_spharm[subj].flip[i]) nFlips++;
	}
	m_spharm[subj].neg_area_threshold = neg_area + 1e-5;

	log += "--Negative area: " + to_string(neg_area) + "\n";
	log += "--Initial flips: " + to_string(nInitFlips) + "\n";
	log += "--Adjusted flips: " + to_string(nFlips) + "\n";

	return log;
}

string HSD::initArea(int subj)
{
	string log;
	int nVertex = m_spharm[subj].sphere->nVertex();
	int nFace = m_spharm[subj].sphere->nFace();
	if (m_spharm[subj].area0 == NULL) m_spharm[subj].area0 = new float[nFace];
	if (m_spharm[subj].area1 == NULL) m_spharm[subj].area1 = new float[nFace];
	
	// compute area
	float maxarea = 0;
	float minarea = 4 * PI;
	for (int i = 0; i < nFace; i++)
	{
		const int *id = m_spharm[subj].sphere->face(i)->list();
		const float *v1 = m_spharm[subj].sphere->vertex(id[0])->fv();
		const float *v2 = m_spharm[subj].sphere->vertex(id[1])->fv();
		const float *v3 = m_spharm[subj].sphere->vertex(id[2])->fv();

		Vector V1(v1), V2(v2), V3(v3);

		Vector U = (V2 - V1).cross(V3 - V1);
		float area = U.norm() / 2;

		m_spharm[subj].area0[i] = area;
		if (m_spharm[subj].area0[i] < 1e-6) m_spharm[subj].area0[i] = 1e-6;
		m_spharm[subj].area1[i] = m_spharm[subj].area0[i];
		
		if (minarea > area) minarea = area;
		if (maxarea < area) maxarea = area;
	}
	log += "--Min/Max: " + to_string(minarea) + ", " + to_string(maxarea) + "\n";

	return log;
}

string HSD::initLandmarks(int subj, const char **landmark)
{
	string log;
	log += "\t" + string(landmark[subj]) + "\n";
	FILE *fp = fopen(landmark[subj], "r");
	int i = 0;
	while (!feof(fp))
	{
		// indices for corresponding points
		int srcid;

		// landmark information
		int id;
		if (fscanf(fp, "%d", &id) == -1) break;
		const float *v = m_spharm[subj].sphere->vertex(id)->fv();
		
		double *Y = new double[(m_degree + 1) * (m_degree + 1)];
		point *p = new point();
		p->p[0] = v[0]; p->p[1] = v[1]; p->p[2] = v[2];
		//SphericalHarmonics::basis(m_degree, p->p, Y);
		double vd[3] = {v[0], v[1], v[2]};
		SphericalHarmonics::basis(m_degree, vd, Y);
		p->subject = subj;
		p->Y = Y;
		p->id = id;

		m_spharm[subj].landmark.push_back(p);
		
		i++;
	}
	fclose(fp);
	return log;
}

void HSD::initPairwise(const char *tmpVariance)
{
	int nSamples = m_nQuerySamples;
	if (tmpVariance != NULL)
	{
		int tmp = (m_spharm[0].fixed) ? 0: 1;
		int nVertex = m_spharm[tmp].sphere->nVertex();
		float *var = new float[nVertex * (m_nProperties + m_nSurfaceProperties)];
		FILE *fp = fopen(tmpVariance, "r");
		for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
		{
			for (int i = 0; i < nVertex; i++)
			{
				if (fscanf(fp, "%f", &var[nVertex * k + i]) == -1)	// previous deformation field degree
				{
					cout << "Fatal error: something goes wrong during I/O processing" << endl;
					fclose(fp);
					exit(1);
				}
			}
		}
		fclose(fp);
		// bary centric
		float coeff[3];
		for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
		{
			for (int i = 0; i < nSamples; i++)
			{
				int fid = m_spharm[tmp].tree_cache[i];
				Face *f = (Face *)m_spharm[tmp].sphere->face(fid);
				Vertex *a = (Vertex *)f->vertex(0);
				Vertex *b = (Vertex *)f->vertex(1);
				Vertex *c = (Vertex *)f->vertex(2);

				Vector N = Vector(a->fv(), b->fv()).cross(Vector(b->fv(), c->fv())).unit();
				Vector V_proj = Vector(&m_propertySamples[i * 3]) * ((Vector(a->fv()) * N) / (Vector(&m_propertySamples[i * 3]) * N));

				Coordinate::cart2bary((float *)a->fv(), (float *)b->fv(), (float *)c->fv(), (float *)V_proj.fv(), coeff, 1e-5);

				m_variance[m_nSamples * k + i] = propertyInterpolation(&var[nVertex * k], fid, coeff, m_spharm[tmp].sphere);
			}
			SurfaceUtil::smoothing(m_ico_mesh, 3, &m_variance[m_nSamples * k]);
			double m = 0;
			for (int i = 0; i < nSamples; i++)
				m += m_variance[m_nSamples * k + i] / nSamples;
			m_cost_var = m;
			m *= 0.1;
			if (m < 1e-5) m = 1;
			for (int i = 0; i < nSamples; i++)
			{
				if (m_variance[m_nSamples * k + i] < m)
					m_variance[m_nSamples * k + i] = m;
			}
		}
	
		delete [] var;
	}
	else
	{
		for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
			for (int i = 0; i < nSamples; i++)
				m_variance[m_nSamples * k + i] = 1.0;
	}
}

bool HSD::updateCoordinate(const float *v0, float *v1, const double *Y, const double *coeff, float degree, const float *pole, const float *tan1, const float *tan2)
{
	// spharm basis
	int n = (degree + 1) * (degree + 1);

	double delta[3] = {0, 0, 0};
	for (int i = 0; i < n; i++)
	{
		delta[0] += Y[i] * coeff[i];
		delta[1] += Y[i] * coeff[(m_degree + 1) * (m_degree + 1) + i];
		delta[2] += Y[i] * coeff[2 * (m_degree + 1) * (m_degree + 1) + i];
	}
	
	// cart coordinate
	float axis0[3], axis1[3];
	Coordinate::sph2cart(pole[0], pole[1], axis0);
	
	// exponential map (linear)
	const float *axis = (Vector(axis0) + Vector(tan1) * delta[1] + Vector(tan2) * delta[2]).unit().fv();
	axis1[0] = axis[0]; axis1[1] = axis[1]; axis1[2] = axis[2];

	// standard pole
	Vector P = axis0;
	Vector Q = axis1;
	float angle = (float)sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
	Vector A = P.cross(Q); A.unit();

	float rv[3];
	float rot[9];
	Coordinate::rotation(A.fv(), angle, rot);
	Coordinate::rotPoint(v0, rot, rv);

	if (delta[0] == 0)
	{
		memcpy(v1, rv, sizeof(float) * 3);
		if (delta[1] == 0 && delta[2] == 0)
			return true;
		else 
			return false;
	}
	
	// rotation
	Coordinate::rotation(Q.fv(), (float)delta[0], rot);
	Coordinate::rotPoint(rv, rot, v1);

	return true;
}

void HSD::updateDeformation(int subject, bool enforce)
{
	// note: the deformation happens only if the coefficients change; otherwise, nothing to do
	bool updated = m_updated[subject] && !enforce;
	
	// check if the coefficients change
	int n = (m_degree_inc + 1) * (m_degree_inc + 1);
	for (int i = 0; i < n && updated; i++)
		if (m_spharm[subject].coeff[i] != m_spharm[subject].coeff_prev_step[i] ||
			m_spharm[subject].coeff[(m_degree + 1) * (m_degree + 1) + i] != m_spharm[subject].coeff_prev_step[(m_degree + 1) * (m_degree + 1) + i] ||
			m_spharm[subject].coeff[2 * (m_degree + 1) * (m_degree + 1) + i] != m_spharm[subject].coeff_prev_step[2 * (m_degree + 1) * (m_degree + 1) + i])
			updated = false;
	
	// deform the sphere based on the current coefficients if necessary
	for (int i = 0; i < m_spharm[subject].vertex.size() && !updated; i++)
	{
		Vertex *v = (Vertex *)m_spharm[subject].sphere->vertex(i);
		float v1[3];
		const float *v0 = v->fv();
		updateCoordinate(m_spharm[subject].vertex[i]->p, v1, m_spharm[subject].vertex[i]->Y, (const double *)m_spharm[subject].coeff, m_degree, m_spharm[subject].pole, m_spharm[subject].tan1, m_spharm[subject].tan2); // update using the current incremental degree
		Vector V(v1); V.unit();
		v->setVertex(V.fv());
	}
	m_updated[subject] = updated;
	
	if (m_icosahedron < m_fine_res) updateArea(subject);
}

void HSD::updateLandmark(int subj_id)
{
	int nLandmark = m_spharm[0].landmark.size();

	for (int i = 0; i < nLandmark; i++)
	{
		for (int subj = 0; subj < m_nSubj; subj++)
		{
			if (subj_id != -1 && subj != subj_id) continue;
			int id = m_spharm[subj].landmark[i]->id;
			updateCoordinate(m_spharm[subj].landmark[i]->p, &m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3], m_spharm[subj].landmark[i]->Y, (const double *)m_spharm[subj].coeff, m_degree_inc, m_spharm[subj].pole, m_spharm[subj].tan1, m_spharm[subj].tan2);
		}
	}
}

void HSD::updateProperties(int subj_id)
{
	int nLandmark = m_spharm[0].landmark.size();
	int nSamples = m_nQuerySamples;	// # of sampling points for property map agreement
	
	const float err = 0;

	#pragma omp parallel for
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (subj_id != -1 && subj_id != subj) continue;
		if (!m_updated[subj])
		{
			m_updated[subj] = true;
			m_spharm[subj].tree->update();
		}
		else continue;	// don't compute again since tree is the same as the previous. The feature vector won't be changed
		for (int i = 0; i < nSamples; i++)
		{
			int fid = -1;
			float coeff[3];
			if (m_spharm[subj].tree_cache[i] != -1)	// if previous cache is available
			{
				Face *f = (Face *)m_spharm[subj].sphere->face(m_spharm[subj].tree_cache[i]);
				Vertex *a = (Vertex *)f->vertex(0);
				Vertex *b = (Vertex *)f->vertex(1);
				Vertex *c = (Vertex *)f->vertex(2);

				Vector N = Vector(a->fv(), b->fv()).cross(Vector(b->fv(), c->fv())).unit();
				Vector V_proj = Vector(&m_propertySamples[i * 3]) * ((Vector(a->fv()) * N) / (Vector(&m_propertySamples[i * 3]) * N));

				// bary centric
				Coordinate::cart2bary((float *)a->fv(), (float *)b->fv(), (float *)c->fv(), (float *)V_proj.fv(), coeff, 1e-5);

				fid = (coeff[0] >= err && coeff[1] >= err && coeff[2] >= err) ? m_spharm[subj].tree_cache[i]: -1;
			}
			if (fid == -1)	// if no closest face is found
			{
				fid = m_spharm[subj].tree->closestFace(&m_propertySamples[i * 3], coeff);
				if (fid == -1)	// something goes wrong
				{
					cout << "Fatal error: Property - no closest triangle found at ";
					cout << "[" << m_propertySamples[i * 3] << ", " << m_propertySamples[i * 3 + 1] << ", " << m_propertySamples[i * 3 + 2] << "]";
					cout << " in " << subj << endl;
					exit(1);
				}
			}

			int nVertex = m_spharm[subj].sphere->nVertex();
			for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
			{
				m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + nLandmark * 3 + m_nSamples * k + i] = propertyInterpolation(&m_spharm[subj].property[nVertex * k], fid, coeff, m_spharm[subj].sphere);
			}
			m_spharm[subj].tree_cache[i] = fid;
		}
	}
	if (!m_guess && (m_degree_inc == 0 || m_pairwise)) updatePropertyStats();
}

void HSD::updatePropertyStats(void)
{
	int nLandmark = m_spharm[0].landmark.size();
	int nSamples = m_nQuerySamples;	// # of sampling points for property map agreement

	double mean = 0;
	#pragma omp parallel for
	for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
	{
		for (int i = 0; i < nSamples; i++)
		{
			double m = 0;
			for (int subj = 0; subj < m_nSubj; subj++)
			{
				if ((m_nDeformableSubj == m_nSubj) || (m_nDeformableSubj < m_nSubj && m_spharm[subj].fixed))
					m += m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + nLandmark * 3 + m_nSamples * k + i];
			}
			// use the mean of the fixed subjects
			m = (m_nSubj == m_nDeformableSubj) ? m / m_nSubj: m / (m_nSubj - m_nDeformableSubj);
			m_mean[m_nSamples * k + i] = m;
		}
		if (m_pairwise) continue;
		if (m_degree_inc == 0 || m_icosahedron < m_fine_res)
		{
			for (int i = 0; i < nSamples; i++)
				m_variance[m_nSamples * k + i] = 1.0;
		}
		else
		{
			for (int i = 0; i < nSamples; i++)
			{
				double m = m_mean[m_nSamples * k + i];
				double sd = 0;
				for (int subj = 0; subj < m_nSubj; subj++)
				{
					double p = m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + nLandmark * 3 + m_nSamples * k + i];
					double w = m_feature_weight[nLandmark * 3 + m_nSamples * k + i];
					double pm = (p - m);
					//pm *= w;
					//pm /= (m_nProperties + m_nSurfaceProperties) * nSamples;
					sd += pm * pm / (m_nSubj - 1);
				}
				m_variance[m_nSamples * k + i] = sd;
			}
			//SurfaceUtil::smoothing(m_ico_mesh, 3, &m_mean[m_nSamples * k]);
			SurfaceUtil::smoothing(m_ico_mesh, 3, &m_variance[nSamples * k]);
			double m = 0;
			for (int i = 0; i < nSamples; i++)
				m += m_variance[m_nSamples * k + i] / nSamples;
			m_cost_var = m;
			m *= 0.1;
			if (m < 1e-5) m = 1;
			for (int i = 0; i < nSamples; i++)
			{
				if (m_variance[m_nSamples * k + i] < m)
					m_variance[m_nSamples * k + i] = m;
			}
		}
	}
}

void HSD::updateArea(int subj)
{
	int nFace = m_spharm[subj].sphere->nFace();
	if (m_spharm[subj].fixed) return;
	for (int i = 0; i < nFace; i++)
	{
		const int *id = m_spharm[subj].sphere->face(i)->list();
		const float *v1 = m_spharm[subj].sphere->vertex(id[0])->fv();
		const float *v2 = m_spharm[subj].sphere->vertex(id[1])->fv();
		const float *v3 = m_spharm[subj].sphere->vertex(id[2])->fv();

		Vector V1(v1), V2(v2), V3(v3);

		Vector U = (V2 - V1).cross(V3 - V1);

		float area = U.norm() / 2;
		if (area < 1e-6) area = 1e-6;

		m_spharm[subj].area1[i] = area;
	}
}

void HSD::updateDisplacement(int subj_id, int degree)
{
	#pragma omp parallel for
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (subj_id != -1 && subj_id != subj) continue;
		if (m_spharm[subj].fixed) continue;
		int nVertex = m_spharm[subj].sphere->nVertex();
		for (int i = 0; i < nVertex; i++)
		{
			float v[3];
			updateCoordinate(m_spharm[subj].vertex[i]->p, v, m_spharm[subj].vertex[i]->Y, (const double *)m_spharm[subj].coeff, degree, m_spharm[subj].pole, m_spharm[subj].tan1, m_spharm[subj].tan2);
			m_spharm[subj].vertex[i]->p0[0] = v[0];
			m_spharm[subj].vertex[i]->p0[1] = v[1];
			m_spharm[subj].vertex[i]->p0[2] = v[2];
		}
#ifdef _USE_CUDA_BLAS
		for (int i = 0; i < nVertex; i++) memcpy(&m_spharm[subj].vertex0[i * 3], m_spharm[subj].vertex[i]->p0, sizeof(float) * 3);
#endif
	}
}

double HSD::trace(int subj_id)
{
	int nLandmark = m_spharm[0].landmark.size();	// # of landmarks: we assume all the subject has the same number
	int nSamples = m_nQuerySamples;	// # of sampling points for property map agreement
	
	double E = 0;	// trace
	
	// update landmark
	if (nLandmark > 0)
	{
		updateLandmark(subj_id);
		m_cost_lm = varLandmarks(subj_id);
	}
	
	// update properties
	if (nSamples > 0)
	{
		updateProperties(subj_id);
		m_cost_prop = varProperties(subj_id);
	}

	m_cost_area = (m_degree_inc == 0 || m_icosahedron >= m_fine_res || m_lambda2 == 0) ? 0: varArea(subj_id);
	m_cost_disp = (m_degree_inc == 0 || m_icosahedron < m_fine_res || m_lambda2 == 0) ? 0: varDisplacement(subj_id);
	
	// the amount of allowable deformation
	if (m_icosahedron < m_fine_res)
		E = m_cost_lm + m_eta * m_cost_prop + m_lambda1 * m_cost_area;
	else
		E = m_cost_lm + m_eta * m_cost_prop + m_lambda2 * m_cost_disp;
	
	return E;
}

#ifdef _USE_SYSV
void HSD::linear(double *A, double *b, int dim, int *ipiv, double *work)
{
	int n = dim;
	int nrhs = 1;
	int lwork = -1;	// dimension of the work array
	int info;			// information (0 for successful exit)
	int lda = n;
	int ldb = n;
	char uplo = 'L'; // Lower triangle
	
	//dposv_(uplo, &n, &n, A, &n, b, &n, &info);
	//dgetrf_(&n, &n, A, &n, ipiv, &info);
	dsysv_(&uplo, &n, &nrhs, A, &lda, ipiv, b, &ldb, work, &lwork, &info);
	lwork = min((int)work[0], (m_degree + 1) * (m_degree + 1) * 3 * 64);
	dsysv_(&uplo, &n, &nrhs, A, &lda, ipiv, b, &ldb, work, &lwork, &info);
}
#else
void HSD::inverse(double *M, int dim, int *ipiv, double *work)
{
	int n = dim;
	int lwork = n * n;	// dimension of the work array
	int lda = n;			// lda: leading dimension
	int info;				// information (0 for successful exit)
	char uplo[] = "L"; // Lower triangle
	
	dgetrf_(&n, &n, M, &n, ipiv, &info);
	dgetri_(&n, M, &n, ipiv, work, &lwork, &info);
	/*dsytrf_( uplo, &n, M, &lda, &ipiv, &work, &lwork, &info );
	dsytri_( uplo, &n, M, &lda, &ipiv, &work, &info );*/
}
#endif

void HSD::ATB(double *A, int nr_rows_A, int nr_cols_A, double *B, int nr_cols_B, double *C)
{
	int m = nr_cols_A, n = nr_cols_B, k = nr_rows_A;
	int lda = m, ldb = n, ldc = m;
	double alpha = 1;
	double beta = 0;
	
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void HSD::ATDA(double *A, double *D, int nr_rows_A, int nr_cols_A, double *B)
{
	double *DA = &A[m_nMaxVertex * 3 * (m_degree + 1) * (m_degree + 1)];
	memset(DA, 0, sizeof(double) * nr_rows_A * nr_cols_A);

	for (int row = 0; row < nr_rows_A; row++)
		cblas_daxpy(nr_cols_A, D[row], &A[row * nr_cols_A], 1, &DA[row * nr_cols_A], 1);
	/*for (int row = 0; row < nr_rows_A; row++)
		for (int col = 0; col < nr_cols_A; col++)
			DA[row * nr_cols_A + col] = D[row] * A[row * nr_cols_A + col];*/

	ATB(A, nr_rows_A, nr_cols_A, DA, nr_cols_A, B);
}

float HSD::propertyInterpolation(float *refMap, int index, float *coeff, Mesh *mesh)
{
	float property = 0;

	if (index != -1)
	{
		Face *f = (Face *)mesh->face(index);
		Vertex *a = (Vertex *)f->vertex(0);
		Vertex *b = (Vertex *)f->vertex(1);
		Vertex *c = (Vertex *)f->vertex(2);
		property = refMap[a->id()] * coeff[0] + refMap[b->id()] * coeff[1] + refMap[c->id()] * coeff[2];
	}

	return property;
}

double HSD::propertyInterpolation(double *refMap, int index, float *coeff, Mesh *mesh)
{
	double property = 0;

	if (index != -1)
	{
		Face *f = (Face *)mesh->face(index);
		Vertex *a = (Vertex *)f->vertex(0);
		Vertex *b = (Vertex *)f->vertex(1);
		Vertex *c = (Vertex *)f->vertex(2);
		property = refMap[a->id()] * coeff[0] + refMap[b->id()] * coeff[1] + refMap[c->id()] * coeff[2];
	}

	return property;
}

double HSD::cost(int subj_id)
{
	const double tol = 1e-5;
	#pragma omp parallel for
	for (int i = 0; i < m_nSubj; i++)
	{
		m_spharm[i].isFlip = false;
		m_spharm[i].step_adjusted = false;
		if (subj_id != -1 && subj_id != i) continue;
		if (m_spharm[i].fixed) continue;
		if (m_spharm[i].step == 0) continue;
		updateDeformation(i);	// update defomation fields
		m_spharm[i].isFlip = (m_degree_inc == 0) ? false: testTriangleFlip(m_spharm[i].sphere, m_spharm[i].flip, m_spharm[i].neg_area_threshold);
		while (m_degree_inc != 0 && m_spharm[i].isFlip && m_spharm[i].step > tol)
		{
			memcpy(m_spharm[i].coeff, m_spharm[i].coeff_prev_step, sizeof(double) * (m_degree + 1) * (m_degree + 1) * 3);
			if (!m_pairwise)
			{
				if (m_spharm[i].isFlip) m_spharm[i].step *= 0.5;
				if (m_spharm[i].step <= tol)
				{
					m_spharm[i].step = 0;
				}
				else
				{
					m_spharm[i].step_adjusted = true;
					for (int j = 0; j < (m_degree_inc + 1) * (m_degree_inc + 1); j++)
					{
						m_spharm[i].coeff[j] -= m_spharm[i].step * m_spharm[i].gradient[j];
						m_spharm[i].coeff[(m_degree + 1) * (m_degree + 1) + j] -= m_spharm[i].step * m_spharm[i].gradient[(m_degree + 1) * (m_degree + 1) + j];
						m_spharm[i].coeff[2 * (m_degree + 1) * (m_degree + 1) + j] -= m_spharm[i].step * m_spharm[i].gradient[2 * (m_degree + 1) * (m_degree + 1) + j];
					}
				}
			}
			updateDeformation(i, true);	// force update defomation fields
			if (m_pairwise) break;
			else m_spharm[i].isFlip = testTriangleFlip(m_spharm[i].sphere, m_spharm[i].flip, m_spharm[i].neg_area_threshold);	// make sure no flips
		}
	}
	bool failed = true;
	for (int i = 0; i < m_nSubj; i++)
	{
		if (m_spharm[i].fixed) continue;
		failed &= m_spharm[i].isFlip;
	}
	if (failed) return m_mincost;
	double cost = trace(subj_id);

	return cost;
}

double HSD::varLandmarks(int subj_id)
{
	double cost = 0;
	int nLandmark = m_spharm[0].landmark.size();

	for (int i = 0; i < nLandmark; i++)
	{
		double m[3] = {0, 0, 0};	// mean
		for (int subj = 0; subj < m_nSubj; subj++)
		{
			if (subj_id != -1 && subj_id != subj) continue;
			int id = m_spharm[subj].landmark[i]->id;
			updateCoordinate(m_spharm[subj].landmark[i]->p, &m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3], m_spharm[subj].landmark[i]->Y, (const double *)m_spharm[subj].coeff, m_degree_inc, m_spharm[subj].pole, m_spharm[subj].tan1, m_spharm[subj].tan2);
			
			// mean locations
			if ((m_nDeformableSubj == m_nSubj) || (m_nDeformableSubj < m_nSubj && m_spharm[subj].fixed))
				for (int k = 0; k < 3; k++) m[k] += m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3 + k];
		}
		// forcing the mean to be on the sphere
		double norm = sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
		for (int k = 0; k < 3; k++) m[k] /= norm;
		
		// projection
		/*for (int subj = 0; subj < m_nSubj; subj++)
		{
			float newp[3];
			Coordinate::proj2plane(m[0], m[1], m[2], -1, &m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3], newp);
			memcpy(&m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3], newp, sizeof(float) * 3);
		}*/
		
		double sd = 0;
		for (int subj = 0; subj < m_nSubj; subj++)
		{
			if (subj_id != -1 && subj_id != subj) continue;
			double inner = 0;
			float *fpoint = &m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3];
			double w = m_feature_weight[i * 3];
			
			for (int k = 0; k < 3; k++)
			{
				inner += m[k] * fpoint[k];
			}
			if (inner > 1) inner = 1;
			else if (inner < -1) inner = -1;
			double arclen = acos(inner);

			//arclen *= w;
			//arclen /= nLandmark;
			sd += arclen * arclen / nLandmark;
		}
		cost += sd;
	}
	if (subj_id == -1) cost /= (m_nSubj - 1);

	return cost;
}

double HSD::varProperties(int subj_id)
{
	double cost = 0;
	int nLandmark = m_spharm[0].landmark.size();
	int nSamples = m_nQuerySamples;	// # of sampling points for property map agreement

	for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
	{
		double sd = 0;
		for (int i = 0; i < nSamples; i++)
		{
			double m = m_mean[m_nSamples * k + i];
			for (int subj = 0; subj < m_nSubj; subj++)
			{
				if (subj_id != -1 && subj_id != subj) continue;
				if (m_spharm[subj].fixed) continue;
				double p = m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + nLandmark * 3 + m_nSamples * k + i];
				double w = m_feature_weight[nLandmark * 3 + m_nSamples * k + i];
				//sd += (p - m) * (p - m) * w * w;
				double pm = (p - m);
				//pm *= w;
				double var = (m_pairwise && m_degree_inc == 0) ? 1: m_variance[m_nSamples * k + i];
				sd += pm * pm / ((m_nProperties + m_nSurfaceProperties) * nSamples) / var;
			}
		}
		cost += sd;
	}
	cost /= m_nSubj;

	return cost;
}

double HSD::varArea(int subj_id)
{
	double cost = 0;
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (subj_id != -1 && subj_id != subj) continue;
		if (m_spharm[subj].fixed) continue;
		int nFace = m_spharm[subj].sphere->nFace();
		for (int i = 0; i < nFace; i++)
		{
			double ratio = abs(log(m_spharm[subj].area1[i] / m_spharm[subj].area0[i]));
			cost += ratio / nFace;
		}
	}
	if (subj_id == -1) cost /= m_nSubj;
	return cost;
}

double HSD::varDisplacement(int subj_id)
{
	double cost = 0;
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (subj_id != -1 && subj_id != subj) continue;
		if (m_spharm[subj].fixed) continue;
		int nVertex = m_spharm[subj].sphere->nVertex();
		for (int i = 0; i < nVertex; i++)
		{
			Vertex *vert = (Vertex *)m_spharm[subj].sphere->vertex(i);
			const float *v = vert->fv();
			float inner = Vector(m_spharm[subj].vertex[i]->p0) * Vector(v);
			if (inner > 1) inner = 1;
			if (inner < -1) inner = -1;
			double degree = acos(inner);
			cost += degree * degree / nVertex;
			//cost += degree / nVertex;
			//cost += log(degree + 1e-5) / nVertex;
			//cost += sqrt(degree) / nVertex;
		}
	}
	if (subj_id == -1) cost /= m_nSubj;
	return cost;
}

bool HSD::testTriangleFlip(Mesh *mesh, const bool *flip, float neg_area_threshold)
{
	double area = 0;
	for (int i = 0; i < mesh->nFace(); i++)
	{
		const float *v1 = mesh->vertex(mesh->face(i)->list(0))->fv();
		const float *v2 = mesh->vertex(mesh->face(i)->list(1))->fv();
		const float *v3 = mesh->vertex(mesh->face(i)->list(2))->fv();

		Vector V1(v1), V2(v2), V3(v3);
		Vector V = (V1 + v2 + V3) / 3;

		Vector U = (V2 - V1).cross(V3 - V1);

		if ((V * U < 0 && !flip[i]) || (V * U > 0 && flip[i]))
			area += U.norm();
		if (area > 	neg_area_threshold) return true;
	}
	return false;
}

void HSD::minGradientDescent(int deg_beg, int deg_end, int subj_id)
{
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int nSamples = m_nQuerySamples;
	int nLandmark = m_spharm[0].landmark.size();
	nSuccessIter = 0;
    
	double t = 1;
	
	m_mincost = cost(subj_id);
	cout << "[" << nIter++ << "] " << m_mincost << endl;
	bool success = true;
	double norm = 1;
	const double tol = 1e-5;
	bool advance = true;
	
	for (int subj = 0; subj < m_nSubj; subj++) m_spharm[subj].step = 1;
	
	for (; t > tol; nIter++)
	{
		if (nIter > m_maxIter && m_icosahedron >= m_fine_res && deg_beg == 0 && deg_end == m_degree) break;
		if (success)
		{
			//updateGradient(deg_beg, deg_end, 0.001 * t / (nIter + 1), subj_id);
			updateGradient(deg_beg, deg_end, 0.001 * t / (nSuccessIter + 1), subj_id);
			/*norm = 0;
			for (int i = 0; i < m_csize * 3; i++)
				norm += m_gradient[i] * m_gradient[i];
			norm = sqrt(norm);*/
		}
		#pragma omp parallel for
		for (int i = 0; i < m_nSubj; i++)
		{
			if (m_spharm[i].step == 0) continue;
			for (int j = 0; j < (m_degree_inc + 1) * (m_degree_inc + 1); j++)
			{
				m_spharm[i].coeff[j] -= m_spharm[i].step * m_spharm[i].gradient[j];
				m_spharm[i].coeff[(m_degree + 1) * (m_degree + 1) + j] -= m_spharm[i].step * m_spharm[i].gradient[(m_degree + 1) * (m_degree + 1) + j];
				m_spharm[i].coeff[2 * (m_degree + 1) * (m_degree + 1) + j] -= m_spharm[i].step * m_spharm[i].gradient[2 * (m_degree + 1) * (m_degree + 1) + j];
			}
		}
		double currentCost = cost(subj_id);
		if (currentCost < m_mincost)
		{
			// write the current optimal solutions
			if (m_realtime_coeff)
			{
				for (int subj = 0; subj < m_nSubj; subj++)
				{
					if (!m_spharm[subj].fixed)
					{
						//if (m_output != NULL) saveSphere(m_output[subj], subj);
						if (m_outputcoeff != NULL) saveCoeff(m_outputcoeff[subj], subj);
					}
				}
			}
			memcpy(m_coeff_prev_step, m_coeff, sizeof(double) * m_csize * 3);
			
			if (advance)
			{
				t *= 2;
				//t *= 0.9;
				for (int subj = 0; subj < m_nSubj; subj++) m_spharm[subj].step *= 2;
			}
			success = true;
			
			//currentCost = cost(subj_id);
			double diff = (m_mincost - currentCost);
			if (m_icosahedron < m_fine_res)
				cout << "[" << nIter << "] " << currentCost << ": " << m_cost_prop << " " << m_cost_area;
			else
				cout << "[" << nIter << "] " << currentCost << ": " << m_cost_prop << " " << m_cost_disp;
			for (int i = 0; i < m_nSubj; i++)
				if (m_spharm[i].isFlip) cout << " " << i;
			cout << endl;
			fflush(stdout);

			if (diff < tol)
			{
				m_mincost = currentCost;
				break;
			}
			m_mincost = currentCost;
			nSuccessIter++;
		}
		else
		{
			memcpy(m_coeff, m_coeff_prev_step, sizeof(double) * m_csize * 3);
			#pragma omp parallel for
			for (int i = 0; i < m_nSubj; i++)
			{
				if (subj_id != -1 && subj_id != i) continue;
				updateDeformation(i, true);	// update defomation fields
			}
			if (nLandmark > 0) updateLandmark(subj_id);
			// update properties
			if (nSamples > 0) updateProperties(subj_id);
			t *= 0.5;
			for (int subj = 0; subj < m_nSubj; subj++)
			{
				if (!m_spharm[subj].step_adjusted) m_spharm[subj].step *= 0.5;
				if (m_spharm[subj].step <= tol)
					m_spharm[subj].step = 0;
				m_spharm[subj].step_adjusted = false;
			}
			//t /= 0.9;
			success = false;
			advance = false;
		}
	}
	memcpy(m_coeff, m_coeff_prev_step, sizeof(double) * m_csize * 3);
}

void HSD::updateGradient(int deg_beg, int deg_end, double lambda, int subj_id)
{
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int nLandmark = m_spharm[0].landmark.size();	// # of landmarks: we assume all the subject have the same number
	int nSamples = m_nQuerySamples;	// # of sampling points for property map agreement
	int csize = (m_degree + 1) * (m_degree + 1);

	memset(m_gradient, 0, sizeof(double) * m_csize * 3);

	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (m_spharm[subj].fixed) continue;
		if (m_spharm[subj].step == 0) continue;
		memset(&m_Hessian[csize * 3 * csize * 3 * subj], 0, sizeof(double) * size * size * 3 * 3);
		// update landmark
		if (nLandmark > 0) updateGradientLandmark(deg_beg, deg_end, subj);
#ifdef _USE_CUDA_BLAS
		for (int i = 0; i < m_spharm[subj].sphere->nVertex(); i++) memcpy(&m_spharm[subj].vertex1[i * 3], m_spharm[subj].sphere->vertex(i)->fv(), sizeof(float) * 3);
		int sid = subj % m_nCThreads;
		int ssid = subj / m_nCThreads;
		if (size == csize) ssid = 0;
#endif

		// update properties
		if (nSamples > 0)
		{
#ifdef _USE_CUDA_BLAS
			updateGradientProperties_cuda(deg_beg, deg_end, subj, sid, ssid);
#else
			updateGradientProperties(deg_beg, deg_end, subj);
#endif
		}

		if (m_degree_inc > 0 && m_icosahedron >= m_fine_res)
		{
#ifdef _USE_CUDA_BLAS
			updateGradientDisplacement_cuda(deg_beg, deg_end, subj, sid, ssid);
#else
			updateGradientDisplacement(deg_beg, deg_end, subj);
#endif
		}
	}
#ifdef _USE_CUDA_BLAS
	cudaDeviceSynchronize();
#endif
	if (m_icosahedron >= m_fine_res)
	{
		#pragma omp parallel for
		for (int subj = 0; subj < m_nSubj; subj++)
		{
			if (m_spharm[subj].fixed) continue;
			if (m_spharm[subj].step == 0) continue;
			updateNewGradient(deg_beg, deg_end, lambda, subj);
		}
	}
}

void HSD::updateHessian(int deg_beg, int deg_end, int nSamples, int subj)
{
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int csize = (m_degree + 1) * (m_degree + 1);

	double *M = m_Hessian_work;
#ifdef _USE_CUDA_BLAS
	Gradient::ATDA(m_gradient_raw, m_gradient_diag, nSamples, size * 3, M);
#else
	ATDA(m_gradient_raw, m_gradient_diag, nSamples, size * 3, M);
#endif
	for (int i = 0; i < size * 3 * size * 3; i++) m_Hessian[csize * 3 * csize * 3 * subj + i] += M[i];
}

void HSD::updateNewGradient(int deg_beg, int deg_end, double lambda, int subj)
{
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int csize = (m_degree + 1) * (m_degree + 1);

#ifdef _USE_SYSV
	double *G = &m_gradient_new[csize * 3 * subj];
#else
	double *G = &m_gradient_new[csize * 3 * 2 * subj];
#endif

	//double alpha = (m_icosahedron >= m_fine_res && deg_beg == 0 && deg_end == m_degree) ? 0.001 / exp(nIter): 0;
	double alpha = (m_icosahedron >= m_fine_res && deg_beg == 0 && deg_end == m_degree) ? 0.001 / exp(nSuccessIter): 0;
	double *M = &m_Hessian[csize * 3 * csize * 3 * subj];
	for (int i = 0; i < size * 3; i++)
		if (alpha != 0) M[size * 3 * i + i] += M[size * 3 * i + i] * lambda + alpha;
		else M[size * 3 * i + i] *= (1 + lambda);

	memcpy(G, &m_spharm[subj].gradient[n0], sizeof(double) * size);
	memcpy(&G[size], &m_spharm[subj].gradient[(m_degree + 1) * (m_degree + 1) + n0], sizeof(double) * size);
	memcpy(&G[size * 2], &m_spharm[subj].gradient[2 * (m_degree + 1) * (m_degree + 1) + n0], sizeof(double) * size);

#ifdef _USE_SYSV
	// Ax = b
	linear(M, G, size * 3, &m_ipiv[(csize * 3) * subj], &m_work[64 * csize * 3 * subj]);
	memcpy(&m_spharm[subj].gradient[n0], &G[size * 0], sizeof(double) * size);
	memcpy(&m_spharm[subj].gradient[(m_degree + 1) * (m_degree + 1) + n0], &G[size * 1], sizeof(double) * size);
	memcpy(&m_spharm[subj].gradient[2 * (m_degree + 1) * (m_degree + 1) + n0], &G[size * 2], sizeof(double) * size);
#else
	// inv(A) * b
	inverse(M, size * 3, &m_ipiv[(csize * 3 + 1) * subj], &m_work[csize * 3 * csize * 3 * subj]);
	//Gradient::ATB(M, size * 3, size * 3, G, 1, &G[size * 3]);
	ATB(M, size * 3, size * 3, G, 1, &G[size * 3]);
	memcpy(&m_spharm[subj].gradient[n0], &G[size * 3], sizeof(double) * size);
	memcpy(&m_spharm[subj].gradient[(m_degree + 1) * (m_degree + 1) + n0], &G[size * 4], sizeof(double) * size);
	memcpy(&m_spharm[subj].gradient[2 * (m_degree + 1) * (m_degree + 1) + n0], &G[size * 5], sizeof(double) * size);
#endif
}

void HSD::updateGradientLandmark(int deg_beg, int deg_end, int subj_id)
{
	int nLandmark = m_spharm[0].landmark.size();	// # of landmarks: we assume all the subject has the same number
	int nCoeff = (m_degree + 1) * (m_degree + 1);
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int csize = (m_degree + 1) * (m_degree + 1);

	double normalization = 1.0 / (double)((m_nSubj - 1) * nLandmark);
	
	// landmark gradient
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (subj_id != -1 && subj_id != subj) continue;
		#pragma omp parallel for
		for (int i = 0; i < nLandmark; i++)
		{
			// x_bar
			double x_bar[3] = {0, 0, 0};
			for (int j = 0; j < m_nSubj; j++)
				for (int k = 0; k < 3; k++) x_bar[k] += m_feature[j * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3 + k];
			// forcing the mean to be on the sphere
			double norm = sqrt(x_bar[0] * x_bar[0] + x_bar[1] * x_bar[1] + x_bar[2] * x_bar[2]);
			//cout << "[" << i << "] x_bar: " << x_bar[0] << " " << x_bar[1] << " " << x_bar[2] << endl;
			for (int k = 0; k < 3; k++) x_bar[k] /= norm;
		
			const double *Y = m_spharm[subj].landmark[i]->Y;
			double *coeff = m_spharm[subj].coeff;
			double delta[3] = {0, 0, 0};
			for (int j = 0; j < nCoeff; j++)
			{
				delta[0] += Y[j] * coeff[j];
				delta[1] += Y[j] * coeff[(m_degree + 1) * (m_degree + 1) + j];
				delta[2] += Y[j] * coeff[2 * (m_degree + 1) * (m_degree + 1) + j];
			}
			
			// z
			float z_hat[3];
			Coordinate::sph2cart(m_spharm[subj].pole[0], m_spharm[subj].pole[1], z_hat);
			const float *u1 = m_spharm[subj].tan1;
			const float *u2 = m_spharm[subj].tan2;
			//const float *z_dot_orth = (Vector(u1) * (float)delta[1] + Vector(u2) * (float)delta[2]).unit().fv();
			float z_dot_orth[3];
			memcpy(z_dot_orth, (Vector(u1) * (float)delta[1] + Vector(u2) * (float)delta[2]).unit().fv(), sizeof(float) * 3);
			float degree = (float)sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
			Vector Z_ddot = Vector(z_hat).cross(Vector(z_dot_orth));
			//const float *z_ddot = Z_ddot.unit().fv();
			float z_ddot[3];
			memcpy(z_ddot, Z_ddot.unit().fv(), sizeof(float) * 3);
			float rot[9];
			Coordinate::rotation(z_ddot, degree, rot);
			float z_dot[3];
			Coordinate::rotPoint(z_hat, rot, z_dot);
			//cout << "z_hat: " << z_hat[0] << " " << z_hat[1] << " " << z_hat[2] << endl;
			
			// x_hat
			float *x_hat = &m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + i * 3];
			
			double x_hat_dot_x_bar = Vector(x_hat) * Vector(x_bar);
			
			//cout << "x_hat_dot_x_bar: " << x_hat_dot_x_bar << endl;
			double dxdg = 0, dxdu2 = 0, dxdu1 = 0, dEdx = 0, d2Edx2 = 0;
			m_gradient_diag[i] = 0;
			if (fabs(x_hat_dot_x_bar) < 1.0 - 1e-6)
			{
				// z_hatXu
				float z_hatXu1[3], z_hatXu2[3];
				z_hatXu1[0] = u2[0]; z_hatXu1[1] = u2[1]; z_hatXu1[2] = u2[2];
				z_hatXu2[0] = -u1[0]; z_hatXu2[1] = -u1[1]; z_hatXu2[2] = -u1[2];
				
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
				dEdx = 2.0 * acos(x_hat_dot_x_bar);
				double drdx = -1.0 / sqrt(one_minus_x_hat_dot_x_bar_sq);
				dxdg *= drdx;
				dxdu1 *= drdx;
				dxdu2 *= drdx;

				if (m_icosahedron >= m_fine_res) m_gradient_diag[i] = normalization;
				//cout << "dxdg: " << dxdg << " dxdu1: " << dxdu1 << " dxdu2: " << dxdu1 << endl;
			}
			for (int j = n0; j < n; j++)
			{
				m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + j - n0] = Y[j] * dxdg * dEdx * normalization;
				m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size + j - n0] = Y[j] * dxdu1 * dEdx * normalization;
				m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size * 2 + j - n0] = Y[j] * dxdu2 * dEdx * normalization;
				if (m_icosahedron >= m_fine_res)
				{
					m_gradient_raw[size * 3 * i + j - n0] = Y[j] * dxdg;
					m_gradient_raw[size * 3 * i + size + j - n0] = Y[j] * dxdu1;
					m_gradient_raw[size * 3 * i + size * 2 + j - n0] = Y[j] * dxdu2;
				}
			}
		}
		for (int i = 0; i < nLandmark; i++)
		{
			for (int j = n0; j < n; j++)
			{
				m_spharm[subj].gradient[j] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + j - n0];
				m_spharm[subj].gradient[(m_degree + 1) * (m_degree + 1) + j] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size + j - n0];
				m_spharm[subj].gradient[2 * (m_degree + 1) * (m_degree + 1) + j] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size * 2 + j - n0];
			}
		}
		if (m_icosahedron >= m_fine_res)
			updateHessian(deg_beg, deg_end, nLandmark, subj);
	}
}

#ifdef _USE_CUDA_BLAS
void HSD::updateGradientProperties_cuda(int deg_beg, int deg_end, int subj_id, int sid, int ssid)
{
	int nLandmark = m_spharm[0].landmark.size();	// # of landmarks: we assume all the subject has the same number
	int nSamples = m_nQuerySamples;	// # of sampling points for property map agreement
	int nCoeff = (m_degree + 1) * (m_degree + 1);
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int csize = (m_degree + 1) * (m_degree + 1);
	double normalization = m_eta * 1.0 / (double)(((m_nProperties + m_nSurfaceProperties) * nSamples) * m_nSubj);
	
	int k = 0;
	int subj = subj_id;
	int nVertex = m_spharm[subj].sphere->nVertex();
	int nFace = m_spharm[subj].sphere->nFace();
	const float *feature = &m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + nLandmark * 3 + m_nSamples * k];
	const float *propertySamples = m_propertySamples_pinned;
	const double *variance = &m_variance[m_nSamples * k];
	const float *property = &m_spharm[subj].property[nVertex * k];
	const double *mean = &m_mean[m_nSamples * k];
	const float *vertex = m_spharm[subj].vertex1;
	const int *face = m_spharm[subj].face;

	m_cuda_grad[sid]->updateGradientProperties(vertex, nVertex, face, nFace, m_spharm[subj].neighbor, m_spharm[subj].nNeighbor, feature, propertySamples, m_nSamples, variance, property, m_spharm[subj].pole, m_spharm[subj].Y, m_spharm[subj].coeff, m_degree, deg_beg, deg_end, normalization, mean, m_spharm[subj].tan1, m_spharm[subj].tan2, m_spharm[subj].tree_cache, m_spharm[subj].gradient, &m_Hessian[csize * 3 * csize * 3 * subj], m_icosahedron >= m_fine_res, ssid, m_resampling);
}
#endif

void HSD::updateGradientProperties(int deg_beg, int deg_end, int subj_id)
{
	int nLandmark = m_spharm[0].landmark.size();	// # of landmarks: we assume all the subject has the same number
	int nSamples = m_nQuerySamples;	// # of sampling points for property map agreement
	int nCoeff = (m_degree + 1) * (m_degree + 1);
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int csize = (m_degree + 1) * (m_degree + 1);

	const double normalization = m_eta * 1.0 / (double)(((m_nProperties + m_nSurfaceProperties) * nSamples) * m_nSubj);
	
	double *grad = m_gradient_work;
	double *ones = &m_gradient_work[csize * 3];
	memset(grad, 0, sizeof(double) * size * 3);

	for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
	{
		for (int subj = 0; subj < m_nSubj; subj++)
		{
			if (subj_id != -1 && subj_id != subj) continue;
			#pragma omp parallel for
			for (int i = 0; i < nSamples; i++)
			{
				// m_bar
				const double m_bar = m_mean[m_nSamples * k + i];
				// spherical coordinate
				const float *y = &m_propertySamples[i * 3];
				//cout << "y: " << y[0] << " " << y[1] << " " << y[2] << " " << phi << " " << theta << endl;
				float cY[3];
				// approximation of Y
				int fid = m_spharm[subj].tree_cache[i];
				const float *v1 = m_spharm[subj].sphere->face(fid)->vertex(0)->fv();
				const float *v2 = m_spharm[subj].sphere->face(fid)->vertex(1)->fv();
				const float *v3 = m_spharm[subj].sphere->face(fid)->vertex(2)->fv();
				Vector N = Vector(v1, v2).cross(Vector(v2, v3));
				N.unit();
				Vector Yproj = Vector(y) * ((Vector(v1) * N) / (Vector(y) * N));
				Coordinate::cart2bary((float *)v1, (float *)v2, (float *)v3, (float *)Yproj.fv(), cY, 1e-5);
				int id1 = m_spharm[subj].sphere->face(fid)->vertex(0)->id();
				int id2 = m_spharm[subj].sphere->face(fid)->vertex(1)->id();
				int id3 = m_spharm[subj].sphere->face(fid)->vertex(2)->id();
				const double *Y1 = m_spharm[subj].vertex[id1]->Y;
				const double *Y2 = m_spharm[subj].vertex[id2]->Y;
				const double *Y3 = m_spharm[subj].vertex[id3]->Y;
				double *coeff = m_spharm[subj].coeff;

				double delta[3] = {0, 0, 0};
				for (int j = 0; j < nCoeff; j++)
				{
					double Y = (Y1[j] * cY[0] + Y2[j] * cY[1] + Y3[j] * cY[2]);
					delta[0] += Y * coeff[j];
					delta[1] += Y * coeff[(m_degree + 1) * (m_degree + 1) + j];
					delta[2] += Y * coeff[2 * (m_degree + 1) * (m_degree + 1) + j];
				}
				// m
				const double m = m_feature[subj * (nLandmark * 3 + m_nSamples * (m_nProperties + m_nSurfaceProperties)) + nLandmark * 3 + m_nSamples * k + i];

				// z
				float z_hat[3];
				Coordinate::sph2cart(m_spharm[subj].pole[0], m_spharm[subj].pole[1], z_hat);
				const float *u1 = m_spharm[subj].tan1;
				const float *u2 = m_spharm[subj].tan2;
				//const float *z_dot_orth = (Vector(u1) * (float)delta[1] + Vector(u2) * (float)delta[2]).unit().fv();
				float z_dot_orth[3];
				memcpy(z_dot_orth, (Vector(u1) * (float)delta[1] + Vector(u2) * (float)delta[2]).unit().fv(), sizeof(float) * 3);
				float degree = (float)sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
				Vector Z_ddot = Vector(z_hat).cross(Vector(z_dot_orth));
				//const float *z_ddot = Z_ddot.unit().fv();
				float z_ddot[3];
				memcpy(z_ddot, Z_ddot.unit().fv(), sizeof(float) * 3);
				float rot[9];
				Coordinate::rotation(z_ddot, degree, rot);
				float z_dot[3];
				Coordinate::rotPoint(z_hat, rot, z_dot);

				double grad_m[3] = {0, 0, 0};
				int nid[2] = {id2, id3};
				const int *neighbor = nid;
				int nNeighbor = 2;
				double r1 = (Vector(v1) * N) / (Vector(y) * N);
				if (cY[0] == 1 || cY[1] == 1 || cY[2] == 1)
				{
					if (cY[0] == 1) id1 = m_spharm[subj].sphere->face(fid)->vertex(0)->id();
					else if (cY[1] == 1) id1 = m_spharm[subj].sphere->face(fid)->vertex(1)->id();
					else if (cY[2] == 1) id1 = m_spharm[subj].sphere->face(fid)->vertex(2)->id();
					neighbor = m_spharm[subj].sphere->vertex(id1)->list();
					nNeighbor = m_spharm[subj].sphere->vertex(id1)->nNeighbor();
					r1 = 1;
				}

				v1 = m_spharm[subj].sphere->vertex(id1)->fv();
				float area = 0;
				for (int j = 0; j < nNeighbor; j++)
				{
					id2 = neighbor[j];
					id3 = neighbor[(j + 1) % nNeighbor];
					v2 = m_spharm[subj].sphere->vertex(id2)->fv();
					v3 = m_spharm[subj].sphere->vertex(id3)->fv();

					// dp/dx
					Vector N = Vector(v1, v2).cross(Vector(v2, v3));
					area += N.norm();
					const float *nf = N.unit().fv();
					double r2 = r1 / (Vector(y) * N);
					double dpdx[3][3] = {{r1 - r2 * y[0] * nf[0], -r2 * y[0] * nf[1], -r2 * y[0] * nf[2]},
										{-r2 * y[1] * nf[0], r1 - r2 * y[1] * nf[1], -r2 * y[1] * nf[2]},
										{-r2 * y[2] * nf[0], -r2 * y[2] * nf[1], r1 - r2 * y[2] * nf[2]}};
					// dm/dp
					const float m1 = m_spharm[subj].property[m_spharm[subj].sphere->nVertex() * k + id1];
					const float m2 = m_spharm[subj].property[m_spharm[subj].sphere->nVertex() * k + id2];
					const float m3 = m_spharm[subj].property[m_spharm[subj].sphere->nVertex() * k + id3];
					Vector DA1DP = Vector(v2, v3).cross(N).unit() * Vector(v2, v3).norm() * m1;
					Vector DA2DP = Vector(v3, v1).cross(N).unit() * Vector(v3, v1).norm() * m2;
					Vector DA3DP = Vector(v1, v2).cross(N).unit() * Vector(v1, v2).norm() * m3;
					Vector DMDP = (DA1DP + DA2DP + DA3DP);

					const float *dmdp = DMDP.fv();

					// grad_m
					grad_m[0] += dpdx[0][0] * dmdp[0] + dpdx[0][1] * dmdp[1] + dpdx[0][2] * dmdp[2];
					grad_m[1] += dpdx[1][0] * dmdp[0] + dpdx[1][1] * dmdp[1] + dpdx[1][2] * dmdp[2];
					grad_m[2] += dpdx[2][0] * dmdp[0] + dpdx[2][1] * dmdp[1] + dpdx[2][2] * dmdp[2];

					if (nNeighbor == 2) break;
				}
				grad_m[0] /= area; grad_m[1] /= area; grad_m[2] /= area;

				//cout << "grad_m: " << grad_m[0] << " " << grad_m[1] << " " << grad_m[2] << endl;
				
				// z_hatXu
				float z_hatXu1[3], z_hatXu2[3];
				z_hatXu1[0] = u2[0]; z_hatXu1[1] = u2[1]; z_hatXu1[2] = u2[2];
				z_hatXu2[0] = -u1[0]; z_hatXu2[1] = -u1[1]; z_hatXu2[2] = -u1[2];
		
				// [z_dot]x
				double z_dot_x[3][3] = {{0, -z_dot[2], z_dot[1]},
									 {z_dot[2], 0, -z_dot[0]},
									 {-z_dot[1], z_dot[0], 0}};
				// [z_hatXu1]x
				const double z_hatXu1_x[3][3] = {{0, -z_hatXu1[2], z_hatXu1[1]},
										 {z_hatXu1[2], 0, -z_hatXu1[0]},
										 {-z_hatXu1[1], z_hatXu1[0], 0}};
				// [z_hatXu2]x
				const double z_hatXu2_x[3][3] = {{0, -z_hatXu2[2], z_hatXu2[1]},
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
				double dEdx = 2 * (m - m_bar);
				//cout << "dxdg: " << dxdg << " dxdu1: " << dxdu1 << " dxdu2: " << dxdu2 << endl;
				
				double var = (m_pairwise && m_degree_inc == 0) ? 1: m_variance[m_nSamples * k + i];
				if (m_icosahedron >= m_fine_res) m_gradient_diag[i] = normalization / var;
				//totalArea += area;
				
				for (int j = n0; j < n; j++)
				{
					double Y = (Y1[j] * cY[0] + Y2[j] * cY[1] + Y3[j] * cY[2]);
					m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + j - n0] = Y * dxdg * dEdx / var;
					m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size + j - n0] = Y * dxdu1 * dEdx / var;
					m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size * 2 + j - n0] = Y * dxdu2 * dEdx / var;
					if (m_icosahedron >= m_fine_res)
					{
						m_gradient_raw[size * 3 * i + j - n0] = Y * dxdg;
						m_gradient_raw[size * 3 * i + size + j - n0] = Y * dxdu1;
						m_gradient_raw[size * 3 * i + size * 2 + j - n0] = Y * dxdu2;
					}
				}
			}
			/*for (int i = 0; i < nSamples; i++)
			{
				for (int j = n0; j < n; j++)
				{
					grad[j - n0] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + j - n0];
					grad[size + j - n0] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size + j - n0];
					grad[size * 2 + j - n0] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size * 2 + j - n0];
				}
			}*/
			ATB(&m_gradient_raw[m_nMaxVertex * 3 * csize], nSamples, size * 3, ones, 1, grad);
			for (int j = n0; j < n; j++)
			{
				m_spharm[subj].gradient[j] += grad[j - n0] * normalization;
				m_spharm[subj].gradient[(m_degree + 1) * (m_degree + 1) + j] += grad[size + j - n0] * normalization;
				m_spharm[subj].gradient[2 * (m_degree + 1) * (m_degree + 1) + j] += grad[size * 2 + j - n0] * normalization;
			}
			if (m_icosahedron >= m_fine_res)
				updateHessian(deg_beg, deg_end, m_nSamples, subj);
		}
	}
}

#ifdef _USE_CUDA_BLAS
void HSD::updateGradientDisplacement_cuda(int deg_beg, int deg_end, int subj_id, int sid, int ssid)
{
	int nLandmark = m_spharm[0].landmark.size();	// # of landmarks: we assume all the subject has the same number
	int nCoeff = (m_degree + 1) * (m_degree + 1);
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int csize = (m_degree + 1) * (m_degree + 1);

	int subj = subj_id;
	int nVertex = m_spharm[subj].sphere->nVertex();
	
	const double normalization = m_lambda2 / nVertex / (double)(m_nSubj);
	
	const float *vertex0 = m_spharm[subj].vertex0;
	const float *vertex1 = m_spharm[subj].vertex1;

	m_cuda_grad[sid]->updateGradientDsiplacement(vertex0, vertex1, nVertex, m_spharm[subj].pole, m_spharm[subj].Y, m_spharm[subj].coeff, m_degree, deg_beg, deg_end, normalization, m_spharm[subj].tan1, m_spharm[subj].tan2, m_spharm[subj].gradient, &m_Hessian[csize * 3 * csize * 3 * subj], m_icosahedron >= m_fine_res, ssid, m_resampling);
}
#endif

void HSD::updateGradientDisplacement(int deg_beg, int deg_end, int subj_id)
{
	int nLandmark = m_spharm[0].landmark.size();	// # of landmarks: we assume all the subject has the same number
	int nCoeff = (m_degree + 1) * (m_degree + 1);
	int n = (deg_end + 1) * (deg_end + 1);
	int n0 = deg_beg * deg_beg;
	int size = n - n0;
	int csize = (m_degree + 1) * (m_degree + 1);

	double normalization = m_lambda2 / (double)(m_nSubj);
	//if (subj_id != -1) normalization = m_lambda2;

	double *grad = m_gradient_work;
	double *ones = &m_gradient_work[csize * 3];
	memset(grad, 0, sizeof(double) * size * 3);
	
	// displacement gradient
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (subj_id != -1 && subj_id != subj) continue;
		int nVertex = m_spharm[subj].sphere->nVertex();
		#pragma omp parallel for
		for (int i = 0; i < nVertex; i++)
		{
			//const float *x_bar = m_spharm[subj].vertex[i]->p0;
			float x_bar[3];
			memcpy(x_bar, m_spharm[subj].vertex[i]->p0, sizeof(float) * 3);
			const double *Y = m_spharm[subj].vertex[i]->Y;
			double *coeff = m_spharm[subj].coeff;
			double delta[3] = {0, 0, 0};
			for (int j = 0; j < nCoeff; j++)
			{
				delta[0] += Y[j] * coeff[j];
				delta[1] += Y[j] * coeff[(m_degree + 1) * (m_degree + 1) + j];
				delta[2] += Y[j] * coeff[2 * (m_degree + 1) * (m_degree + 1) + j];
			}
			
			// z
			float z_hat[3];
			Coordinate::sph2cart(m_spharm[subj].pole[0], m_spharm[subj].pole[1], z_hat);
			const float *u1 = m_spharm[subj].tan1;
			const float *u2 = m_spharm[subj].tan2;
			float z_dot_orth[3];
			memcpy(z_dot_orth, (Vector(u1) * (float)delta[1] + Vector(u2) * (float)delta[2]).unit().fv(), sizeof(float) * 3);
			float degree = (float)sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
			Vector Z_ddot = Vector(z_hat).cross(Vector(z_dot_orth));
			float z_ddot[3];
			memcpy(z_ddot, Z_ddot.unit().fv(), sizeof(float) * 3);
			float rot[9];
			Coordinate::rotation(z_ddot, degree, rot);
			float z_dot[3];
			Coordinate::rotPoint(z_hat, rot, z_dot);
			//cout << "z_hat: " << z_hat[0] << " " << z_hat[1] << " " << z_hat[2] << endl;
			
			// x_hat
			Vertex *vert = (Vertex *)m_spharm[subj].sphere->vertex(i);
			//const float *x_hat = vert->fv();
			float x_hat[3];
			memcpy(x_hat, vert->fv(), sizeof(float) * 3);
			
			double x_hat_dot_x_bar = Vector(x_hat) * Vector(x_bar);
			
			//cout << "x_hat_dot_x_bar: " << x_hat_dot_x_bar << endl;
			double dxdg = 0, dxdu2 = 0, dxdu1 = 0, dEdx = 0, d2Edx2 = 0;
			m_gradient_diag[i] = 0;
			if (fabs(x_hat_dot_x_bar) < 0.999999)	// prevent too much divergence: 1 = x_hat and x_bar matched
			{
				// z_hatXu
				float z_hatXu1[3], z_hatXu2[3];
				z_hatXu1[0] = u2[0]; z_hatXu1[1] = u2[1]; z_hatXu1[2] = u2[2];
				z_hatXu2[0] = -u1[0]; z_hatXu2[1] = -u1[1]; z_hatXu2[2] = -u1[2];
			
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
				dEdx = 2.0 * acos(x_hat_dot_x_bar);
				double drdx = -1.0 / sqrt(one_minus_x_hat_dot_x_bar_sq);
				dxdg *= drdx;
				dxdu1 *= drdx;
				dxdu2 *= drdx;

				if (m_icosahedron >= m_fine_res) m_gradient_diag[i] = normalization / nVertex;
				
				//cout << "dxdg: " << dxdg << " dxdu1: " << dxdu1 << " dxdu2: " << dxdu1 << endl;
			}
			
			for (int j = n0; j < n; j++)
			{
				m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + j - n0] = Y[j] * dxdg * dEdx;
				m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size + j - n0] = Y[j] * dxdu1 * dEdx;
				m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size * 2 + j - n0] = Y[j] * dxdu2 * dEdx;
				if (m_icosahedron >= m_fine_res)
				{
					m_gradient_raw[size * 3 * i + j - n0] = Y[j] * dxdg;
					m_gradient_raw[size * 3 * i + size + j - n0] = Y[j] * dxdu1;
					m_gradient_raw[size * 3 * i + size * 2 + j - n0] = Y[j] * dxdu2;
				}
			}
		}
		/*for (int i = 0; i < nVertex; i++)
		{
			for (int j = n0; j < n; j++)
			{
				grad[j - n0] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + j - n0];
				grad[size + j - n0] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size + j - n0];
				grad[size * 2 + j - n0] += m_gradient_raw[m_nMaxVertex * 3 * csize + size * 3 * i + size * 2 + j - n0];
			}
		}*/
		ATB(&m_gradient_raw[m_nMaxVertex * 3 * csize], nVertex, size * 3, ones, 1, grad);
		for (int j = n0; j < n; j++)
		{
			m_spharm[subj].gradient[j] += grad[j - n0] * normalization / nVertex;
			m_spharm[subj].gradient[(m_degree + 1) * (m_degree + 1) + j] += grad[size + j - n0] * normalization / nVertex;
			m_spharm[subj].gradient[2 * (m_degree + 1) * (m_degree + 1) + j] += grad[size * 2 + j - n0] * normalization / nVertex;
		}
		if (m_icosahedron >= m_fine_res)
			updateHessian(deg_beg, deg_end, nVertex, subj);
	}
}

void HSD::guessInitCoeff(void)
{
	m_nQuerySamples = min(10242, m_nSamples);

	#pragma omp parallel for
	for (int subj = 0; subj < m_nSubj; subj++)
	{
		if (m_spharm[subj].fixed) continue;

		// spharm basis
		int n = (m_degree + 1) * (m_degree + 1);
		int nLandmark = m_spharm[0].landmark.size();
		double *coeff = m_spharm[subj].coeff;
		double *coeff_prev_step = m_spharm[subj].coeff_prev_step;
		double mincost = FLT_MAX;

		bool skip = false;
		for (int i = 0; i < n * 3 && !skip; i++)
			skip = (coeff[i] != 0);
		if (skip) continue;

		// cart coordinate
		float axis0[3], axis1[3];
		Coordinate::sph2cart(m_spharm[subj].pole[0], m_spharm[subj].pole[1], axis0);
		Vector P = axis0;

		for (double c1 = 0; c1 <= PI / 4; c1 += PI / 16)
		{
			for (double c2 = 0; c2 < 2 * PI; c2 += PI / 8)
			{
				if (c1 == 0 && c2 > 0) continue;
				for (double c3 = -PI / 8; c3 <= PI / 8; c3 += PI / 16)
				{
					double delta[3];
					delta[0] = c3;
					delta[1] = c1 * cos(c2);
					delta[2] = c1 * sin(c2);

					// exponential map (linear)
					const float *axis = (P + Vector(m_spharm[subj].tan1) * delta[1] + Vector(m_spharm[subj].tan2) * delta[2]).unit().fv();
					axis1[0] = axis[0]; axis1[1] = axis[1]; axis1[2] = axis[2];

					// standard pole
					Vector Q = axis1;
					float angle = (float)c1;
					Vector A = P.cross(Q); A.unit();

					/*float rv[3];
					float rot[9];

					for (int i = 0; i < m_spharm[subj].vertex.size(); i++)
					{
						Vertex *v = (Vertex *)m_spharm[subj].sphere->vertex(i);
						float v1[3];
						const float *v0 = m_spharm[subj].vertex[i]->p;
						Coordinate::rotation(A.fv(), angle, rot);
						Coordinate::rotPoint(v0, rot, rv);

						// rotation
						Coordinate::rotation(Q.fv(), (float)delta[0], rot);
						Coordinate::rotPoint(rv, rot, v1);

						Vector V(v1); V.unit();
						v->setVertex(V.fv());
					}
					m_updated[subj] = false;

					double cost = trace(subj);*/

					double cost = 0;
					if (nLandmark > m_spharm[subj].landmark.size())
					{
						updateLandmark(subj);
						cost += varLandmarks(subj);
					}
					if (m_nQuerySamples > 0)
					{
						float rot[9], rot1[9], rot2[9];
						float v1[3];
						Coordinate::rotation(A.fv(), -angle, rot1);
						Coordinate::rotation(Q.fv(), (float)-delta[0], rot2);
						rot[0] = rot1[0] * rot2[0] + rot1[1] * rot2[3] + rot1[2] * rot2[6];
						rot[1] = rot1[0] * rot2[1] + rot1[1] * rot2[4] + rot1[2] * rot2[7];
						rot[2] = rot1[0] * rot2[2] + rot1[1] * rot2[5] + rot1[2] * rot2[8];
						rot[3] = rot1[3] * rot2[0] + rot1[4] * rot2[3] + rot1[5] * rot2[6];
						rot[4] = rot1[3] * rot2[1] + rot1[4] * rot2[4] + rot1[5] * rot2[7];
						rot[5] = rot1[3] * rot2[2] + rot1[4] * rot2[5] + rot1[5] * rot2[8];
						rot[6] = rot1[6] * rot2[0] + rot1[7] * rot2[3] + rot1[8] * rot2[6];
						rot[7] = rot1[6] * rot2[1] + rot1[7] * rot2[4] + rot1[8] * rot2[7];
						rot[8] = rot1[6] * rot2[2] + rot1[7] * rot2[5] + rot1[8] * rot2[8];
						for (int i = 0; i < m_nQuerySamples; i++)
						{
							for (int k = 0; k < m_nProperties + m_nSurfaceProperties; k++)
							{
								const float *v0 = &m_propertySamples[i * 3];
								Coordinate::rotPoint(v0, rot, v1);

								float bary[3];
								int fid = m_spharm[subj].tree->closestFace(v1, bary);
								int nVertex = m_spharm[subj].sphere->nVertex();
								double p = propertyInterpolation(&m_spharm[subj].property[nVertex * k], fid, bary, m_spharm[subj].sphere);
								double m = m_mean[m_nSamples * k + i];
								double pm = (p - m);
								cost += pm * pm / ((m_nProperties + m_nSurfaceProperties) * m_nSamples);
							}
						}
					}

					if (cost < mincost)
					{
						coeff[0] = delta[0] / m_spharm[subj].Y[0];
						coeff[n] = delta[1] / m_spharm[subj].Y[0];
						coeff[2 * n] = delta[2] / m_spharm[subj].Y[0];
						mincost = cost;
						/*cout << mincost << ": ";
						cout << c1 << " " << c2 << " " << c3 << endl;*/
					}
				}
			}
		}
	}
	m_nQuerySamples = m_nSamples;
	#pragma omp parallel for
	for (int subj = 0; subj < m_nSubj; subj++) updateDeformation(subj, true);
	memset(m_updated, 0, sizeof(bool) * m_nSubj);
	trace();
	memcpy(m_coeff_prev_step, m_coeff, sizeof(double) * m_csize * 3);
	m_guess = false;
}

void HSD::optimization(void)
{
	const int step = 1;
	
	if (m_guess)
	{
		cout << "Guess initial coeffs.. ";
		fflush(stdout);
		guessInitCoeff();
		cout << "done" << endl;
	}

	while (m_degree_inc <= m_degree)
	{
		nIter = 0;
		cout << "Phase: " << m_degree_inc << endl;
		if (m_icosahedron < m_fine_res)
			minGradientDescent(0, m_degree_inc);
		else
			minGradientDescent(m_degree_inc, m_degree_inc);
		if (m_degree_inc == m_degree) break;
		m_degree_inc = min(m_degree_inc + step, m_degree);
		if (m_degree_inc == 1 || !m_multi_res)
		{
			if (m_degree_inc == 1)
				for (int subj = 0; subj < m_nSubj; subj++) initTriangleFlipping(subj);
			//updateDisplacement();
			updateDisplacement(-1, m_degree);
			updatePropertyStats();	// needs for samples
		}
	}
	if (!m_multi_res) m_fine_res = m_icosahedron - 1;
	if (m_multi_res || (!m_multi_res && m_icosahedron == 7))
	{
		cout << "Phase: Final" << endl;
		nIter = 0;
		m_degree_inc = m_degree;
		if (!m_multi_res) updateDisplacement(-1, m_degree);
		updatePropertyStats();	// needs for samples
		minGradientDescent(0, m_degree_inc);
	}
}

int HSD::icosahedron(int degree, Mesh *mesh)
{
	vector<vert **> triangles;
	vector<vert *> vertices;
	
	float t = (1 + sqrt(5.0)) / 2.0;
	float s = sqrt(1 + t * t);

	int id = 0;
	// create the 12 vertices
	vert v0; v0.v = Vector(t, 1.0, 0.0) / s; v0.id = id++; vertices.push_back(&v0);
	vert v1; v1.v = Vector(-t, 1.0, 0.0) / s; v1.id = id++; vertices.push_back(&v1);
	vert v2; v2.v = Vector(t, -1.0, 0.0) / s; v2.id = id++; vertices.push_back(&v2);
	vert v3; v3.v = Vector(-t, -1.0, 0.0) / s; v3.id = id++; vertices.push_back(&v3);
	vert v4; v4.v = Vector(1.0, 0.0, t) / s; v4.id = id++; vertices.push_back(&v4);
	vert v5; v5.v = Vector(1.0, 0.0, -t) / s; v5.id = id++; vertices.push_back(&v5);
	vert v6; v6.v = Vector(-1.0, 0.0, t) / s; v6.id = id++; vertices.push_back(&v6);
	vert v7; v7.v = Vector(-1.0, 0.0, -t) / s; v7.id = id++; vertices.push_back(&v7);
	vert v8; v8.v = Vector(0.0, t, 1.0) / s; v8.id = id++; vertices.push_back(&v8);
	vert v9; v9.v = Vector(0.0, -t, 1.0) / s; v9.id = id++; vertices.push_back(&v9);
	vert v10; v10.v = Vector(0.0, t, -1.0) / s; v10.id = id++; vertices.push_back(&v10);
	vert v11; v11.v = Vector(0.0, -t, -1.0) / s; v11.id = id++; vertices.push_back(&v11);
    
	// create the 20 triangles
	vert **f; 
	f = new vert*[3]; f[0] = &v0; f[1] = &v8; f[2] = &v4; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v1; f[1] = &v10; f[2] = &v7; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v2; f[1] = &v9; f[2] = &v11; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v7; f[1] = &v3; f[2] = &v1; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v0; f[1] = &v5; f[2] = &v10; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v3; f[1] = &v9; f[2] = &v6; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v3; f[1] = &v11; f[2] = &v9; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v8; f[1] = &v6; f[2] = &v4; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v2; f[1] = &v4; f[2] = &v9; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v3; f[1] = &v7; f[2] = &v11; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v4; f[1] = &v2; f[2] = &v0; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v9; f[1] = &v4; f[2] = &v6; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v2; f[1] = &v11; f[2] = &v5; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v0; f[1] = &v10; f[2] = &v8; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v5; f[1] = &v0; f[2] = &v2; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v10; f[1] = &v5; f[2] = &v7; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v1; f[1] = &v6; f[2] = &v8; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v1; f[1] = &v8; f[2] = &v10; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v6; f[1] = &v1; f[2] = &v3; triangles.push_back(f);
	f = new vert*[3]; f[0] = &v11; f[1] = &v7; f[2] = &v5; triangles.push_back(f);

	// subdivision
	for (int d = 0; d < degree; d++)
	{
		int nFaces = triangles.size();
		map<pair<int, int>, SurfaceUtil::edge> edgeList;

		for (int i = 0 ; i < nFaces; i++)
		{
			vert **f = triangles[i];
			vert *a = f[0], *b = f[1], *c = f[2];
			Vector ab = a->v + b->v;
			Vector ca = c->v + a->v;
			Vector bc = b->v + c->v;
			
			// normalization
			ab.unit(); ca.unit(); bc.unit();

			SurfaceUtil::edge e1, e2, e3;
			e1.vid1 = a->id; e1.vid2 = b->id; if (a->id > b->id) swap(e1.vid1, e1.vid2);
			e2.vid1 = c->id; e2.vid2 = a->id; if (c->id > a->id) swap(e2.vid1, e2.vid2);
			e3.vid1 = b->id; e3.vid2 = c->id; if (b->id > c->id) swap(e3.vid1, e3.vid2);
			
			map<pair<int, int>, SurfaceUtil::edge>::iterator it;
			
			// update new list
			int id1, id2, id3;
			it = edgeList.find(make_pair(e1.vid1, e1.vid2));
			if (it == edgeList.end())
			{
				vert *v1 = new vert();
				v1->v = ab; v1->id = id++; vertices.push_back(v1);
				e1.fid1 = vertices.size() - 1;
				edgeList.insert(make_pair(make_pair(e1.vid1, e1.vid2), e1));
				id1 = e1.fid1;
			}
			else id1 = it->second.fid1;
			it = edgeList.find(make_pair(e2.vid1, e2.vid2));
			if (it == edgeList.end())
			{
				vert *v2 = new vert();
				v2->v = ca; v2->id = id++; vertices.push_back(v2);
				e2.fid1 = vertices.size() - 1;
				edgeList.insert(make_pair(make_pair(e2.vid1, e2.vid2), e2));
				id2 = e2.fid1;
			}
			else id2 = it->second.fid1;
			it = edgeList.find(make_pair(e3.vid1, e3.vid2));
			if (it == edgeList.end())
			{
				vert *v3 = new vert();
				v3->v = bc; v3->id = id++; vertices.push_back(v3);
				e3.fid1 = vertices.size() - 1;
				edgeList.insert(make_pair(make_pair(e3.vid1, e3.vid2), e3));
				id3 = e3.fid1;
			}
			else id3 = it->second.fid1;
			
			// overwrite the original
			f[0] = vertices[id1]; f[1] = vertices[id2]; f[2] = vertices[id3];
			if ((f[1]->v - f[0]->v).cross(f[2]->v - f[0]->v) * f[0]->v < 0) swap(f[1], f[2]);
			/*Vector f1[3] = {a, v1, v2}; triangles.push_back(f1);
			Vector f2[3] = {c, v2, v3}; triangles.push_back(f2);
			Vector f3[3] = {b, v3, v1}; triangles.push_back(f3);*/
			vert **f1 = new vert*[3]; f1[0] = a; f1[1] = vertices[id1]; f1[2] = vertices[id2];
			if ((f1[1]->v - f1[0]->v).cross(f1[2]->v - f1[0]->v) * f1[0]->v < 0) swap(f1[1], f1[2]);
			triangles.push_back(f1);
			vert **f2 = new vert*[3]; f2[0] = c; f2[1] = vertices[id2]; f2[2] = vertices[id3];
			if ((f2[1]->v - f2[0]->v).cross(f2[2]->v - f2[0]->v) * f2[0]->v < 0) swap(f2[1], f2[2]);
			triangles.push_back(f2);
			vert **f3 = new vert*[3]; f3[0] = b; f3[1] = vertices[id3]; f3[2] = vertices[id1];
			if ((f3[1]->v - f3[0]->v).cross(f3[2]->v - f3[0]->v) * f3[0]->v < 0) swap(f3[1], f3[2]);
			triangles.push_back(f3);
		}
		edgeList.clear();
	}

	for (int i = 0; i < vertices.size(); i++)
	{
		float p[3];
		p[0] = vertices[i]->v[0];
		p[1] = vertices[i]->v[1];
		p[2] = vertices[i]->v[2];
		m_propertySamples.push_back(p[0]);
		m_propertySamples.push_back(p[1]);
		m_propertySamples.push_back(p[2]);
	}
	
	float *vertex = new float[vertices.size() * 3];
	int *face = new int[triangles.size() * 3];
	
	for (int i = 0; i < vertices.size(); i++)
	{
		vertex[i * 3 + 0] = vertices[i]->v[0];
		vertex[i * 3 + 1] = vertices[i]->v[1];
		vertex[i * 3 + 2] = vertices[i]->v[2];
	}
	for (int i = 0; i < triangles.size(); i++)
	{
		face[i * 3 + 0] = triangles[i][0]->id;
		face[i * 3 + 1] = triangles[i][1]->id;
		face[i * 3 + 2] = triangles[i][2]->id;
	}
	
	mesh->setMesh(vertex, face, NULL, vertices.size(), triangles.size(), 0, false);

	// delete resources
	for (int i = 0; i < triangles.size(); i++)
		delete [] triangles[i];
	for (int i = 12; i < vertices.size(); i++)
		delete vertices[i];
	delete [] vertex;
	delete [] face;
	
	return m_propertySamples.size() / 3;
}

int HSD::sphericalCoord(int degree)
{
	// subdivision
	float intv = PI / degree;
	for (int i = 0; i < degree; i++)
	{
		for (int j = 0 ; j < degree; j++)
		{
			float p[3];
			Coordinate::sph2cart(PI - 2 * intv * i, PI / 2 - intv * j, p);
			m_propertySamples.push_back(p[0]);
			m_propertySamples.push_back(p[1]);
			m_propertySamples.push_back(p[2]);
		}
	}

	return m_propertySamples.size() / 3;
}

void HSD::saveCoeff(const char *filename, int id)
{
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "%d\n", m_spharm[id].degree);
	fprintf(fp, "%f %f\n", m_spharm[id].pole[0], m_spharm[id].pole[1]);
	int n = (m_spharm[id].degree + 1) * (m_spharm[id].degree + 1);
	for (int i = 0; i < n; i++)
	{
		fprintf(fp, "%lf %lf %lf\n", m_spharm[id].coeff[i], m_spharm[id].coeff[n + i], m_spharm[id].coeff[2 * n + i]);
	}
	fclose(fp);
}

void HSD::saveSphere(const char *filename, int id)
{
	if (m_resampling)
	{
		int nVertex = m_spharm[id].sphere0->nVertex();
		double *Y = new double[(m_degree + 1) * (m_degree + 1) * nVertex];
		#pragma omp parallel for
		for (int i = 0; i < nVertex; i++)
		{
			Vertex *v = (Vertex *)m_spharm[id].sphere0->vertex(i);	// vertex information on the sphere
			const float *v0 = v->fv();
			double vd[3] = {v0[0], v0[1], v0[2]};
			SphericalHarmonics::basis(m_degree, vd, &Y[i * (m_degree + 1) * (m_degree + 1)]);
			float v1[3];
			updateCoordinate(v0, v1, &Y[i * (m_degree + 1) * (m_degree + 1)], (const double *)m_spharm[id].coeff, m_degree, m_spharm[id].pole, m_spharm[id].tan1, m_spharm[id].tan2);
			Vector V(v1); V.unit();
			v->setVertex(V.fv());
		}
		delete [] Y;
	}
	m_spharm[id].sphere0->saveFile(filename, "vtk");
}

double HSD::optimalMeanCost(double *coeff, int subj)
{
	float pole[3];
	Coordinate::sph2cart(m_spharm[subj].pole[0] + (float)coeff[0], m_spharm[subj].pole[1] + (float)coeff[1], pole);
	float cost = 0;
	for (int i = 0; i < m_spharm[subj].landmark.size(); i++)
	{
		float inner = Vector(pole) * Vector(m_spharm[subj].landmark[i]->p);
		if (inner > 1) inner = 1;
		else if (inner < -1) inner = -1;
		float arclen = acos(inner);
		cost += arclen * arclen;
	}

	return cost;
}

