#include <cstring>
#include <vector>
#include "CLI11.hpp"

std::string dirProperty;
std::string dirSphere;
std::string dirOutput;
std::string dirLandmark;
std::string dirCoeff;
std::string dirSurf;
std::vector<std::string> listProperty;
std::vector<std::string> listSphere;
std::vector<std::string> listOutputCoeff;
std::vector<std::string> listOutput;
std::vector<std::string> listLandmark;
std::vector<std::string> listCoeff;
std::vector<std::string> listSurf;
float weightMap = 1;
std::vector<float> listWeight;
float idprior = 625;
int degree = 15;
int maxIter = 20;
int icosa = 7;
float weightLoc = 0;
std::vector<std::string> listFilter;
std::string tmpVariance;
std::string icoMesh;
bool realtimeCoeff = false;
bool noguess = false;
bool resampling = false;
int nThreads = 0;
std::vector<int> listFixedSubj;
int nCThreads = 0;

void PARSE_ARGS(int argc, char **argv)
{
    
    std::string desc("Hierarchical Spherical Deformation for Cortical Surface Registration "
					 HSD_VERSION "\n"
					 "Author: Ilwoo Lyu\n"
					 "Please refer to the following papers for details:\n"
					 "[1] Lyu et al., Hierarchical Spherical Deformation for Shape Correspondence, Medical Image Computing and Computer Assisted Intervention (MICCAI) 2018, LNCS11070, 853-861, 2018.\n"
					 "[2] Lyu et al., Hierarchical Spherical Deformation for Cortical Surface Registration, Medical Image Analysis, 57, 72-88, 2019.\n"
					 );

    CLI::App app(desc);

	app.add_option("--propertyDir", dirProperty, "provides a directory of property files")->check(CLI::ExistingDirectory);
	app.add_option("--sphereDir", dirSphere, "provides a directory of sphere files")->check(CLI::ExistingDirectory);
	app.add_option("--outputDir", dirOutput, "provides a directory of output files")->check(CLI::ExistingDirectory);
	app.add_option("--landmarkDir", dirLandmark, "provides a directory of landmark files")->check(CLI::ExistingDirectory);
	app.add_option("--coefficientDir", dirCoeff, "provides a directory of previous spherical harmonics coefficient files")->check(CLI::ExistingDirectory);
	app.add_option("--surfaceDir", dirSurf, "provides a directory of surface model files for location information")->check(CLI::ExistingDirectory);
	app.add_option("-p,--property", listProperty, "provides a list of property files")->check(CLI::ExistingFile);
	app.add_option("-s,--sphere", listSphere, "provides a list of sphere files")->required()->check(CLI::ExistingFile);
	app.add_option("--outputcoeff", listOutputCoeff, "provides a list of output coeff files");
	app.add_option("-o,--output", listOutput, "provides a list of output sphere files");
	app.add_option("-l,--landmark", listLandmark, "provides a list of landmark files")->check(CLI::ExistingFile);
	app.add_option("-c,--coefficient", listCoeff, "provides a list of previous spherical harmonics coefficient files")->check(CLI::ExistingFile);
	app.add_option("--surface", listSurf, "provides a list of surface model files for location information")->check(CLI::ExistingFile);
	app.add_option("--weightMap", weightMap, "provides an overall property weight (eta)", true)->check(CLI::NonNegativeNumber);
	app.add_option("-w,--weight", listWeight, "provides weights for each property")->check(CLI::NonNegativeNumber);
	app.add_option("--idprior", idprior, "provides inverse of distortion prior for the regularization", true)->check(CLI::NonNegativeNumber);
	app.add_option("-d,--degree", degree, "provides a degree of spherical harmonics decomposition", true)->check(CLI::NonNegativeNumber);
	app.add_option("--maxIter", maxIter, "provides the maxmum number of iterations", true)->check(CLI::NonNegativeNumber);
	app.add_option("--icosahedron", icosa, "provides a icosahedron subdivision level for uniform sampling points", true)->check(CLI::Range(0,7));
	app.add_option("--locationWeight", weightLoc, "provides a weighting factor of location information", true);
	app.add_option("--filter", listFilter, "provides a list of suffix filters to select desired property files");
	app.add_option("--tmpVar", tmpVariance, "provides a prior of feature variance (only works on pairwise registration)")->check(CLI::ExistingFile);
	app.add_option("--icomesh", icoMesh, "provides a pre-defined icosahedron mesh")->check(CLI::ExistingFile);
	app.add_flag("--writecoeff", realtimeCoeff, "enables real-time coefficient writing whenever the cost function is minimized, which may lead to significant IO overhead");
	app.add_flag("--noguess", noguess, "disables an initial guess for rigid alignment");
	app.add_flag("--resample", resampling, "enables resampling of geometric properties using the current icosahedron level");
	app.add_option("--nThreads", nThreads, "sets the number of OpenMP cores (0: OMP_NUM_THREADS or 1)", true)->check(CLI::NonNegativeNumber);
#ifdef _USE_CUDA_BLAS
	app.add_option("--nStreams", nCThreads, "sets the number of CUDA streams (0: use full GPU capacity)", true)->check(CLI::NonNegativeNumber);
#endif
	app.add_option("--fixedSubjects", listFixedSubj, "specifies indices (starting from 0) of the subjects not being deformed during the optimization (typically for template models)")->check(CLI::NonNegativeNumber);
	try
	{
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError &e)
	{
		exit(app.exit(e));
	}
}


