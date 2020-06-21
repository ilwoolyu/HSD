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

	/*
	app.add_option("--sphereDir", dirSphere, "Specify a directory of sphere files")->check(CLI::ExistingDirectory)->group("Directory inputs");
	app.add_option("--propertyDir", dirProperty, "Specify a directory of property files")->check(CLI::ExistingDirectory)->group("Directory inputs");
	app.add_option("--outputDir", dirOutput, "Specify a directory of output files")->check(CLI::ExistingDirectory)->group("Directory inputs");
	//app.add_option("--landmarkDir", dirLandmark, "Specify a directory of landmark files")->check(CLI::ExistingDirectory);
	app.add_option("--coefficientDir", dirCoeff, "Specify a directory of previous spherical harmonics coefficient files")->check(CLI::ExistingDirectory)->group("Directory inputs");
	app.add_option("--filter", listFilter, "Specify a list of suffix filters to select desired property files")->group("Directory inputs");
	//app.add_option("--surfaceDir", dirSurf, "Specify a directory of surface model files for location information")->check(CLI::ExistingDirectory);*/
	app.add_option("-s,--sphere", listSphere, "Specify a list of sphere files")->required()->check(CLI::ExistingFile)->group("File inputs");
	app.add_option("-p,--property", listProperty, "Specify a list of property files")->required()->check(CLI::ExistingFile)->group("File inputs");
	app.add_option("-o,--output", listOutput, "Specify a list of output sphere files")->group("File inputs");
	app.add_option("--outputcoeff", listOutputCoeff, "Specify a list of output coeff files")->group("File inputs");
	//app.add_option("-l,--landmark", listLandmark, "Specify a list of landmark files")->check(CLI::ExistingFile);
	app.add_option("-c,--coefficient", listCoeff, "Specify a list of previous spherical harmonics coefficient files")->check(CLI::ExistingFile)->group("File inputs");
	//app.add_option("--surface", listSurf, "Specify a list of surface model files for location information")->check(CLI::ExistingFile);
	//app.add_option("--weightMap", weightMap, "Specify an overall property weight (eta)", true)->check(CLI::NonNegativeNumber);
	app.add_option("-d,--degree", degree, "Specify a degree of spherical harmonics decomposition", true)->check(CLI::NonNegativeNumber)->group("Optimization");
	app.add_option("--icosahedron", icosa, "Select a icosahedral subdivision level for uniform sampling points", true)->check(CLI::Range(0,7))->group("Optimization");
	app.add_option("--weight", listWeight, "Specify weights for each property")->check(CLI::NonNegativeNumber)->group("Optimization");
	app.add_option("--idprior", idprior, "Specify inverse of distortion prior for the regularization", true)->check(CLI::NonNegativeNumber)->group("Optimization");
	app.add_option("--maxIter", maxIter, "Specify the maxmum number of iterations at the final phase", true)->check(CLI::NonNegativeNumber)->group("Optimization");
	//app.add_option("--locationWeight", weightLoc, "Specify a weighting factor of location information", true);
	app.add_option("--icomesh", icoMesh, "Specify a pre-defined icosahedral mesh, which overrides --icosaherdon")->check(CLI::ExistingFile)->group("Optimization");
	app.add_flag("--noguess", noguess, "Do not execute an initial guess for rigid alignment")->group("Optimization");
	app.add_flag("--resample", resampling, "Resample geometric properties using the current icosahedral level")->group("Optimization");
	app.add_flag("--writecoeff", realtimeCoeff, "Write coefficient whenever the energy is minimized, which may lead to significant IO overhead")->group("Optimization");
	app.add_option("--fixedSubjects", listFixedSubj, "Select indices (starting from 0) of the subjects not being deformed during the optimization (typically for template models)")->check(CLI::NonNegativeNumber)->group("Pair-wise registration");
	app.add_option("--tmpVar", tmpVariance, "Specify a prior of feature variance (only works on pairwise registration)")->check(CLI::ExistingFile)->group("Pair-wise registration");
	app.add_option("--nThreads", nThreads, "Specify the number of OpenMP cores (0: OMP_NUM_THREADS or 1)", true)->check(CLI::NonNegativeNumber)->group("Multi-threading");
#ifdef _USE_CUDA_BLAS
	app.add_option("--nStreams", nCThreads, "Specify the number of CUDA streams (0: use full GPU capacity)", true)->check(CLI::NonNegativeNumber)->group("Multi-threading");
#endif
	try
	{
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError &e)
	{
		exit(app.exit(e));
	}
}


