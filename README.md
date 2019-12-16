# Hierarchical Spherical Deformation for Cortical Surface Registration

## Description
We present hierarchical spherical deformation for group-wise shape correspondence to address template selection bias and to minimize registration distortion. In this work, we aim at a continuous and smooth deformation field to guide accurate cortical surface registration. In conventional spherical registration methods, global rigid alignment and local deformation are independently preformed. Motivated by the composition of precession and intrinsic rotation, we simultaneously optimize global rigid rotation and non-rigid local deformation by utilizing spherical harmonics interpolation of local composite rotations in a single framework. To this end, we indirectly encode local displacements by such local composite rotations as functions of spherical locations. Furthermore, we introduce an additional regularization term to the spherical deformation, which maximizes its rigidity while reducing registration distortion. To improve surface registration performance, we employ the second order approximation of the energy function that enables fast convergence of the optimization. In the experiments, we show an improved shape correspondence with high accuracy in cortical surface parcellation and significantly low registration distortion in surface area and edge length.

![image](https://user-images.githubusercontent.com/9325798/48306768-a3784c00-e504-11e8-930c-94fccf3ce7e6.png)

## Environment
* Parallel processing: this tool supports OpenMP and will perform the best efficiency with CUDA.
* Memory: it requires about 0.5Gb per subject.
## Installation
You can download and compile the source code using <a href="https://cmake.org/">CMake</a>. Or you can pull <a href="https://hub.docker.com/r/ilwoolyu/cmorph/">docker image</a>:
```
$ docker pull ilwoolyu/cmorph:<version>
```
## Usage
### Input
* surface file (.vtk): triangular sphere mesh
* feature file (.txt): scalar map - scalar value per line corresponding to vertex, i.e., # of lines = # of vertices

### Output
* surface file (.vtk): triangular deformed sphere mesh

### Basic commands
This tools supports N many subjects in theory as long as memory capacity is allowed. Let's assume N=3 in this example.
The following command line will generate `s?.sphere.reg.vtk`:
```
$ HSD \
      -s s1.sphere.vtk,s2.sphere.vtk,s3.sphere.vtk \
      -p s1.curv.txt,s2.curv.txt,s3.curv.txt \
      -o s1.sphere.reg.vtk,s2.sphere.reg.vtk,s3.sphere.reg.vtk
```
To change the degree of spherical harmonics:
```
$ HSD -d <# of degree>
```
To change the level of icosahedron subdivision for feature map resampling:
```
$ HSD --icosahedron <level>
```
To load pre-defined icosahedron mesh for feature map resampling (this option will ignore `--icosahedron`):
```
$ HSD --icomesh <filename>
```
To report spherical harmonics coefficients:
```
$ HSD --writecoeff s1.coeff.txt,s2.coeff.txt,s3.coeff.txt
```
and to use initial spherical harmonics coefficients:
```
$ HSD -c s1.coeff.txt,s2.coeff.txt,s3.coeff.txt
```
To enable multi-thread support (OpenMP):
```
$ HSD --nThreads <# of threads>
```
### Multi-resolution approach
If multi-feature maps are available, surface registration can be performed in a multi-resolution manner. Once again, we assume N=3 with the following features: <curvature map of inflated surfaces: `s1.inflated.curv.txt`, `s2.inflated.curv.txt`, `s3.inflated.curv.txt`>, <sulcal depth map: `s1.sulc.txt`, `s2.sulc.txt`, `s3.sulc.txt`>, and <curvature map of cortical surfaces: `s1.curv.txt`, `s2.curv.txt`, `s3.curv.txt`>. Let's coregister *inflated.curv* maps first at low resolution `--icosahedron 4`:
```
$ HSD \
      -s s1.sphere.vtk,s2.sphere.vtk,s3.sphere.vtk \
      -p s1.inflated.curv.txt,s2.inflated.curv.txt,s3.inflated.curv.txt \
      -o s1.sphere.reg0.vtk,s2.sphere.reg0.vtk,s3.sphere.reg0.vtk \
      --icosahedron 4
```
The multi-resolution approach is quite straightforward. We can feed the registration results to the next step by increasing the sampling level `--icosahedron 5`:
```
$ HSD \
      -s s1.sphere.reg0.vtk,s2.sphere.reg0.vtk,s3.sphere.reg0.vtk \
      -p s1.sulc.txt,s2.sulc.txt,s3.sulc.txt \
      -o s1.sphere.reg1.vtk,s2.sphere.reg1.vtk,s3.sphere.reg1.vtk \
      --icosahedron 5
```
Let's use the same *sulc* features but higher resolution:
```
$ HSD \
      -s s1.sphere.reg1.vtk,s2.sphere.reg1.vtk,s3.reg1.sphere.vtk \
      -p s1.sulc.txt,s2.sulc.txt,s3.sulc.txt \
      -o s1.sphere.reg2.vtk,s2.sphere.reg2.vtk,s3.sphere.reg2.vtk \
      --icosahedron 6
```
Finally, we coregister all surfaces together using dense features:
```
$ HSD \
      -s s1.sphere.reg2.vtk,s2.sphere.reg2.vtk,s3.sphere.reg2.vtk \
      -p s1.curv.txt,s2.curv.txt,s3.curv.txt \
      -o s1.sphere.reg.vtk,s2.sphere.reg.vtk,s3.sphere.reg.vtk \
      --icosahedron 7
```
>**Note**: You can also create and use spherical harmonics coefficients for each resolution `s1.coff.txt`, `s2.coff.txt`, `s3.coff.txt` with --writecoeff and -c options rather than create and feed deformed spheres (-s and -o). This will save storage and time for file writing.

### Pairwise registration
In case of pairwise registration, one of the subjects can be regarded as a template. 
The following command line set the first subject as a template and other subjects are registered to this.
```
$ HSD --fixedSubjects <index #: 0|1>
```
Set 0 if the first subject serves as a template; 1 otherwise.
This tool also supports template (prior) variance, if any, for the pairwise registration.
```
$ HSD --tmpVar temp_var.txt
```
See more options:
```
$ HSD -h
```
In Docker, you need a sudo acces. To run, type:
```
$ docker run \
         -v <LOCAL_INPUT_PATH>:/INPUT/ \
         -v <LOCAL_OUTPUT_PATH>:/OUTPUT/ \
         --rm ilwoolyu/cmorph:<version> \
         HSD \
             -s /INPUT/s1.sphere.vtk,/INPUT/s2.sphere.vtk,/INPUT/s3.sphere.vtk \
             -p /INPUT/s1.curv.txt,/INPUT/s2.curv.txt,/INPUT/s3.curv.txt \
             -o /OUTPUT/s1.sphere.reg.vtk,/OUTPUT/s2.sphere.reg.vtk,/OUTPUT/s3.sphere.reg.vtk
```

To support cublas (GPU linear solver), type:

```
$ docker run \
         --gpus all \
         -v <LOCAL_INPUT_PATH>:/INPUT/ \
         -v <LOCAL_OUTPUT_PATH>:/OUTPUT/ \
         --rm ilwoolyu/cmorph:<version> \
         HSD-cuda \
             -s /INPUT/s1.sphere.vtk,/INPUT/s2.sphere.vtk,/INPUT/s3.sphere.vtk \
             -p /INPUT/s1.curv.txt,/INPUT/s2.curv.txt,/INPUT/s3.curv.txt \
             -o /OUTPUT/s1.sphere.reg.vtk,/OUTPUT/s2.sphere.reg.vtk,/OUTPUT/s3.sphere.reg.vtk
```

*Docker (> v19.03) supports native GPU devices. The use of NVIDIA Docker is deprecated. Please see [link](https://github.com/NVIDIA/nvidia-docker) for details.*

To utilize cublas, you need to install <a href="https://github.com/NVIDIA/nvidia-docker">NVIDIA Container Runtime for Docker</a> (deprecated).
```
$ nvidia-docker run \
         -v <LOCAL_INPUT_PATH>:/INPUT/ \
         -v <LOCAL_OUTPUT_PATH>:/OUTPUT/ \
         --rm ilwoolyu/cmorph:<version> \
         HSD-cuda \
             -s /INPUT/s1.sphere.vtk,/INPUT/s2.sphere.vtk,/INPUT/s3.sphere.vtk \
             -p /INPUT/s1.curv.txt,/INPUT/s2.curv.txt,/INPUT/s3.curv.txt \
             -o /OUTPUT/s1.sphere.reg.vtk,/OUTPUT/s2.sphere.reg.vtk,/OUTPUT/s3.sphere.reg.vtk
```
Please refer to our papers [[1](#ref1),[2](#ref2)] for technical details (theory, parameters, methodological validation, etc.).

## Requirements for build
<a href="https://github.com/ilwoolyu/MeshLib">MeshLib (general mesh processing)</a><br />
<a href="https://github.com/Slicer/SlicerExecutionModel">SlicerExecutionModel (CLI)</a>

## References
<ol>
<li><a id="ref1"></a>Lyu, I., Kang, H., Woodward, N., Styner, M., Landman, B., <a href="https://doi.org/10.1016/j.media.2019.06.013">Hierarchical Spherical Deformation for Cortical Surface Registration</a>, <i>Medical Image Analysis</i>, 57, 72-88, 2019</li>
<li><a id="ref2"></a>Lyu, I., Styner, M., Landman, B., <a href="https://doi.org/10.1007/978-3-030-00928-1_96">Hierarchical Spherical Deformation for Shape Correspondence</a>, <i>Medical Image Computing and Computer Assisted Intervention (MICCAI) 2018</i>, LNCS11070, 853-861, 2018</li>

