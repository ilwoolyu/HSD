# Hierarchical Spherical Deformation for Cortical Surface Registration

## Description
We present hierarchical spherical deformation for group-wise shape correspondence to address template selection bias and to minimize registration distortion. In this work, we aim at a continuous and smooth deformation field to guide accurate cortical surface registration. In conventional spherical registration methods, global rigid alignment and local deformation are independently preformed. Motivated by the composition of precession and intrinsic rotation, we simultaneously optimize global rigid rotation and non-rigid local deformation by utilizing spherical harmonics interpolation of local composite rotations in a single framework. To this end, we indirectly encode local displacements by such local composite rotations as functions of spherical locations. Furthermore, we introduce an additional regularization term to the spherical deformation, which maximizes its rigidity while reducing registration distortion. To improve surface registration performance, we employ the second order approximation of the energy function that enables fast convergence of the optimization. In the experiments, we show an improved shape correspondence with high accuracy in cortical surface parcellation and significantly low registration distortion in surface area and edge length.

##
### Environment
* Parallel processing: this tool supports OpenMP and will perform the best efficiency with CUDA.
* Memory: it requires about 0.5Gb per subject.
##
### Usage
#### Input
* surface file (.vtk): triangular sphere mesh
* feature file (.txt): scalar map - scalar value per line (corresponding to vertex)

#### Output
* surface file (.vtk): triangular deformed sphere mesh

The following command line will generate "s?.sphere.reg.vtk":
```
HSD -i s1.sphere.vtk,s2.sphere.vtk,s3.sphere.vtk -o s1.sphere.reg.vtk,s2.sphere.reg.vtk,s3.sphere.reg.vtk -p s1.curv.txt,s2.curv.txt,s3.curv.txt
```
To change the degree of spherical harmonics:
```
HSD -d #
```
In case of pairwise registration, one of the subjects can be regarded as a template. The following command line set the first subject as a template and other subjects are registered to this.
```
HSD --fixedSubjects 0
```
The tool supports template (prior) variance for the pairwise registration.
```
HSD --tmpVar temp_var.txt
```
See more options:
```
HSD -h
```
Please refer to our papers [1] for technical details (theory, parameters, methodological validation, etc.).
##
### Requirements
<a href="https://github.com/ilwoolyu/MeshLib">MeshLib (general mesh processing)</a>
<a href="https://github.com/Slicer/SlicerExecutionModel">SlicerExecutionModel (CLI)</a>

### References
<ol>
<li>Lyu, I., Styner, M., Landman, B., <a href="https://doi.org/10.1007/978-3-030-00928-1_96">Hierarchical Spherical Deformation for Shape Correspondence</a>, <i>Medical Image Computing and Computer Assisted Intervention (MICCAI) 2018</i>, LNCS11070, 853-861, 2018.</li>

