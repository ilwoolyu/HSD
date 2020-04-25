# Group-wise Registration on FreeSurfer Outputs
This implementation uses a Bash script for group-wise cortical surface registration using HSD. The tool is designed specifically for FreeSurfer outputs and supports both native HSD executable file and the Docker image (<a href="https://hub.docker.com/r/ilwoolyu/cmorph/">CMorph</a>).

## Installation
Download and grant an execute permission to `hsd_group`.
```
$ chmod a+x hsd_group
```

(Optional) Download `sphere` and set `$HSD_SPHERE_ROOT`.
```
$ export HSD_SPHERE_ROOT=PATH_TO_ICO_SPHERE
```

## Input
* FreeSurfer output directory: `INPUT_DIR`

## Output
* Output directory: `OUTPUT_DIR`
* Registered spheres: `?h.sphere.reg` (FreeSurfer format), `?h.sphere.reg.vtk` (VTK format)
* Spherical harmonics coefficients: `?h.coeff.txt`

A typical hierarchy of FreeSurfer outputs looks like
```
INPUT_DIR
├── subj_1
│   ├── label
│   ├── mri
│   ├── scripts
│   ├── stats
│   ├── surf
│   └── touch
.
.
└── subj_N
    ├── label
    ├── mri
    ├── scripts
    ├── stats
    ├── surf
    └── touch
```

## Quick command
The basic command line needs hemisphere and FreeSurfer output directory.
```
$ ./hsd_group hemi={lh|rh} PATH_TO_INPUT_DIR
```
For example, to run group-wise registration on all the outputs of the left hemisphere in `INPUT_DIR`:
```
$ ./hsd_group lh INPUT_DIR
```
The command will create sub-directory (`HSD`) for each subject in `INPUT_DIR`.
```
INPUT_DIR
├── subj_1
│   ├── HSD
│   ├── label
│   ├── mri
│   ├── scripts
│   ├── stats
│   ├── surf
│   └── touch
.
.
└── subj_N
    ├── HSD
    ├── label
    ├── mri
    ├── scripts
    ├── stats
    ├── surf
    └── touch
```
The sub-directory will contain three output files.
```
$ ls
lh.coeff.txt  lh.sphere.reg  lh.sphere.reg.vtk
```
>**Note**: The script will pull all `surf` directories in `INPUT_DIR`. Any corrupted information in `surf` (e.g., file deletion, empty directory, etc.) can lead to a failure of the processing.

## Options
* The tool supports OpenMP multi-threading. For multi-core CPU equipped machines, OpenMP can significantly accelerate the processing. To enable this function, add `--thread #` to the command.
* The tool supports the Docker image (<a href="https://hub.docker.com/r/ilwoolyu/cmorph/">CMorph</a>). To run Docker container, group membership to `docker` is required, and add `--docker tag`. Check the latest `tag` at https://hub.docker.com/r/ilwoolyu/cmorph/.
* To use pre-defined icosahedron mesh, setup environmental variable `export HSD_SPHERE_ROOT=PATH_TO_ICO_SPHERE`, which avoids computing icosahedral mesh. A set of template icosahedral meshes is provided in this release (see `sphere` folder in this repository).
* To change `OUTPUT_DIR`, add `--out` (default: `INPUT_DIR`) to the command.
* To change sub-directory name, add `--subfolder` (default: `HSD`) to the command.

## Requirements
* The script works on HSD `v1.2.6` or higher.
* The script uses a part of <a href="https://surfer.nmr.mgh.harvard.edu/">FreeSurfer</a> executable files.
  * `mris_convert` to convert surface and geometric features
  * `mris_rescale` to ensure a radius of 100
