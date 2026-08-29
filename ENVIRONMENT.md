# Software environments

The repository preserves figure-level analysis scripts rather than one unified
software package. Run commands from the repository root. Historical Milgram
data and result paths remain in the scripts and must exist for the corresponding
analysis.

## Figure 1

Image synthesis uses the TensorFlow 1 environment recorded in
`figures/figure_01_image_synthesis/environment.yml`: Python 3.7, TensorFlow
1.14, NumPy, Pillow, ImageIO, tqdm, and IPython. The 5,000-set NSD control also
uses Matplotlib, pickle5, scikit-image, and OpenCV. The GPU entry point requires
a TensorFlow-1-compatible CUDA/cuDNN installation.

## Figures 3--6

The localization and overlap analyses use Python 3 with NumPy, SciPy, pandas,
Nibabel, Nilearn, Matplotlib, scikit-learn, Pillow, ImageIO, tqdm, and pickle5.
Searchlight workers additionally require BrainIAK, mpi4py, and an MPI runtime.
Surface rendering uses the plotting dependencies imported by the relevant
figure script.

Group inference and image operations call FSL command-line tools, including
`randomise`, `fslmerge`, and `fslmaths`. Cluster submission scripts require
Slurm (`sbatch`, `srun`, and job arrays). Figure 2 is a schematic and has no
standalone runtime environment.

## Supplementary figures

The supplementary analyses use the environments of their corresponding main
analyses. Figures S3, S8, S9, S13, S15, and S18 use the Python analysis stack
described above. Figures S6, S7, and S10--S17 also require the Nibabel,
Nilearn, FSL, BrainIAK, MPI, and surface-rendering tools used by Figures 3--6.
Figure S4 and the optimization in Figure S5 require the TensorFlow 1
environment used for Figure 1.

The scripts were developed in historical cluster environments and are not
presented as a newly version-pinned cross-platform package. Before a full rerun,
first execute the smallest available worker or test mode for the target figure
and verify its expected output files.
