# EMRI_identification

#### Introduction

In this pipeline, we demonstrated the identification of EMRI signals without any additional prior information on physical parameters. This pipeline includes physical waveform search, phenomenological waveform semi-coherent search, and parameter inversion process from phenomenological parameters to physical parameters. For a detailed description of the entire pipeline, please refer to the following:[Identification of Gravitational-waves from Extreme Mass Ratio Inspirals
](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.124034)


The entire pipeline runs in a Python 3 environment and parallel processing.
This is the first version of the pipeline, mainly showcasing the core principles and code for each step of the search.

#### Illustrate

This is a document description primarily covering each step of the search process. In our pipeline, we first perform a harmonic search for physical waveforms, followed by a semi-coherent search for phenomenological waveforms, and finally, parameter inversion from phenomenological waveforms to physical waveforms.

Input file:
Simulation data: EMRI signal(Michelson response)
file: signal_1.hdf5

files:
**physics_harmonic_detect**： harmonic search for physical waveforms

Output file: template.txt ( the parameters of the matching template and the signal-to-noise ratio distribution of the main harmonic modes. )

**phenomenological_waveform_search**: A semi-coherent search for phenomenological waveforms

Output file: nested_run.sav (run nested sampling for each time segment each harmonic mode)


**fit_para_to_physics_para**: parameter inversion from phenomenological waveforms to physical waveforms

Output file: gaid.npy (Matrix of parameter grid)

**examples**

signal file: signal_1.hdf5 (examples signal)
read_result_to_figure： read_sesults_*_search.ipynb














