# EMRI_identification

#### Introduction

In this pipeline, we demonstrated the identification of EMRI signals without any additional prior information on physical parameters. This pipeline includes physical waveform search, phenomenological waveform semi-coherent search, and parameter inversion process from phenomenological parameters to physical parameters. For a detailed description of the entire pipeline, please refer to the following:[Identification of Gravitational-waves from Extreme Mass Ratio Inspirals
](https://arxiv.org/abs/2310.03520)


The entire pipeline runs in a Python 3 environment and parallel processing.
This is the first version of the pipeline, mainly showcasing the core principles and code for each step of the search.

#### Illustrate

This is a document description primarily covering each step of the search process. In our pipeline, we first perform a harmonic search for physical waveforms, followed by a semi-coherent search for phenomenological waveforms, and finally, parameter inversion from phenomenological waveforms to physical waveforms.

files:
**physics_harmonic_detect**： harmonic search for physical waveforms

**phenomenological_waveform_search**: A semi-coherent search for phenomenological waveforms

**fit_para_to_physics_para**: parameter inversion from phenomenological waveforms to physical waveforms









