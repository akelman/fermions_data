# Fermions Data

This repository stores the data for the paper *Projected Entangled Pair States for Lattice Gauge Theories with Dynamical Fermions*.
arxiv: [https://arxiv.org/abs/2412.16951](https://arxiv.org/abs/2412.16951)

The script `paper_plots.py` generates all of the plots. It can be run (after installation of the required packages) using the command `python paper_plots.py`. This will save (and overwrite) the plots in the directory `generated_plots`.

The script `inspect_data.py` provides a quick way to read the data files from the command line. The command 
`python inspect_data.py --fname path/to/summary/file.pkl`
will print the contents of that file.

## Installation

To generate the plots, any recent version of python with the required packages should suffice (we used python 3.12).

Download this repo: either clone from github (e.g. `git clone git@gitlab.com:git@github.com:akelman/fermions_data.git` if using ssh) or download from Zenodo using the web interface. If needed, unzip the data (so that it is stored in the directory `/data/` relative to this readme).

Create a virtual environment (`python -m venv env_name`) and then activate it (`source env_name/bin/activate`), and install the required packages (`pip install -r requirements.txt`). 

You can now run the scripts as specified above.
When you are done handling the data, deactivate the environment, `deactivate`.