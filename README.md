# continuous_control_bci
An analysis of EEG data for continuous real-time BCI control.

## Installation

Install pipenv with 
`pip install pipenv`

Then create a virtual environment with
`pipenv install`

## Run instructions

Before you can run any analysis you should prepare the ICA decomposition with
`pipenv run make_ica.py`

You can run the different analysis scripts with 
`pipenv run some_script.py`

The scripts are generally separated into a `calibration` version and a `driving` version to run the analysis on the calibration or driving data. 
The data that this all corresponds is not yet publicly available, but may be made available on request.

## ICA instructions
With 20 participants in 2 contexts there are 40 sessions for which ICs need to be rejected.
The ICA decomposition can all be generated with `make_ica.py`.
Then you can inspect each decomposition with `show_ica.py`. Any ICs marked as bad during this inspection will be stored as bad.
Subsequent analyses will run with those ICs are rejected.


## Notebooks
There are several Jupyter notebooks in `/notebooks`. These are kept for posteriority and may be considered useful, 
but they are not intended to be reproduced.  

## Questions
Any questions may be directed to `ivo.de.jong@rug.nl`.