# LoDoPaB-CT challenge utilities

This repository contains utilities for the
[LoDoPaB-CT challenge](https://lodopab.grand-challenge.org).

The functions in ``challenge_set.py`` help accessing the downloaded observation
data from which the challenge asks to compute reconstructions.

To save the reconstructions to files, use the function ``save_reconstruction()``
from ``submission.py``.
A packed zip-File of the written files can be uploaded as a submission to the
challenge website.

The challenge data is available from
[zenodo.org](https://zenodo.org/record/3874937).
After downloading, unpack the zip file and set
``challenge_set.config['data_path']`` to the directory containing the unpacked
files.
