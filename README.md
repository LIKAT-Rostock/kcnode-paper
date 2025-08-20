# Kinetic-constrained neural ODE (KCNODE)

> This repository supplements our Chemical Engineering Journal paper ([doi:10.1016/j.cej.2023.146869](https://doi.org/10.1016/j.cej.2023.146869)) and the earlier ChemRXIV preprint ([doi:10.26434/chemrxiv-2023-x39xt](https://doi.org/10.26434/chemrxiv-2023-x39xt)). The catalytic data is available via the link [Repo4Cat/21.11165/2d97-u7yd](https://hdl.handle.net/21.11165/4cat/2d97-u7yd).

## Contents of the repository

* **data_generator.py** - Functions for generating the data for the numerical experiment
* **KCNODE.py** - Classes and methods for building neural ODE models

---------------

* **Baseline model.ipynb** - Example using the baseline model
* **KCNODE FT.ipynb.ipynb** - Example using the KCNODE model for the process of CO<sub>2</sub>  hydrogenation to hydrocarbons via FT (real data)
* **KCNODE methanation.ipynb** - Example using the KCNODE model for the process of CO<sub>2</sub>  hydrogenation to methane (numerical experiment)
* **Training.ipynb** - Example demonstrating training of the neural ODE models

---------------

* **/trained_models** - Directory with a trained model of neural ODE
  * **baseline.pt** - The baseline neural ODE model.
  * **KCNODE_methanation.pt** - The KCNODE model for CO<sub>2</sub> hydrogenation to CH<sub>4</sub>
  * **KCNODE_FT.pt** - The KCNODE model for CO<sub>2</sub> hydrogenation to hydrocarbons via FT

## Used environment

The code was developed on Windows 10 but it should be platform independent.

* **Python version**: 3.9.12 (amd64)
* **Packages:**
  * torch:1.12.0
  * torchdyn:1.0.3
  * numpy:1.23.0
  * scipy:1.8.1
  * pandas:1.4.3
  * matplotlib:3.5.2
  * tqdm:4.64.0
  * doepy:0.0.1
  * notebook:6.5.3

Install the dependencies with `pip install -r requirements.txt`

## Authors

* Aleksandr Fedorov [![](https://info.orcid.org/wp-content/uploads/2020/12/orcid_16x16.gif)](https://orcid.org/0000-0001-6434-6623) (on GitHub [fizorg103](https://github.com/fizorg103))
* David Linke [![](https://info.orcid.org/wp-content/uploads/2020/12/orcid_16x16.gif)](https://orcid.org/0000-0002-5898-1820) (on GitHub [dalito](https://github.com/dalito))

## Acknowledgement

Financial support from German Federal Ministry of Education and Research (BMBF) through the project InnoSyn (FKZ: 03SF0616B) and from German Research Foundation (DFG) through the project "[NFDI4Cat](https://www.nfdi4cat.org) - NFDI for Catalysis-Related Sciences" (DFG project no. [441926934](https://gepris.dfg.de/gepris/projekt/441926934)) within the National Research Data Infrastructure ([NFDI](https://www.nfdi.de)) programme of the Joint Science Conference (GWK) is gratefully acknowledged.
