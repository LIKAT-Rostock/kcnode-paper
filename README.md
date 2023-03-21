# Kinetic-constrained neural ODE (KCNODE)

The structure of the project:
----------------
* **data_generator.py** - Functions for generating the data for the numerical experiment
* **KCNODE.py** - Classes and methods for building neural ODE models
---------------
* **Baseline model.ipynb** - Example using the baseline model
* **KCNODE FT.ipynb.ipynb** - Example using the KCNODE model for the process of CO<sub>2</sub>  hydrogenation to hydrocarbons via FT (real data)
* **KCNODE methanation.ipynb** - Example using the KCNODE model for the process of CO<sub>2</sub>  hydrogenation to methane (numerical experiment)
* **Training.ipynb** - Example demonstrating the training of the neural ODE models
---------------
* **\trained_models** - Directory with a trained model of neural ODE
    * **baseline.pickle** - The baseline neural ODE model.
    * **KCNODE_methanation.pickle** - The KCNODE model for CO<sub>2</sub> hydrogenation to CH<sub>4</sub> 
    * **KCNODE_FT.pickle** - The KCNODE model for CO<sub>2</sub> hydrogenation to hydrocarbons via FT

----------------
* **Python version**:3.9.12
----------------
* **Dependencies:**

    * torch:1.12.0
    * torchdyn:1.0.3
    * numpy:1.23.0
    * scipy:1.8.1
    * pandas:1.4.3
    * matplotlib:3.5.2    
    * tqdm:4.64.0
    * doepy:0.0.1
    * notebook:6.5.3
    
Install dependencies with `pip install -r requirements.txt`
