# Humanoid Navigation using Control Barrier Functions

# Installation/Quick start
First you need to create your python environment with <code style="color : GreenYellow">conda</code> and install <code style="color : GreenYellow">pip</code> for the packages:
```
conda create -n amr_mpc
conda install pip
```
Then you need to activate your environment as follows:
```
conda activate amr_mpc
```
In order to install everything you just need to execute the following command inside the repository:
```
python -m pip install -e .
```
Or you can directly use the standard virtual environment <code style="color : GreenYellow">.venv</code> as follows:

```
python3 -m venv .venv

source .venv/bin/activate

pip install -e .
```

## CONTAINER INSTRUCTIONS

- Open **Docker** 

- Enter the folder where you have cloned the repository:
    ```cd AMRProject```

- Build the Container, with arm64 for Mac ARM Processor:

    ```docker build --platform linux/arm64 -t amr_project .```
    
- To run the Container without re-building it every time. It saves the changes instantaneously.

    ```docker run --platform linux/arm64 -it --name amr_project -v "$(pwd):/amr_prj" amr_project```

    With this command you are inside the Container and you are able to Run the files.
    
- To Re-Start the Container (when closing and opening the pc, for example):

    ```docker run -it --name amr_project```

#### Other useful commands
- Stop the Container (from an external terminal): 
    ```docker stop amr_project```

- Stop the Container (from the internal terminal):
    **ctrl + D**

- Remove the Container (from an external terminal): 
    ```docker rm amr_project```