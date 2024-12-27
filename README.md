# Humanoid Navigation using Control Barrier Functions

## CREATE Virtual Environment and install Libraries

- Enter the following commands:
    1. ```python3 -m venv venv```
    2. ```source .venv/bin/activate```
    3. ```pip install -r requirements.txt```

## Install the Package

- Enter this command: ```pip install -e .```

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
