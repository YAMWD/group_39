# WDPS 2024 Group_39 Assignment

This is the repository of Group 39 assignment for course Web Data Processing Systems

## Set up
1. create a new `venv` and install some additional packages

    ```bash
    # create venv
    python3 -m venv venv
    # activate venv
    source venv/bin/activate
    # install requirements
    pip install -r requirements.txt
    ```

2. Download the model `llama-2-7b.Q4_K_M.gguf` from the Docker container in `~/models`, 
and place it in the `models` directory.

## Run the code
To run the code, set **input** and **output** file paths in `main.py`, and then run the following command:

```bash
python main.py
```
