### Create a Virtual Environnment

From the following command from the root folder, `artifical-intelligence`:

```shell
python -m venv ./venv
```

Alternatively, you can create a virtual environment using VS Code by opening a `.ipynb` file, clicking `Select Kernel`, and selecting options to create a new virtual enviornment. Supply `requirements.txt` as a dependency. 

---
### Install Dependencies

```shell
pip install -r requirements.txt
```

---
### Install the module locally from the root folder

```shell
pip install -e .
```

Then run an entry point, i.e.

```shell
play
```