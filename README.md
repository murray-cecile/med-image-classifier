## Classifying abnormal masses in mammograms

### Tammy Glazer, Parth Khare, Cecile Murray

### Downloading data

1. Add desired images to the cart and download. `get-mass-case-ids.sh` will help create a comma-separated list of training and testing cases using the metadata csvs.

2. Use NBIA data retriever to convert images from .tcia to .dcm (longest step)

3. Run `sh ingest-data.sh` followed by the full file path to the images to pull .dcm files out of nested directory and sensibly rename.

### Setting up and using the virtual environment

If you use conda, you might have to `conda deactivate` before these steps.

 First, create the virtual environment:

```python -mvenv venv_name```

Then activate it:

```source ./venv_name/bin/activate```

Then install required packages based on the list in requirements.txt:

```pip3 install -r requirements.txt```

You can install additional packages as usual. To add them to the list of required packages, you can run:

``` pip3 freeze > requirements.txt```