# Political Polarization - Rumour Sequences
Rumour classification project in within the scope of the research lab 'Political Polarization' at University of Koblenz.


## How to run it?

1. Clone the repository and get into it
```bash
    git clone https://github.com/NicoZettler/polpol-rumour-sequences.git
    cd polpol-rumour-sequences
```


2. Create "resources" folder and download/place test data inside (see "Data" section below)
```bash
    mkdir resources
```

3. Create and activate virtual environment

```bash
    conda create -n polenv python
    conda activate polenv
```

4. Install dependencies:

```bash
    pip install -r requirements.txt
```

5. Install Tokenizer based on nltk, but with a few more features added
```bash
    pip install git+https://github.com/erikavaris/tokenizer.git
```

6. Run the application

```bashh
    python TD_CLEARumor.py
```

## Data

Download files from the [RumourEval-2019 competition page](https://competitions.codalab.org/competitions/19938)
⋅⋅* rumoureval-2019-training-data.zip
⋅⋅* rumoureval-2019-test-data.zip
⋅⋅* home_scorer_macro.py
and put them inside the resources folder created in step 2.
