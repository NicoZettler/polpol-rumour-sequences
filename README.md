# Political Polarization - Rumour Sequences
Rumour classification project in within the scope of the research lab 'Political Polarization' at University of Koblenz.


## How to run it?

1. Clone the repository and get into it
```bash
    git clone https://github.com/NicoZettler/polpol-rumour-sequences.git
    cd polpol-rumour-sequences
```

2. Create and activate virtual environment

```bash
    conda create -n polenv python
    conda activate polenv
```

3. Install dependencies:

```bash
    pip install -r requirements.txt
```

4. Install Tokenizer based on nltk, but with a few more features added
```bash
    pip install git+https://github.com/erikavaris/tokenizer.git
```

5. Run the application

```bashh
    python TD_CLEARumor.py
```
