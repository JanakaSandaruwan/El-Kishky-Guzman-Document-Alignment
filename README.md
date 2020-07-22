# docalign

This package reproduces the algorithm described by El-Kishky and Guzmán (2020)[1] to align massively multilingual document with cross-lingual Sentence-Mover’s Distance.
The code in this repository is focused on aligning English and Sinhala document pairs, but it can also be used with any other language pair with a few code level changes.
<a href="https://github.com/facebookresearch/LASER">LASER PROJECT</a> is used to get sentence embeddings as descibed in the original paper.

### Build docalign
```pip install -r requirements.txt```<br>
Install <a href="https://github.com/ysenarath/sinling">sinling</a><br>

### Embed your own documents
For source document<br>
```python embedding_creator.py example/source.json ./example/se.json en```<br><br>
For target document<br>
```python embedding_creator.py example/target.json ./example/te.json si```<br><br>

Source and target documets should be in json format as follows.<br>
[<br>
&nbsp;&nbsp;&nbsp;&nbsp;{
        "content": "doc1"
    },<br>
&nbsp;&nbsp;&nbsp;&nbsp;{
        "content": "doc2"
    }<br>
]

### Run docalign (using embeded documents)
```python main.py ./example/se.json ./example/te.json``` <br><br>

optional params :<br>
```python main.py --help```<br>
```python embedding_creator.py --help```

### References

[1] Ahmed El-Kishky and Francisco Guzman. 2020. Massively multilingual document alignment with crosslingual sentence-mover’s distance
