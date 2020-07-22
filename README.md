# docalign

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