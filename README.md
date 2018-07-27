## Code used for blog post [Achieving Human-level text prediction](https://benkrause/github.io/human-level-text-prediction/post.html). 

Code is a mess, but this at least should make it possible to replicate what I did.

#### Requirements: python 3, chainer, cupy

#### Instructions for use:  

1. Run the following unix command to recover the original model file

`cat model_part.aa model_part.ab > model`

2. download the text8 dataset from [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip), put the unzipped text8 file in the project directory

3. Run the main file simply with:

`python main.py`

Takes an hour or so to run, spits out the entropy of the 75 characters of text from the book "Jefferson the Virginian" to allow for direct comparison to human prediction [from this classic paper]("https://pdfs.semanticscholar.org/130b/1c7786328bf8f4ebea56e6d2f1cb992404ab.pdf"). Default settings should give a cross-entropy of 1.31 bits/character.