# Analysis-of-ConceptNet-Word-Embeddings
To analyse the ConceptNet Word Embeddings for Pictionary game purpose

# Instructions for running the code
- Clone the repository
- Download this PreTrained word-embeddings:
  - Download link: https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz
- Extract the above and place the file in it named "numberbatch-en-17.06.txt" in the folder in Analysis-of-ConceptNet-Word-Embeddings/data/
- Run command(the code is in python-3):
  - cd code/
  - python tensorboard_visualization.py
  - tensorboard --logdir model_dir -> Projector

### Few Resources that I used for concept-Net's word-embeddings 'NumberBatch'
- Github Repo: https://github.com/commonsense/conceptnet-numberbatch
- PreTrained word-embeddings Download link: https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz

### Link used for implementing Tensorboard visualization of this word embeddings
- https://stackoverflow.com/questions/50492676/visualize-gensim-word2vec-embeddings-in-tensorboard-projector
