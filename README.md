# Neptune Classification

We have Inspection Reports (IRs) that are documents that explain the Human Errors in Nuclear Power Plants. <br>
Every sentence in a document can be labelled by the type of Human Error. <br>

The types of Human Errors are: <br>
T: Team Cognition <br>
P: Procedural <br>
O: Organizational <br>
D: Design <br>
H: Human <br>

The sentences in IR documents need to be classified into the type of these Human Errors.

We have 3 coders: Sally, Trixy and Frenard. They have labelled the sentences of various Instruction Reports (IRs).
For the sentences which do not belong to any of the type of Human Error, were labelled as <b> U: useless </b>, by them. For the sentences which were labelled by them as one of the type of Human Error, will be referred as <b> useful </b>

## Dataset
The entire text in an IR file is split by periods(.) into sentences.
Every IR .txt file is converted into .csv file with columns: text, line, start_pos, end_pos, file, label.
Every row specifies a sentence of IR, line no. in which that sentence lies, position where the sentence
starts, position where the sentence ends, filename and label assigned by coders, respectively.

Because of 3 different coders we have considered 9 'Coder Types'. These 'Coder Types' basically define different types of Dataset. The 'Dataset Types' depends upon 'Coder Types':
1. Sally - for all <b> useful </b> sentences labelled by sally
2. Trixy - for all <b> useful </b> sentences labelled by Trixy
3. Frenard - for all <b> useful </b> sentences labelled by Frenard
4. Sally_Trixy_Agree - for all <b> useful </b> sentences labelled by Sally and Trixy where they gave same labels
5. Sally_Frenard_Agree - for all <b> useful </b> sentences labelled by Sally and Trixy where they gave same labels
6. Sally_Trixy_DisAgree - for all <b> useful </b> sentences labelled by Sally where Sally and Trixy gave different labels
7. Trixy_Sally_DisAgree - for all <b> useful </b> sentences labelled by Trixy where Sally and Trixy gave different labels
8. Sally_Frenard_DisAgree - for all <b> useful </b> sentences labelled by Sally where Sally and Frenard gave different labels
9. Frenard_Sally_DisAgree - for all <b> useful </b> sentences labelled by Frenard where Sally and Frenard gave different labels<br>

## Prerequisites
- <a href="https://realpython.com/installing-python/"> Python 3.x </a> 
- <a href="https://pip.pypa.io/en/stable/installing/"> Pip3 </a>
- <a href="https://jupyter.org/install"> Jupyter Notebook </a>
- <a href="https://colab.research.google.com/notebooks/intro.ipynb#recent=true"> Google Colab </a>

## Project Structure

### 1. InterCoderReliability
Kappa and Alpha Scores are used as a metric. These metrics are used to calculate the Inter-coder Reliability between 
Sally and Trixy and between Sally and Frenard. The metrics are used to compare which pair of coders has better 
Inter-coder Reliability.

### 2. ClassificationModel_GloVe
This classification model includes: GloVe + CNN + LSTM.

### 3. ClassificationModel_GoogleW2V
This classification model includes: pre-trained Google Word2Vec + CNN + LSTM

### 4. ClassificationModel_TrainedW2V
This classification model includes: Training self Word2Vec + CNN + LSTM

### 5. ClassificationModel_BERT
This classification model implements BERT using tensorflow 1.x.

### 6. ClassificationModel_ULMFit
This classification model implements ULMFit using FastAI. 

### 7. Classification_Results_Compilations
All the classification results of the above models are compiled and compared

## Classification
We have implemented different Classification Models and have compared their results for different 'Dataset Types'. <br>
All the above Classification Models are used in:
### 1. Binary Classification:
This is done to observe how accurate a model is in predicting if a sentence is <b> useless </b> or <b> useful </b>. It uses the Dataset which contains all the sentences labelled by Sally. Equal no. of <b> useless </b> or <b> useful </b> sentences are taken in this Dataset. 
### 2. Multi-class Classification: 
This is done to observe how accurate a model is in predicting the type of Human Error of a sentence. This is done on all the 'Dataset Types' for the 'Coder Types' mentioned above. 
