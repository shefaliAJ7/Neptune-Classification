# Classification

We have Instructional Reports (IRs) that are documents that explains the Human Errors in Nuclear Power Plants.
Every sentence in a document can be labelled by the type of Human Error.

The types of Human Errors are:
T: Team Cognition
P: Procedural
O: Organizational
D: Design
H: Human

The sentences in IR documents needs to be classified into the type of these Human Errors.

We have 3 coders: Sally, Trixy and Frenard. They have labelled the sentences of various Instruction Reports (IRs).
For the sentences, which do not belong to any of the type of Human Error, were labelled as U: uselss, by them.

Dataset:
The entire text in an IR file is split by periods(.) into sentences.
Every IR .txt file is converted into .csv file with columns: text, line, start_pos, end_pos, file, label.
Every row specifies a sentence of IR, line no. in which that sentence lies, position where the sentence
starts, position where the sentence ends, filename and label assigned by coders, respectively.

We have implemented different Classification Models and have compared their results for different Coders.

Prerequisites:
- Python 3.x
- Pip
- Jupyter Notebook
- Google Colab

Project Structure:

1. InterCoderReliability
Kappa and Alpha Scores are used as a metric. These metrics are used to calculate the Inter-coder Reliability between 
Sally and Trixy and between Sally and Frenard. The metrics are used to compare which pair of coders has better 
Inter-coder Reliability.

2. ClassificationModel_GloVe
This classification model includes: GloVe + CNN + LSTM.

3. ClassificationModel_GoogleW2V
This classification model includes: pre-trained Google Word2Vec + CNN + LSTM

4. ClassificationModel_TrainedW2V
This classification model includes: Training self Word2Vec + CNN + LSTM

5. ClassificationModel_BERT
This classification model implements BERT using tensorflow 1.x.

6. ClassificationModel_ULMFit
This classification model implements ULMFit using FastAI. 

7. Classification_Results_Compilations
All the classification results of the above models are compiled and compared
