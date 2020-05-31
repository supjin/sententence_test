import logging

import scipy
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
from datetime import datetime
from sentence_transformers import models, losses

#### Just some code to print debug information to stdout
from CSTSataReader import CSTSBenchmarkDataReader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = 'bert-base-nli-mean-tokens'
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# sts_reader = STSBenchmarkDataReader('./datasets/stsbenchmark', normalize_scores=True)
sts_reader=CSTSBenchmarkDataReader('./ChineseSTS/dataset', normalize_scores=True)
# Load a pre-trained sentence transformer model

word_embedding_model = models.Transformer(r'D:\PycharmProjects\bert\ERNIE_1.0_max-len-512-pytorch\ERNIE_1.0_max-len-512-pytorch')
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# model = SentenceTransformer(r'output/training_stsbenchmark_continue_training-bert-base-nli-mean-tokens-2020-05-30_23-18-49')

it = sts_reader.get_examples("sts-test.csv")
print("++++++++++++++++")
text1=[]
text2=[]
num =0
for item in it:
    if num>30:
        break
    print(item.texts)
    print(item.label)
    text1.append(item.texts[0])
    text2.append(item.texts[1])
    print(item.guid)
    num+=1

closest_n=3
corpus_embeddings = model.encode(text1)
query_embeddings = model.encode(text2)
for query, query_embedding in zip(text2, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("======================")
    print("Query:", query)
    print("Top 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(text1[idx].strip(), "(Score: %.4f)" % (1-distance))



print('_______________')
# Apply mean pooling to get one fixed sized sentence vector

test_data = SentencesDataset(examples=sts_reader.get_examples("sts-dev.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
model.evaluate(evaluator)

