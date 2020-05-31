import scipy
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bert-base-nli-mean-tokens')
# model = SentenceTransformer('bert-base-uncased')
from sentence_transformers import models, losses

# # Use BERT for mapping tokens to embeddings
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import NLIDataReader, STSBenchmarkDataReader

word_embedding_model = models.Transformer(r'D:\PycharmProjects\bert\ERNIE_1.0_max-len-512-pytorch\ERNIE_1.0_max-len-512-pytorch')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Corpus with example sentences
corpus = ['我不想吃饭了怎么办.',
          '我觉得可以先吃点儿再去看电影吧.',
          '我不太喜欢去商店吃东西.',
          '有个男人正在骑马.',
          '有一个女人正在减肥活动.',
          '有两个人正在吃东西但是我觉得他们不太想吃.',
          '我正在进行一个比较傻逼的活动.',
          '我觉得医疗美容行业可以继续进行.',
          '双眼皮多少钱做一个.']

corpus_embeddings = model.encode(corpus)
queries = ['今天想去吃东西.', '前几天我看到有人在整容，我也想去.', 'A cheetah chases prey on across a field.']
query_embeddings = model.encode(queries)
closest_n = 5


for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))


# # 加载词典 pre-trained model tokenizer (vocabulary)
# # 加载词典 pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained(r'C:\Users\66419\Downloads\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12')
# tokenizer = BertTokenizer.from_pretrained(r'D:\PycharmProjects\bert\ERNIE_1.0_max-len-512-pytorch\ERNIE_1.0_max-len-512-pytorch')
#
# # Tokenized input
# text = "这是百度的ernie模型"
# tokenized_text = tokenizer.tokenize(text)
#
# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# print(tokenized_text)
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet',r'##eer', '[SEP]']
#
# # 将 token 转为 vocabulary 索引
# indexed_tokens = tokenizer.convert_tokens_to_ids( tokenized_text)
# # 定义句子 A、B 索引
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
#
# # 将 inputs 转为 PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
#
# # 加载模型 pre-trained model (weights)
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()
#
# # GPU & put everything on cuda
# #tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = torch.tensor([segments_ids])
#
#
# #segments_tensors = segments_tensors.to('cuda')
# #model.to('cuda')
#
# # 得到每一层的 hidden states
# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, segments_tensors)
# # 模型 bert-base-uncased 有12层，所以 hidden states 也有12层
# assert len(encoded_layers) == 12
# print(encoded_layers[0][0].size())
#
