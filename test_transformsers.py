import torch
from transformers import *
# 加载词典 pre-trained model tokenizer (vocabulary)
# 加载词典 pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(r'D:\PycharmProjects\bert\ERNIE_1.0_max-len-512-pytorch\ERNIE_1.0_max-len-512-pytorch',do_lower_case=True)
# tokenizer = BertTokenizer.from_pretrained(r'D:\PycharmProjects\bert\ERNIE_1.0_max-len-512-pytorch\ERNIE_1.0_max-len-512-pytorch')

# Tokenized input
text = "这是百度的模型"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet',r'##eer', '[SEP]']

# 将 token 转为 vocabulary 索引
indexed_tokens = tokenizer.convert_tokens_to_ids( tokenized_text)
# 定义句子 A、B 索引
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 将 inputs 转为 PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# 加载模型 pre-trained model (weights)
model = BertModel.from_pretrained(r'D:\PycharmProjects\bert\ERNIE_1.0_max-len-512-pytorch\ERNIE_1.0_max-len-512-pytorch')
model.eval()

# GPU & put everything on cuda
#tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = torch.tensor([segments_ids])


#segments_tensors = segments_tensors.to('cuda')
#model.to('cuda')

# 得到每一层的 hidden states
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)
# 模型 bert-base-uncased 有12层，所以 hidden states 也有12层
print(encoded_layers[0][0].size())
