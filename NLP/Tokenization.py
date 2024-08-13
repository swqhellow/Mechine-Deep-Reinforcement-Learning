# text = "Hello, world! This is tokenization."
# tokens = text.split('o',-1)
# print(tokens)
# Output: ['Hello,', 'world!', 'This', 'is', 'tokenization.']

# import nltk
# from nltk.util import ngrams
# with open (r'Learning\NLP\abstract.txt','r') as file:
#     for text in file:
#         print(text)
#         bigrams = list(ngrams(text.split(), 3))
#         print(bigrams)
# file.close()
import jieba
with open(r'Learning\NLP\chinese.txt','r',encoding='utf-8') as file:
    text = file.read()
    tokens = jieba.lcut(text)
    print(tokens)
# Output: ['我', '喜欢', '自然语言处理']
