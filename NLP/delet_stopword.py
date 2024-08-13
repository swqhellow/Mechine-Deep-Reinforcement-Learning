import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 示例文本
text = "This is a sample sentence, showing off the stop words filtration."

# 令牌化文本指的是将文本字符串分解成更小的单元，
# 这些单元通常被称为"令牌"（tokens）。令牌可以是单词、短语、符号或其他有意义的字符序列。
tokens = word_tokenize(text)

# 获取英语的停用词列表
stop_words = set(stopwords.words('english'))

# 去除停用词
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("Original Tokens:", tokens)
print("Filtered Tokens:", filtered_tokens)
