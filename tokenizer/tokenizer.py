import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o')

print("Vocab Size", encoder.n_vocab) # 2,00,019 (200K)

text = "My Name is Chetan Rakheja"
tokens = encoder.encode(text)

print("Tokens:", tokens)

decoded = encoder.decode(tokens)
print("Decoded:", decoded)