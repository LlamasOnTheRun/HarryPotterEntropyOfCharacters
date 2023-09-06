from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("No wait please don't tokenize me!")
tokens = tokenizer.tokenize("No wait please don't tokenize me, but this time with sub-words!")
print(inputs)
print(inputs["input_ids"])
print(tokens)
print(tokenizer.decode(inputs["input_ids"]))  # Will show the decoded tokens

tokenizer = AutoTokenizer.from_pretrained("albert-base-v1")
tokens = tokenizer.tokenize("Oh man anyone but albert!")
print(tokens)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
final_inputs = tokenizer.prepare_for_model(input_ids)  # Will add special tokens
print(final_inputs["input_ids"])

print("##################### LEARNING TOKENIZATION WITH BATCHING INPUTS #####################")

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sentences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I love this!",
]
tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
print(ids[0])
print(ids[1])
