# Full Disclosure, this is some code snippets I ran to see how to use models from Hugging Face. This has nothing
# to do with the main program that gets Harry Potter Characters and puts them in a compressed manner

from transformers import AutoModel, BertConfig, BertModel

# Each of the below is considered a checkpoint
# The AutoModel class will find the appropriate model class based on what is used in from_pretrained(...)
bert_model = AutoModel.from_pretrained("bert-base-cased")
print(type(bert_model))

gpt_model = AutoModel.from_pretrained("gpt2")
print(type(gpt_model))

bart_model = AutoModel.from_pretrained("facebook/bart-base")
print(type(bart_model))

bert_config = BertConfig.from_pretrained("bert-base-cased", num_hidden_layers=10)
print(bert_config)
bert_model = BertModel(bert_config)
# Saves the model in local directory for later used
bert_model.save_pretrained("llamas-bert-model")

