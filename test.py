# Full Disclosure, this is some code snippets I ran to see how to use models from Hugging Face. This has nothing
# to do with the main program that gets Harry Potter Characters and puts them in a compressed manner

from transformers import pipeline
import pprint

# Performs Text Classification
classifier = pipeline("sentiment-analysis")
results = classifier(["Harry is a fan of jazz"])
# Individual Results to say Positive or Negative
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# Labels are provided
classifier = pipeline("zero-shot-classification")
results = classifier(
    "Harry is really good at Jazz",
    candidate_labels=["Music", "Jazz"],
    template="The music being discussed is about {}",
)
pprint.pprint(results)

# Will be different upon each execution
classifier = pipeline("text-generation")
results = classifier(
    "We will discuss the finite opportunity for this star academy. The programs available"
)
pprint.pprint(results)

classifier = pipeline("fill-mask") # Pretrain method of BERT
results = classifier(
    "Jimmy had a chance yes he really did when he was <mask>", top_k=2
)
pprint.pprint(results)

classifier = pipeline(
    "text-generation",
    model="distilgpt2"  # Lightweight model
)
results = classifier(
    "In this course, we will go over the fundamentals of",
    max_length=30,  # Max number of words to be generated
    num_return_sequences=2  # Num of generated text to return
)
pprint.pprint(results)

classifier = pipeline(
    "ner",
    grouped_entities=True
)
results = classifier(
    "My name is Ryan Llamas. I have an interest in Natural Language Processing and Information Theory."
)
pprint.pprint(results)
