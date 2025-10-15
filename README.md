The script builds a very simple retrieval‑based chatbot over the current Wikipedia page 
Main steps:
Fetch & parse content: Downloads the Wikipedia HTML (with SSL verification disabled) and uses Goose to extract cleaned article text.
Sentence splitting: Uses NLTK sentence tokenizer to create a list of original sentences.
Preprocessing:
Lowercases text.
Uses spaCy to tokenize and filters out stopwords, single‑character tokens, numbers, punctuation, spaces.
Joins remaining tokens back into a cleaned string.
Answer generation (answer function):
Preprocesses every original sentence on each user query (no caching).
Adds the preprocessed user query to the list.
Builds a TF‑IDF matrix over all cleaned sentences (including the query).
Computes cosine similarity between the query vector (last row) and all sentences.
Selects the most similar prior sentence; if similarity > threshold (default 0.4) returns the original (unprocessed) sentence, else a fallback message.
Greeting handling: Simple rule-based check for greeting words and returns a random greeting response.
Interactive loop: Repeatedly prompts the user until they type exit/quit/stop.
