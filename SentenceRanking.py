"""
This project aims to perform extractive summaries on articles
"""

import os
os.chdir('/Users/jaoming/Documents/Active Projects/Text Summarisation')

import numpy as np                                      # array manipulation
import spacy                                            # nlp module
from spacy.lang.en.stop_words import STOP_WORDS
import re                                               # text cleaning
import networkx as nx                                   # matrix manipulation
from sklearn.metrics.pairwise import cosine_similarity  # similarity computation

# Standard NLP parser
nlp = spacy.load('en_core_web_sm')

# Custom NLP parser (not used for this case)
def set_custom_boundaries(doc):
       # adds support to parse quotes as sentences
       for token in doc[:-1]:
              if token.text == "'":
                     doc[token.i+1].is_sent_start = False
              elif token.text == ".":
                     doc[token.i+1].is_sent_start = True
       return doc

custom_nlp = spacy.load('en_core_web_sm')
custom_nlp.add_pipe(set_custom_boundaries, before = 'parser')

# Extracting pre-trained word-vectors
word_embeddings = {} # these are vectors that represent words
g_file = open('glove.6B.100d.txt', encoding = 'utf-8')
for line in g_file:
       values = line.split()
       word = values[0]
       coefs = np.asarray(values[1:], dtype = 'float32')
       word_embeddings[word] = coefs
g_file.close()

# Main Class
class ExtractSummary():
       """
       This class aims to create extractive summaries of articles
       """
       def extracting_sentences(self, doc):
              """
              Function:     Extracts the sentences from the document and cleans them up before returning them

              Inputs:       Document

              Returns:      Cleaned individual sentences
              """
              dirty_sentences = [i.strip() for i in doc.sents]
              clean_sentences = []
              for dirt in dirty_sentences:
                     # un-capitalise the words in the sentence and remove stop words
                     clean = " ".join([i.lower_ for i in dirt if i not in STOP_WORDS])
                     
                     # removing anything that's not a word
                     clean = re.sub("[^a-zA-Z0-9-]", " ", clean).strip()
                     clean = re.sub("\s+", " ", clean)
                     # clean = re.sub("\s-\s", "-", clean)
                     clean_sentences.append(clean)
              return clean_sentences, dirty_sentences

       def convert_to_sentence_vectors(self, sentences):
              """
              Function:     Converts sentences into a vector form by summing up the word vectors in that sentence and dividing it over the number of words in that sentence

              Inputs:       Sentences

              Returns:      A list of sentence vectors
              """
              sentence_vectors = []
              for sentence in sentences:
                     sent_len = len(sentence.split())
                     if sent_len != 0:
                            vect = sum([word_embeddings.get(word, np.zeros((100, ))) for word in sentence.split()])
                            vect = vect / (sent_len + 0.001)
                     else:
                            vect = np.zeros((100, ))
                     sentence_vectors.append(vect)
              return sentence_vectors

       def matrix_preparation(self, sent_vects, num_sentences):
              """
              Function:     Prepares the sentence ranking matrix for optimisation

              Inputs:       Sentence vectors

              Returns:      Matrix
              """
              similarity_matrix = np.zeros([num_sentences, num_sentences])

              for i in range(num_sentences):
                     for j in range(num_sentences):
                            if i != j:
                                   similarity_matrix[i][j] = cosine_similarity(sent_vects[i].reshape(1, 100), sent_vects[j].reshape(1, 100))[0, 0]
              return similarity_matrix

       def summarise(self, article):
              """
              Function:     Main function of the class that executes the summarising

              Inputs:       The article in text form

              Returns:      Extracted Summary (Top 5 sentences)
              """
              # Parsing the article into a NLP instance
              doc = nlp(article) # choose which parser you want to use

              # Extracting clean individual sentences
              sentences, original_sentences = self.extracting_sentences(doc)
              n_sentences = len(sentences)

              # Converting sentences into vectors
              sentence_vectors = self.convert_to_sentence_vectors(sentences)

              # Creating the similarity matrix for the sentences
              sim_mat = self.matrix_preparation(sentence_vectors, n_sentences)

              # Applying the page-ranking algorithm to the matrix
              nx_graph = nx.from_numpy_array(sim_mat)
              scores = nx.pagerank(nx_graph)

              # Extracting the sentence and scores of the Top 5
              ranked_sentences = sorted([(scores[i], s) for i, s in enumerate(original_sentences)], reverse = True)

              return ranked_sentences

# Example texts
text = """President Donald Trump has threatened to close down social-media platforms that he argues censor conservative voices after Twitter on Tuesday tagged some of his messages with a fact-check warning. 'Republicans feel that Social Media Platforms totally silence conservatives voices,' Trump tweeted Wednesday. 'We will strongly regulate, or close them down, before we can ever allow this to happen. We saw what they attempted to do, and failed, in 2016.' Twitter had long been criticized for allowing the president to spread conspiracy theories and smears against opponents despite its policies against the promotion of disinformation. It recently came under increasing calls for it to take action against Trump after he spent weeks promoting a baseless conspiracy theory alleging that the MSNBC cohost Joe Scarborough was involved in the death of a staffer, Lori Klausutis, while he was serving as a US congressman. Twitter has declined to take action against the president for the messages about Scarborough, but on Tuesday for the first time it put a fact-check tag on some of Trump’s tweets. The president wrote two tweets claiming 'There is NO WAY (ZERO!) that Mail-In Ballots will be anything less than substantially fraudulent.' Twitter tagged each of the two messages with a blue exclamation mark and warning message, linking to articles in The Washington Post, CNN, and other outlets that debunk the president’s assertion. Trump doubled down on his voter-fraud claims in a follow-up tweet Wednesday. 'We can’t let a more sophisticated version of that happen again,' Trump wrote. 'Just like we can’t let large scale Mail-In Ballots take root in our Country. It would be a free for all on cheating, forgery and the theft of Ballots. Whoever cheated the most would win. Likewise, Social Media. Clean up your act, NOW!!!!' Trump has long accused social-media companies of bias toward conservatives. In June 2019 he invited several far-right provocateurs and conspiracy theorists, some of whom had had hate speech removed by social-media platforms, to the White House for a social-media summit. He has also credited being able to communicate on Twitter as a key factor in his election to the White House, remarking that it allows him to communicate with voters directly, unfiltered by media organizations he accuses of partisan bias.
"""

cat = """
Domestic cats, no matter their breed, are all members of one species. Relationship with Humans. Felis catus has had a very long relationship with humans. Ancient Egyptians may have first domesticated cats as early as 4,000 years ago. Plentiful rodents probably drew wild felines to human communities. The cats' skill in killing them may have first earned the affectionate attention of humans. Early Egyptians worshipped a cat goddess and even mummified their beloved pets for their journey to the next world—accompanied by mummified mice! Cultures around the world later adopted cats as their own companions. Hunting Abilities. Like their wild relatives, domestic cats are natural hunters able to stalk prey and pounce with sharp claws and teeth. They are particularly effective at night, when their light-reflecting eyes allow them to see better than much of their prey. Cats also enjoy acute hearing. All cats are nimble and agile, and their long tails aid their outstanding balance. Communication. Cats communicate by marking trees, fence posts, or furniture with their claws or their waste. These scent posts are meant to inform others of a cat's home range. House cats employ a vocal repertoire that extends from a purr to a screech. Diet. Domestic cats remain largely carnivorous, and have evolved a simple gut appropriate for raw meat. They also retain the rough tongue that can help them clean every last morsel from an animal bone (and groom themselves). Their diets vary with the whims of humans, however, and can be supplemented by the cat's own hunting successes.
"""

