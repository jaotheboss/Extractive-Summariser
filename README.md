# Extractive-Summariser

![Extractive Summary](https://github.com/jaotheboss/Extractive-Summariser/blob/master/extractive_summary.png)

This project aims to develop a script that could effectively summarise articles in the extractive sense

## Objective:
This is simple. Basically, this script aims to summarise articles without changing the original sentences. 

## Methodology:
This script takes a page out of Google's pagerank algorithm to solve for relevant and important sentences in an article. Instead of ranking webpages, this algorithm ranks sentences within the article. 

In summary, basically the algorithm considers each and every sentence in the article and calculates how much a particular sentence has been referenced. If a sentence has been referred to (or as similar) many times, it is likely that it is an important sentence.

## To-do List:
- [x] Create a method such that we can simply input websites instead of a full text
- [ ] Make the script available for more than 1 news site (a html parser effort)
- [x] Tweak the parameters to improve on the extractive summary

# Updates:
1. Implemented BERT architecture encoders, which is a transformer, that generates contextualised sentence vectors for the sentence ranking algorithm 
2. Added in my own BERT sentence embedder API into the mix
3. This allowed us to remove the GloVe embeddings, which took up a lot of space
4. Added a fail-safe incase the 'request' module is not able to extract the html code
