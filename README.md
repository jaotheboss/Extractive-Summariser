# Extractive-Summariser
This project aims to develop a script that could effectively summarise articles in the extractive sense

## Objective:
This is simple. Basically, this script aims to summarise articles without changing the original sentences. 

## Methodology:
This script takes a page out of Google's pagerank algorithm to solve for relevant and important sentences in an article. Instead of ranking webpages, this algorithm ranks sentences within the article. 

In summary, basically the algorithm considers each and every sentence in the article and calculates how much a particular sentence has been referenced. If a sentence has been referred to (or as similar) many times, it is likely that it is an important sentence.

## To-do List:
- [ ] Create a method such that we can simply input websites instead of a full text