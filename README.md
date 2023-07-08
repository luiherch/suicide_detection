# Suicide Detection
![Python](https://img.shields.io/badge/python-3.10.10-green)
## Description
Text classification project to detect suicide ideation and depression. The dataset has been obtained from [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?resource=download), which is a collection from "SuicideWatch" and "depression" subreddits of the Reddit platform. Thanks to NIKHILESWAR KOMATI for creating the Dataset.

I have used the Naive Bayes classification for classifying the texts into suicide and non-suicide.

## Motivation
I discovered probabilistic classifiers in my major, but although they are quite simple, I didn't pay much attention to them. It wasn't until I came across a video from [StatQuest](https://www.youtube.com/watch?v=O2L2Uv9pdDA), which explained how Naive Bayes works in such a simple way that I immediately felt eager to try it out.

Regarding the chosen dataset, I feel it is a really interesting topic with a huge impact nowadays. Around 4000 people commit suicide in Spain every year, according to the [Spanish Ministry of Health](https://www.sanidad.gob.es/estadEstudios/estadisticas/estadisticas/estMinisterio/mortalidad/mortalidad.htm). Moreover, it has become the first cause of death among people aged between 14 to 29. Loneliness, depression, bullying, fear, and stigma are the main issues that hide beneath this silent pandemic.

Being able to detect depression and suicidal thoughts could help us to identify and prevent suicides.

## Natural Language Processing
SpaCy library has been used in order to process the documents. To extract useful information for every document, the following techniques have been performed:
1. Words with non-alphabetical characters have been removed.
2. Stop words have been removed since it is considered that they do not provide any useful information.
3. Nouns, verbs, adjectives, and adverbs have been lemmatized.

## About lemmatization
Lemmatization is a technique to reduce inflection and derived forms of words. It is similar to *stemming*, however, lemmatization takes into account the context in which the word is used to extract the proper lemma. Lemmatization is considered a more accurate and refined way of obtaining the lemma of a word, whereas stemming is usually faster. In this case, the decision was to use lemmatization to try to achieve better results.

## About Naive Bayes
Naive Bayes is a supervised classification algorithm that is based on Bayes' Theorem. It makes the assumption that the features used to classify an object are independent of each other. The algorithm computes the probability of an object belonging to a group given the features of that object. Finally, the group with the highest probability is selected.

Multinomial Bayes is a variant for multinomially distributed data, where the features have discrete counts such as the word frequencies. This method seems more adequate for this specific case.

## Results
![Suicide Wordcloud](https://github.com/luiherch/suicide_detection/blob/main/img/suicide_wordcloud.png?raw=true)
![Non-Suicide Wordcloud](https://github.com/luiherch/suicide_detection/blob/main/img/non_suicide_wordcloud.png?raw=true)
![Confusion matrix](https://github.com/luiherch/suicide_detection/blob/main/img/predictions_cm.png?raw=true)

## Future work
There are several ways to improve the classifier:
- Typo correction
- Hyper-parameter tuning
- Multiprocessing of the texts