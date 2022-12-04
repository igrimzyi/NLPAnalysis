import pandas as pd;
import numpy as np; 
import matplotlib.pyplot as plt;
import seaborn as sns;

import nltk 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer;
from tqdm.notebook import tqdm

plt.style.use('ggplot')



df = pd.read_csv('./Reviews.csv')

df.head()

ax = df['Score'] \
        .value_counts() \
        .sort_index() \
        .plot(kind='bar', title='Count by stars', figsize=(10,5));

ax.set_xlabel('Review Stars');


plt.show();


example = df['Text'][50];

print(example);

tokens = nltk.word_tokenize(example);
tokens[:10]

nltk.pos_tag(tokens); 

tagged = nltk.pos_tag(tokens);
tagged[:10];

entities = nltk.chunk.ne_chunk(tagged);

entities.pprint()

sia = SentimentIntensityAnalyzer()

print(sia.polarity_scores(example))

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text'];
    myid = row['Id']
