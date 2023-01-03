import pandas as pd;
import numpy as np; 
import matplotlib.pyplot as plt;
import seaborn as sns;

import nltk 




plt.style.use('ggplot')



df = pd.read_csv('./Reviews.csv')

df.head()

ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar', 
          title = 'Count of Reviews by Stars',
          figsize=(10,5))
ax.set_xlabel('Review Stars')


# plt.show();


example = df['Text'][50];


tokens = nltk.word_tokenize(example);
tokens[:10]

nltk.pos_tag(tokens); 

tagged = nltk.pos_tag(tokens);
tagged[:10];

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()

sia.polarity_scores(example);

# running the polarity score on entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)



vaders = pd.DataFrame(res).T;

vaders = vaders.reset_index().rename(columns={'index':'Id'});

vaders = vaders.merge(df, how='left');

# Vaders sentiment score and metadata
vaders.head()

ax = sns.barplot(data=vaders, x='Score', y='compound');
ax.set_title('Comp scores')
plt.show()

    
