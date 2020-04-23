import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import warnings
import os
import json

from Pipeline import topic_modeling

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print('Creating LDA visualizations...')
    for i, topic_area in enumerate(['business', 'finance', 'general', 'tech']):
        print('{}: {}'.format(i, topic_area))
        df = pd.read_csv('data/covid19_articles_{}.csv'.format(topic_area), index_col=1)
        corpus = list(df['content_string'])
        cnt_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                        stop_words = 'english',
                                        lowercase = True,
                                        token_pattern = r'\b[a-zA-Z]{3,}\b',
                                        max_df = 0.5, 
                                        min_df = 10)

        # run LDA
        topic_modeling.visaulize_LDA(corpus, cnt_vectorizer, 
                                    'data/lda/{}.json'.format(topic_area), n_topics=20)

    # parse and load json into csv files
    os.remove('data/lda/vis.csv')
    os.remove('data/lda/topic_term.csv')
    for i, filename in enumerate(os.listdir('data/lda')):
        with open('data/lda/'+filename, 'r') as file:
            d = json.load(file)
            d['mdsDat']['topic_area'] = filename.split('.json')[0]
            d['tinfo']['topic_area'] = filename.split('.json')[0]
            if i == 0:
                header = True
            else:
                header = False
            pd.DataFrame(d['mdsDat']).to_csv('data/lda/vis.csv', mode='a', header=header, index=False)
            pd.DataFrame(d['tinfo']).to_csv('data/lda/topic_term.csv', mode='a', header=header, index=False)
