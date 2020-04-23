from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import json


def visaulize_LDA(corpus, sklearn_vectorizer, save_path, n_topics):

    # fit/transform corpus
    sklearn_vectorizer.fit(corpus)
    dtm = sklearn_vectorizer.transform(corpus)

    # fit LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)

    # visualize
    vis_data = pyLDAvis.sklearn.prepare(lda, dtm, sklearn_vectorizer)

    # save
    with open(save_path, 'w') as file:
        json.dump(vis_data.to_dict(), file)


