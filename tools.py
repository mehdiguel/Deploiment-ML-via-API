import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from unidecode import unidecode
import re

def lecture(path):
    return pd.read_csv(path)

def macronet(df):
    df.drop_duplicates(subset=['content'], inplace=True)
    df['len'] = df['content'].apply(lambda x: len(x.split(' ')))
    df = df[df['len'] > 15]
    return df

def nett(texte):
    """
    :param texte: commentaire unique
    :return: commentaire unique nettoyé
    """

    p = "[a-z]{1,}"

    with open("stop-w_fr.txt") as f:
        sw = f.readlines()
    sw = list(set([unidecode(el.strip()).lower() for el in sw]))
    sw.remove('pas')
    sw.remove('n')
    sw.remove('ne')
    sw.remove('tres')
    sw.remove('trop')
    sw.remove('bien')
    sw.remove('bon')

    final=''
    texte=unidecode(texte.strip()).lower()
    for elem in re.findall(p,texte):
        if elem in sw:
            continue
        else:
            final=final+' '+elem
    return final


def equilibre(df):
    df.loc[df[df['score'] < 3].index, "score"] = 0
    df.loc[df[df['score'] > 3].index, "score"] = 1
    df=df[df['score']!=3]
    seuil = df['score'].value_counts().min()
    df2 = pd.concat([df[df['score'] == 1].sample(seuil), df[df['score'] == 0].sample(seuil)])
    return df2

def feature(df,user):
    """
    :param df: df à featuriser
    avec sauvegarde du vocablaire du TFIDF
    :return: Matrice BOW Tfidf
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['content_nett'])
    # Save vectorizer.vocabulary_
    pickle.dump(vectorizer.vocabulary_, open("data/"+user+'_'+"feature.pkl", "wb"))
    x = vectorizer.transform(df['content_nett'])
    return x

def train(x,y,user):
    """
    :param X: Matrice BOW Tfidf
    avec sauvegarde du model de classification
    :return: metrice d'entrainement
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    clf = LogisticRegression(C=1.5).fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    filename = 'data/' + user + '_' +str(round(acc,2))+'_clf.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    return acc


def prediction_user(phrase,user):
    #Load vocabulary from TFIDF
    path_voc=user.split('_')[0]+'_'+"feature.pkl"
    with open("data/"+path_voc, "rb") as f:
        voca = pickle.load(f)
    transformer = TfidfVectorizer(vocabulary = voca )
    # Featuring de la phrase user
    user_sentence = transformer.fit_transform([nett(phrase)])
    # Load classifier
    cls=pickle.load(open("data/"+user, "rb"))
    # Predict from loaded classifier
    #cls.predict(user),cls.predict_proba(user).max()
    return cls.predict(user_sentence)
