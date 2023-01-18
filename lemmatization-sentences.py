import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def lemmatize(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    print(tokens)

    pos_tags = nltk.pos_tag(tokens)
    print(pos_tags)

    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = []
    for token, tag in pos_tags:
        if tag.lower()[0] in 'nvar':
            lemma = lemmatizer.lemmatize(token, tag.lower()[0])
            lemmas.append(lemma)
            print(lemma)
    return lemmas


if __name__ == '__main__':
    s1 = 'Vegetables are types of plants.'
    l1 = lemmatize(s1)
    print(l1)

    s2 = 'A Vegetables is a types of plant.'
    l2 = lemmatize(s2)
    print(l2)

    print(f'{l1==l2 = }')
