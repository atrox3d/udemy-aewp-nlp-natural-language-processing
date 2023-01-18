import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas
import numpy
import wikipedia
import wikipedia.exceptions


def lemmatize(sentence):
    # print(f'lemmatize | {sentence=}')
    # tokenize sentence
    tokens = nltk.word_tokenize(sentence.lower())
    # print(f'lemmatize | {tokens=}')

    # analyze tokens and tag part-of-sentence
    pos_tags = nltk.pos_tag(tokens)
    # print(f'lemmatize | {pos_tags=}')

    # instantiate lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = []

    # get token and POS from list of tuples
    for token, tag in pos_tags:
        # get lower first char of POS and check if part of:
        # nouns, verbs, adjectives, adverbs
        if tag.lower()[0] in 'nvar':
            lemma = lemmatizer.lemmatize(token, tag.lower()[0])
            lemmas.append(lemma)
            # print(f'lemmatize | {lemma=}')
    # print(f'lemmatize | {lemmas=}')
    return lemmas


def process(text, question, coeff=0.3):
    tv = TfidfVectorizer(tokenizer=lemmatize)

    # split text into sentences
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)
    # print(f'{sentence_tokens}')

    tf = tv.fit_transform(sentence_tokens)
    df = pandas.DataFrame(tf.toarray(), columns=tv.get_feature_names_out())

    # for index, st in enumerate(sentence_tokens):
    #     print(index, st)
    # print(df.to_string())

    tf_question = tf[-1]
    tf_sentences = tf[:-1]
    values: numpy.ndarray = cosine_similarity(tf_question, tf_sentences)
    # print(f'{sentence_tokens}')
    # print(f'{values=}')
    # print(f'{type(values)=}')

    # print(values.argsort())
    # print(values.max())
    # print(values.argmax())
    vf = values.flatten()
    # print(vf[values.argmax()])
    # print(sentence_tokens[values.argmax()])

    if values.argmax() > coeff:
        return sentence_tokens[values.argmax()]


if __name__ == '__main__':
    # with open('sentences.txt') as f:
    #     sentences = f.read()
    # sentences = wikipedia.page('Vegetables').content
    while True:
        try:
            topic = input('Choose a topic: ')
            # result = wikipedia.search(topic)
            # print(result)
            # continue
            if topic == 'quit':
                break
            sentences = wikipedia.page(topic).content
        except wikipedia.exceptions.DisambiguationError as wde:
            print(wde)
        while True:
            question = input('Hi, what do you want to know?\n')
            print(f'{question = }')

            if question == 'quit':
                break
            elif question == 'content':
                print(sentences)
                continue

            answer = process(sentences, question)
            print(f'{answer or "I dont know"}')

