from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')


x = 'was'
y = 'is'

print(f'{x==y = }')

lemmatizer = WordNetLemmatizer()

lemmax = lemmatizer.lemmatize(x, 'v')
print(f'{lemmax=}')

lemmay = lemmatizer.lemmatize(y, 'v')
print(f'{lemmay=}')

print(f'{lemmax==lemmay = }')

lemmaveg = lemmatizer.lemmatize('vegetable', 'n')
print(f'{lemmaveg=}')

lemmavegs = lemmatizer.lemmatize('vegetables', 'n')
print(f'{lemmavegs=}')

print(f'{lemmaveg==lemmavegs = }')




