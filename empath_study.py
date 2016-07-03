from empath import Empath
from textblob import TextBlob
import matplotlib
import matplotlib.mlab as mlab
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lexicon = Empath()

pos_txt = open('persuasiveargs.txt', 'r').read()
neg_txt = open('notpersuasiveargs.txt', 'r').read()

cat_pos = lexicon.analyze(pos_txt, normalize=True)
cat_neg = lexicon.analyze(neg_txt, normalize=True)

for k in cat_pos:
    if cat_neg[k] != 0:
        r = cat_pos[k] / cat_neg[k]
        if r > 2:
            print('1 Categoria: ', k, 'Pos:', cat_pos[k], 'Neg:', cat_neg[k])

    if cat_pos[k] != 0:
        r = cat_neg[k] / cat_pos[k]
        if r > 2:
            print('2 Categoria: ', k, 'Pos:', cat_pos[k], 'Neg:', cat_neg[k])

blob_pos = TextBlob(pos_txt)
polarity_pos = []
subjectivity_pos = []
for sentence in blob_pos.sentences:
    polarity_pos.append(sentence.sentiment.polarity)
    subjectivity_pos.append(sentence.sentiment.subjectivity)

blob_neg = TextBlob(neg_txt)
polarity_neg = []
subjectivity_neg = []
for sentence in blob_neg.sentences:
    polarity_neg.append(sentence.sentiment.polarity)
    subjectivity_neg.append(sentence.sentiment.subjectivity)

n, bins, patches = plt.hist(subjectivity_pos, 50,  histtype='stepfilled', color='b', label='Persuasive')
n, bins, patches = plt.hist(subjectivity_neg, 50,  histtype='stepfilled',  color='r', alpha=0.5, label='Not Persuasive')
plt.xlabel('Subjectivity')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
# plt.show()
plt.plot()
plt.savefig('subjectivity.png')
