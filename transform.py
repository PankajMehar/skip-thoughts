import sys
import nltk
import unidecode
import skipthoughts
from decoding import tools

pattern="\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"

model=skipthoughts.load_model()
text=sys.argv[0]
text=unidecode.unidecode(text)
sents=nltk.sent_tokenize(text)
Xsample=[]
for sent in sents:
    words=nltk.regexp_tokenize(sent,pattern)
    cleaned=[]
    for word in words:
        if word.isalpha(): cleaned.append(word)
    cleanedsent=' '.join(cleaned)
    Xsample.append(cleanedsent)
print Xsample[0]
print len(Xsample)
textvector=skipthoughts.encode(model,Xsample)
dec=tools.load_model()
transformedtext = tools.run_sampler(dec, textvector, beam_width=1, stochastic=True, use_unk=False)
print transformedtext[0]
