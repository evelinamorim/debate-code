import sys
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import time
from empath import Empath
import functools
import os
import csv
from scipy import spatial

FSTPERSON_PRO = ["i", "we", "me", "us", "my", "mine", "our", "ours"]
FSTPERSON_PRO_PLURAL = ["we", "us", "our", "ours"]
SNDPERSON_PRO = ["you", "your", "yours"]

POS_TAG_SET = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', '$', "''", '(', ')', ',', '--', '.', ':', '#']

# relevant categories from empath to this problem (?)
EMPATH_CAT_REL =  ['disgust', 'disappointment','joy', 'hate', 'contentment', 'optimism', 'love', 'pain', 'aggression', 'torment', 'envy', 'kill', 'injury', 'violence', 'fun', 'ugliness', 'politeness', 'anger', 'fear', 'horror', 'emotional', 'sympathy', 'confusion', 'science', 'sadness', 'irritability', 'rage', 'affection', 'rational']

stopwords_english = stopwords.words('english')

lexicon = Empath()
wordnet_lemmatizer = WordNetLemmatizer()
CONCRETENESS = {}
AFFECTIVE = {}
freq_pro_cat = {}
freq_against_cat = {}
A = 5

class Debate:

    def __init__(self):
        self.title = ""
        self.abstract = ""
        # for each team (pro or against) there is a list of triplets like 
        #  like: (title, argument, counterargument)
        self.teams = {}
        # for each period of time (before and after), there will be a list 
        # tuples in the following format: (votetype, percentage)
        # vote type can be one of the following strings: strongly against, 
        # mildly against, dont know, mildly for, strongly for
        self.votes = {}
        self.features_names = []
        self.features = []
        self.abstract_cat_empath = {}

        self.ttoken_arg_count = {}
        for t in POS_TAG_SET:
            self.ttoken_arg_count[t] = 0

        self.ttoken_counter_count = {}
        for t in POS_TAG_SET:
            self.ttoken_counter_count[t] = 0

        self.topic_empath_arg = {}
        for t in EMPATH_CAT_REL:
            self.topic_empath_arg[t] = 0

        self.topic_empath_counter = {}
        for t in EMPATH_CAT_REL:
            self.topic_empath_counter[t] = 0

        self.increase = {}

    def add_teams(self, pro, against):

        self.teams = {}
        self.teams["pro"] = pro
        self.teams["against"] = against

    def add_votes(self, before, after):

        self.votes = {}
        self.votes["before"] = before
        self.votes["after"] = after

    def get_ftrs_names(self):

        return self.debate_ftrs_names() + self.arguments_ftrs_names() + self.empath_ftrs_names()

    def debate_ftrs_names(self):

        ftrs_names = []
        ftrs_names.append("nargs")
        ftrs_names.append("nargspro")
        ftrs_names.append("nargsagainst")
        return ftrs_names

    def debate_features(self):

        features = []
        # number of arguments  pro
        nargspro = len(self.teams["pro"])
        # number of arguments against
        nargsagainst = len(self.teams["against"])
        # number of arguments in debate
        nargs = nargspro + nargsagainst
        
        features.append(nargs)
        features.append(nargspro)
        features.append(nargsagainst)

        return features

    def arguments_ftrs_names(self):
        # just to give features added by extract_arguments method
        ftrs_names = []
        ftrs_names.append("nsentarg") # 1
        ftrs_names.append("nsentcounter") # 2
        ftrs_names.append("nwordsarg") # 3
        ftrs_names.append("nwordscounter") # 4
        ftrs_names.append("ndefarg") # 5
        ftrs_names.append("firstpersonarg") # 6
        ftrs_names.append("firstpersonpluralarg") # 7
        ftrs_names.append("indefartarg") # 8
         
        ftrs_names.append("sndpersonarg") # 9
        ftrs_names.append("concretenessarg") # 10
        ftrs_names.append("valencearg") # 11
        ftrs_names.append("arousalarg") # 12
        ftrs_names.append("dominancearg") # 13

        # 14 - 59
        for tt in self.ttoken_arg_count:
            if tt == ',':
                ftrs_names.append("virg_arg")
            elif tt == "''":
                ftrs_names.append("quote_arg")
            else:
                ftrs_names.append("%s_arg" % tt)

        ftrs_names.append("ndefcounter") # 60
        ftrs_names.append("firstpersoncounter") # 61
        ftrs_names.append("firstpersonpluralcounter") # 62
        ftrs_names.append("indefartcounter") # 63
        ftrs_names.append("sndpersoncounter") # 64
        ftrs_names.append("concretenesscounter") # 65
        ftrs_names.append("valencecounter") # 66
        ftrs_names.append("arousalcounter") # 67
        ftrs_names.append("dominancecounter") # 68

        # 69 - 114
        for tt in self.ttoken_counter_count:
            if tt == ",":
                ftrs_names.append("virg_counter")
            elif tt == "''":
                ftrs_names.append("quote_counter")
            else:
                ftrs_names.append("%s_counter" % tt)

        ftrs_names.append("nwords_neg_arg") # 115
        ftrs_names.append("nwords_pos_arg") # 116
        ftrs_names.append("fracwords_pos_arg") # 117

        ftrs_names.append("nwords_neg_counter") # 118
        ftrs_names.append("nwords_pos_counter") # 119
        ftrs_names.append("fracwords_pos_counter") # 120

        ftrs_names.append("support_arg") # 121
        ftrs_names.append("causal_arg") # 122
        ftrs_names.append("complementary_arg") # 123
        ftrs_names.append("conflict_arg") # 124

        ftrs_names.append("support_counter") # 125
        ftrs_names.append("causal_counter") # 126
        ftrs_names.append("complementary_counter") # 127
        ftrs_names.append("conflict_counter") # 128

        ftrs_names.append("common_empath_cat") # 129
        ftrs_names.append("ncommon_stopwords") # 130
        ftrs_names.append("ncommon_voc") # 131
        ftrs_names.append("jaccard_stopwords") # 132
        ftrs_names.append("jaccard_words") # 133

        ftrs_names.append("jaccard_title_arg") # 134
        ftrs_names.append("jaccard_title_counter") # 135
        ftrs_names.append("jaccard_abstract_arg") # 136
        ftrs_names.append("jaccard_abstract_counter") # 137

        ftrs_names.append("class") # 138
        return ftrs_names

    def extract_arguments(self, typeteam):


        features = []
        # tagging of arguments
        arg_lst = []
        nsetences = 0
        for (t, a, c) in self.teams[typeteam]:
             ftrs_arg = []

             t0 = time.time()
             text = word_tokenize(a)
             sent = sent_tokenize(a)
              
             arg_tags = nltk.pos_tag(text)
             t1 = time.time()
             # print('Tokenization of argument ', t1-t0, 's')

             t0 = time.time()
             text_counter = word_tokenize(c)
             sent_counter = sent_tokenize(c)
             counter_tags = nltk.pos_tag(text_counter)
             t1 = time.time()
             # print('Tokenization of counterargument ', t1-t0, 's')

             # 1: number of sentences in arguments
             ftrs_arg.append(len(sent))
             # 2: number of sentences in counterarguments
             ftrs_arg.append(len(sent_counter))

             #  3: number of words in arguments 
             ftrs_arg.append(len(arg_tags))
             #  4: number of words in couterarguments 
             ftrs_arg.append(len(counter_tags))


             # pos tag features for arguments
             ndef = 0 # definite article
             firstperson = 0
             firstpersonplural = 0
             indefart = 0 # indefinite article
             sndperson = 0
             type_tokens = {}
             concreteness_arg = 0
             valence_arg = 0
             arousal_arg = 0
             dominance_arg = 0
             stopwords_arg = set()
             voc_arg = set()
             t0 = time.time()

             for (tok, tag) in arg_tags:

                 lemma_tok = wordnet_lemmatizer.lemmatize(tok)

                 voc_arg.add(tok)
                 if tok in stopwords_english:
                     stopwords_arg.add(tok)

                 if tag in type_tokens:
                     type_tokens[tag] = type_tokens[tag] + 1
                 else:
                     type_tokens[tag] = 1

                 self.ttoken_arg_count[tag] = self.ttoken_arg_count[tag] + 1


                 tokl = tok.lower()
                 if tokl == "the":
                     ndef = ndef + 1
                 elif tokl in FSTPERSON_PRO_PLURAL:
                     firstperson = firstperson + 1
                     firstpersonplural = firstpersonplural + 1
                 elif tokl in FSTPERSON_PRO:
                     firstperson = firstperson + 1
                 elif tokl == "a" or tokl == "an":
                     indefart = indefart + 1
                 elif tokl in SNDPERSON_PRO:
                     sndperson = sndperson + 1

                 if lemma_tok in CONCRETENESS:
                     concreteness_arg = CONCRETENESS[lemma_tok] + concreteness_arg

                 if lemma_tok in AFFECTIVE:
                     (valence, arousal, dominance) = AFFECTIVE[lemma_tok]
                     valence_arg = valence + valence_arg
                     arousal_arg = arousal + arousal_arg
                     dominance_arg = dominance + dominance_arg
             t1 = time.time()
             # print("Extracting pos tag features argument", t1-t0, "s")

             concreteness_arg = concreteness_arg / len(arg_tags)
             valence_arg = valence_arg / len(arg_tags)
             arousal_arg = arousal_arg / len(arg_tags)
             dominance_arg = dominance_arg / len(arg_tags)
             # 5: number of definitive articles in arg
             ftrs_arg.append(ndef)
             # 6: number of first person pronouns in arg
             ftrs_arg.append(firstperson)
             # 7: number of first person plural pronouns in arg
             ftrs_arg.append(firstpersonplural)
             # 8: number of indefinitive articles in arg
             ftrs_arg.append(indefart)
             # 9: number of second person pronouns in arg
             ftrs_arg.append(sndperson)
             # 10: concreteness degree in arg
             ftrs_arg.append(concreteness_arg)
             # 11: valence degree in arg
             ftrs_arg.append(valence_arg)
             # 12: arousal degree in arg
             ftrs_arg.append(arousal_arg)
             # 13: dominance degree in arg
             ftrs_arg.append(dominance_arg)

             t0 = time.time()
             # 14 - 59: type token in arg
             for tt in self.ttoken_arg_count:
                 self.ttoken_arg_count[tt] = self.ttoken_arg_count[tt] / len(arg_tags)
                 ftrs_arg.append(self.ttoken_arg_count[tt])

             t1 = time.time()
             # print("Extracting type token ratio argument", t1-t0, "s")
       
             # pos tag features for counterarguments
             ndef = 0 # definite article
             firstperson = 0
             firstpersonplural = 0
             indefart = 0 # indefinite article
             sndperson = 0
             concreteness_counter = 0
             valence_counter = 0
             arousal_counter = 0
             dominance_counter = 0
             stopwords_counter = set()
             voc_counter = set()
             t0 = time.time()
             for (tok, tag) in counter_tags:

                 lemma_tok = wordnet_lemmatizer.lemmatize(tok)
                 voc_counter.add(tok)
                 if tok in stopwords_english:
                     stopwords_counter.add(tok)
      
                 if tag in type_tokens:
                     type_tokens[tag] = type_tokens[tag] + 1
                 else:
                     type_tokens[tag] = 1


                 self.ttoken_counter_count[tag] = self.ttoken_counter_count[tag] + 1                 
                 tokl = tok.lower()
                 if tokl == "the":
                     ndef = ndef + 1
                 elif tokl in FSTPERSON_PRO_PLURAL:
                     firstperson = firstperson + 1
                     firstpersonplural = firstpersonplural + 1
                 elif tokl in FSTPERSON_PRO:
                     firstperson = firstperson + 1
                 elif tokl == "a" or tokl == "an":
                     indefart = indefart + 1
                 elif tokl in SNDPERSON_PRO:
                     sndperson = sndperson + 1

                 if lemma_tok in CONCRETENESS:
                     concreteness_counter = CONCRETENESS[lemma_tok] + concreteness_counter
                 if lemma_tok in AFFECTIVE:
                     (valence, arousal, dominance) = AFFECTIVE[lemma_tok]
                     valence_counter = valence + valence_counter
                     arousal_counter = arousal + arousal_counter
                     dominance_counter = dominance + dominance_counter

             t1 = time.time()
             # print("Extracting pos features counterargument", t1-t0, "s")
             concreteness_counter = concreteness_counter / len(counter_tags)
             valence_counter = valence_counter / len(counter_tags)
             arousal_counter = arousal_counter / len(counter_tags)
             dominance_counter = dominance_counter / len(counter_tags)

             # 60: number of definitive articles in counter
             ftrs_arg.append(ndef)
             # 61: number of first person pronouns in counter
             ftrs_arg.append(firstperson)
             # 62: number of first person plural pronouns in counter
             ftrs_arg.append(firstpersonplural)
             # 63: number of indefinitive articles in counter
             ftrs_arg.append(indefart)
             # 64: number of second person pronouns in counter
             ftrs_arg.append(sndperson)
             # 65: concreteness degree in counter
             ftrs_arg.append(concreteness_counter)
             # 66: valence degree in counter
             ftrs_arg.append(valence_counter)
             # 67: arousal degree in counter
             ftrs_arg.append(arousal_counter)
             # 68: dominance degree in counter
             ftrs_arg.append(dominance_counter)

             t0 = time.time()
             # 69 - 114: type token in counter
             for tt in self.ttoken_counter_count:
                 self.ttoken_counter_count[tt] = self.ttoken_counter_count[tt] / len(counter_tags)
                 ftrs_arg.append(self.ttoken_counter_count[tt])
             t1 = time.time()
             # print("Extracting token ratio counterargument", t1-t0, "s")

             t0 = time.time()
             cat_arg = lexicon.analyze(a)
             cat_counter = lexicon.analyze(c)
             t1 = time.time()
             # print("Empath analysis ",t1-t0,"s")

             # 115: number of negative words in args
             ftrs_arg.append(cat_arg['negative_emotion'])
             # 116: number of positive words in args 
             ftrs_arg.append(cat_arg['positive_emotion happiness'])
             # 117: frac of positive words in args 
             ftrs_arg.append(cat_arg['positive_emotion happiness']/ len(arg_tags))

             # 118: number of negative words in counter
             ftrs_arg.append(cat_counter['negative_emotion'])
             # 119: number of positive words in counter
             ftrs_arg.append(cat_counter['positive_emotion happiness'])
             # 120: frac of positive words in counter
             ftrs_arg.append(cat_counter['positive_emotion happiness']/ len(counter_tags))

             # Dicourse elemnts by Empath categories to argument 
             # 121: support in args
             ftrs_arg.append(cat_arg['support'])
             # 122: causal_argument in args
             ftrs_arg.append(cat_arg['causal_argument'])
             # 123: complementary in args
             ftrs_arg.append(cat_arg['complementary'])
             # 124: conflict in args
             ftrs_arg.append(cat_arg['conflict'])

             # Dicourse elemnts by Empath categories to counterargument 
             # 125: support in counter
             ftrs_arg.append(cat_counter['support'])
             # 126: causal_argument in counter
             ftrs_arg.append(cat_counter['causal_argument'])
             # 127: complementary in counter
             ftrs_arg.append(cat_counter['complementary'])
             # 128: conflict in counter
             ftrs_arg.append(cat_counter['conflict'])

             # 129: common in empath
             common_cat = common_empath(cat_arg, cat_counter)
             ftrs_arg.append(len(common_cat))

             # 130: common in stop words
             ncommon_stopwords = len(stopwords_arg.intersection(stopwords_counter))
             ftrs_arg.append(ncommon_stopwords)

             # 131: common in content
             ncommon_words = len(voc_arg.intersection(voc_counter))
             ftrs_arg.append(ncommon_words)

             # 132: jaccard stop words
             jaccard_stopwords = ncommon_stopwords / len(stopwords_counter.union(stopwords_counter))
             ftrs_arg.append(jaccard_stopwords)

             # 133: jacaard in content
             jaccard_words = ncommon_words / len(voc_arg.union(voc_counter))
             ftrs_arg.append(jaccard_words)

             # 134: jaccard title x arg
             inter_title = voc_arg.intersection(self.title_tok)
             union_title = voc_arg.union(self.title_tok)
             jaccard_title_arg = len(inter_title) / len(union_title)
             ftrs_arg.append(jaccard_title_arg)
             # 135: jaccard title x counterarg
             inter_title = voc_counter.intersection(self.title_tok)
             union_title = voc_counter.union(self.title_tok)
             jaccard_title_counter = len(inter_title) / len(union_title)
             ftrs_arg.append(jaccard_title_counter)

             # 136: jaccard abstract x arg
             inter_abstract = voc_arg.intersection(self.abstract_tok)
             union_abstract = voc_arg.union(self.abstract_tok)
             jaccard_abstract_arg = len(inter_abstract) / len(union_abstract)
             ftrs_arg.append(jaccard_abstract_arg)
             # 137: jaccard abstract x counterarg
             inter_abstract = voc_counter.intersection(self.abstract_tok)
             union_abstract = voc_counter.union(self.abstract_tok)
             jaccard_abstract_counter = len(inter_abstract) / len(union_abstract)
             ftrs_arg.append(jaccard_abstract_counter)

             # 138: class
             self.add_increase_for_class(ftrs_arg, typeteam)

             features.append(ftrs_arg)

        return features

    def empath_ftrs_names(self):

        ftrs_names = []
        ftrs_names.append("jaccardempath")
        ftrs_names.append("cosineempath")

        for t in self.topic_empath_arg:
            ftrs_names.append("%s_arg" % t)

        for t in self.topic_empath_counter:
            ftrs_names.append("%s_counter" % t)

        ftrs_names.append("jaccard_empath_title_arg")
        ftrs_names.append("jaccard_empath_title_counter")
        ftrs_names.append("jaccard_empath_abstract_arg")
        ftrs_names.append("jaccard_empath_abstract_counter")

        ftrs_names.append("cosine_empath_title_arg")
        ftrs_names.append("cosine_empath_title_counter")
        ftrs_names.append("cosine_empath_abstract_arg")
        ftrs_names.append("cosine_empath_abstract_counter")
        return ftrs_names

    def extract_empath(self, typeteam, title_cat, abstract_cat):

        features = []

        for (t, a, c) in self.teams[typeteam]:
             ftrs_arg = []

             arg_allcat = lexicon.analyze(a, normalize=True)
             counter_allcat = lexicon.analyze(c, normalize=True)

             arg_cat = get_empath_cat(arg_allcat)
             counter_cat = get_empath_cat(counter_allcat)

             arg_keys = set(arg_cat.keys())
             counter_keys = set(counter_cat.keys())

             inter_arg = arg_keys.intersection(counter_keys)
             union_arg = arg_keys.union(counter_keys)

             jaccard_empath = len(inter_arg)/len(union_arg)

             ftrs_arg.append(jaccard_empath)

             vector_arg_empath = list(arg_allcat.values())
             vector_counter_empath = list(counter_allcat.values())

             cosine_similarity = 1 - spatial.distance.cosine(vector_arg_empath,
                                                             vector_counter_empath)

             ftrs_arg.append(cosine_similarity)

             # some empath topics frequency in arg
             for t in self.topic_empath_arg:
                 ftrs_arg.append(arg_allcat[t])

             # some empath topics frequency in counter
             for t in self.topic_empath_counter:
                 ftrs_arg.append(counter_allcat[t])

             #  jaccard with title debate
             title_keys = set(title_cat.keys())
             inter_arg = arg_keys.intersection(title_keys)
             union_arg = arg_keys.union(title_keys)
             if len(union_arg) != 0:
                 jaccard_empath_title_arg = len(inter_arg) / len(union_arg)
             else:
                 jaccard_empath_title_arg = 1
             ftrs_arg.append(jaccard_empath_title_arg)

             inter_arg = counter_keys.intersection(title_keys)
             union_arg = counter_keys.union(title_keys)
             if len(union_arg) != 0:
                  jaccard_empath_title_counter = len(inter_arg) / len(union_arg)
             else:
                  jaccard_empath_title_counter = 1
             ftrs_arg.append(jaccard_empath_title_counter)

             # jaccard with abstract debate
             abstract_keys = set(abstract_cat.keys())
             inter_arg = arg_keys.intersection(abstract_keys)
             union_arg = arg_keys.union(abstract_keys)
             if len(union_arg) != 0:
                 jaccard_empath_abstract_arg = len(inter_arg) / len(union_arg)
             else:
                 jaccard_empath_abstract_arg = 1
             ftrs_arg.append(jaccard_empath_abstract_arg)

             inter_arg = counter_keys.intersection(abstract_keys)
             union_arg = counter_keys.union(abstract_keys)
             if len(union_arg) != 0:
                 jaccard_empath_abstract_counter = len(inter_arg) / len(union_arg)
             else:
                 jaccard_empath_abstract_counter = 1
             ftrs_arg.append(jaccard_empath_abstract_counter)

             # cosine with debate title
             vector_title_empath = list(self.title_cat.values())
             cosine_similarity = 1 - spatial.distance.cosine(vector_arg_empath,
                                                             vector_title_empath)
             ftrs_arg.append(cosine_similarity)             

             cosine_similarity = 1 - spatial.distance.cosine(vector_counter_empath,
                                                             vector_title_empath)
             ftrs_arg.append(cosine_similarity)             

             # cosine with debate abstract
             vector_abstract_empath = list(self.abstract_cat.values())
             cosine_similarity = 1 - spatial.distance.cosine(vector_arg_empath,
                                                             vector_abstract_empath)
             ftrs_arg.append(cosine_similarity)             

             cosine_similarity = 1 - spatial.distance.cosine(vector_counter_empath,
                                                             vector_abstract_empath)
             ftrs_arg.append(cosine_similarity)             
             features.append(ftrs_arg)

        return features

    def extract_empath_cat(self):

        arg_txt = ""
        counter_txt = ""
        for (t, a, c) in self.teams["pro"]:
             arg_txt = arg_txt  + " " + a
             counter_txt = counter_txt + " " + c

        arg_pro_cat = lexicon.analyze(arg_txt, normalize=True)
        counter_pro_cat = lexicon.analyze(counter_txt, normalize=True)

        arg_pro_cat = get_empath_cat(arg_pro_cat)
        counter_pro_cat = get_empath_cat(counter_pro_cat)

        arg_pro_keys = set(arg_pro_cat.keys())
        counter_pro_keys = set(counter_pro_cat.keys())

        inter_arg = arg_pro_keys.intersection(counter_pro_keys)
        diff_arg = arg_pro_keys.difference(counter_pro_keys)
        diff_counter = counter_pro_keys.difference(arg_pro_keys)
        for i in diff_counter:
            if i in freq_pro_cat:
                freq_pro_cat[i] = freq_pro_cat[i] + 1
            else:
                freq_pro_cat[i] = 1
            #print(i, arg_pro_cat[i], "pro")

        arg_txt = ""
        counter_txt = ""
        for (t, a, c) in self.teams["against"]:
             arg_txt = arg_txt  + " " + a
             counter_txt = counter_txt + " " + c

        arg_against_cat = lexicon.analyze(arg_txt, normalize=True)
        counter_against_cat = lexicon.analyze(counter_txt, normalize=True)

        arg_against_cat = get_empath_cat(arg_against_cat)
        counter_against_cat = get_empath_cat(counter_against_cat)

        arg_against_keys = set(arg_against_cat.keys())
        counter_against_keys = set(counter_against_cat.keys())

        inter_arg = arg_against_keys.intersection(counter_against_keys)
        diff_arg = arg_against_keys.difference(counter_against_keys)
        diff_counter = counter_against_keys.difference(arg_against_keys)
        for i in diff_counter:
            if i in freq_against_cat:
                freq_against_cat[i] = freq_against_cat[i] + 1
            else:
                freq_against_cat[i] = 1
            #print(i, arg_against_cat[i], "against")


    def extract_features(self):

        self.title_tok = set(word_tokenize(self.title))
        self.title_cat = lexicon.analyze(self.title, normalize=True)

        self.abstract_tok = set(word_tokenize(self.abstract))
        self.abstract_cat = lexicon.analyze(self.abstract, normalize=True)

        title_cat = get_empath_cat(self.title_cat)
        abstract_cat = get_empath_cat(self.abstract_cat)

        self.ttoken_arg_count = {}
        for t in POS_TAG_SET:
            self.ttoken_arg_count[t] = 0

        self.ttoken_counter_count = {}
        for t in POS_TAG_SET:
            self.ttoken_counter_count[t] = 0

        ### Debate Features ###
        ftrs_debate = self.debate_features()
        self.features_names = self.debate_ftrs_names()

        ### (counter) Argument Features (pro and against)###
        # TODO: add pro and against as features (tipo)
        ftrs_args_pro = self.extract_arguments("pro")
        ftrs_args_against = self.extract_arguments("against")

        ftrs_empath_pro = self.extract_empath("pro", title_cat, abstract_cat)
        ftrs_empath_against = self.extract_empath("against", title_cat, abstract_cat)

        ftrs_args_pro = [ ftrs_debate + x for x in ftrs_args_pro]
        ftrs_args_against = [ ftrs_debate + x for x in ftrs_args_against]

        ftrs_pro = list(zip(ftrs_args_pro, ftrs_empath_pro))
        ftrs_against = list(zip(ftrs_args_against, ftrs_empath_against))
        ftrs_pro = [x + y for (x, y) in ftrs_pro]
        ftrs_against = [x + y for (x, y) in ftrs_against]


        self.features_names = self.features_names + self.arguments_ftrs_names() + self.empath_ftrs_names()
        self.features = ftrs_pro + ftrs_against

    def write_features(self, fd_ftrs):

        for arg in self.features:
            # functools.reduce(lambda x,y: str(x) + ',' + str(y), arg)
            ftrs_line = "%.5f" % arg[0]
            for i in range(1, len(arg)):
                ftrs_line = ftrs_line + (",%.5f" % arg[i]) 
            fd_ftrs.write("%s\n" % ftrs_line)

    def write_header(self, fd_ftrs):
        ftrs_names = self.get_ftrs_names()
        header = functools.reduce(lambda x,y: x + ',' + y, ftrs_names)
        fd_ftrs.write("%s\n" % header)

    def add_increase_for_class(self, ftrs_arg, typeteam):

       
        before_for = 0        
        before_against = 0
        for (type_vote, p) in self.votes["before"]:
            if type_vote == "mildlyfor":
                before_for = int(p) + before_for
            if type_vote == "mildlyagainst":
                before_against = int(p) + before_against

        after_for = 0        
        after_against = 0

        for (type_vote, p) in self.votes["after"]:
            if type_vote == "mildlyfor" or type_vote == "stronglyfor":
                after_for = int(p) + after_for
            if type_vote == "mildlyagainst" or type_vote == "stronglyagainst":
                after_against = int(p) + after_against


        diff_for = after_for - before_for
        if typeteam == "pro":
            self.increase["pro"] = after_for-before_for
            if after_for - after_against >  0:
                 ftrs_arg.append(1)
            else:
                 ftrs_arg.append(0)
            # if diff_for > A:
            #    ftrs_arg.append(1)
            #else:
            #    ftrs_arg.append(0)

        diff_against = after_against - before_against
        if typeteam == "against":
            self.increase["against"] = after_against-before_against
            if after_against - after_for >  0:
                 ftrs_arg.append(1)
            else:
                 ftrs_arg.append(0)
            #if diff_against > A:
            #    ftrs_arg.append(1)
            #else:
            #    ftrs_arg.append(0)


def get_empath_cat(cat):
   # get only non zero categories
   empath_cat = {}
   for c in cat:
       if cat[c] != 0:
           empath_cat[c] = cat[c]   
   return empath_cat

def extract_args_xml(nodes):

    pro_args = []
    against_args = []

    for n in nodes:
        if n.tag == "pro":
            items_pro = n.getchildren()
            for i in items_pro:
                item_elem = i.getchildren()
                pro_args.append((item_elem[0].text, item_elem[1].text, item_elem[2].text))
        elif n.tag == "against":    
            items_against = n.getchildren()
            for i in items_against:
                item_elem = i.getchildren()
                against_args.append((item_elem[0].text, item_elem[1].text, item_elem[2].text))

    return pro_args, against_args

def extract_votes_xml(nodes):

    before_votes = []
    after_votes = []

    for n in nodes:
        if n.tag == "before":
            for c in n.getchildren():
                before_votes.append((c.tag,c.text))
        elif n.tag == "after":
            for c in n.getchildren():
                after_votes.append((c.tag,c.text))

    return before_votes, after_votes

def read_data(fname):

    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.XML(open(fname, 'r').read(), parser)

    debate_list = []
    debate_nodes = root.getchildren()

    for d in debate_nodes:
        debate = Debate()
        
        for node in d.getchildren():
            if node.tag == "title":
                debate.title = node.text
            elif node.tag == "abstract":
                debate.abstract = node.text
                debate.abstract_cat_empath = lexicon.analyze(debate.abstract)
            elif node.tag == "arguments":
                pro_args, against_args = extract_args_xml(node.getchildren())
                debate.add_teams(pro_args, against_args)
            elif node.tag == "votes":
                before_votes, after_votes = extract_votes_xml(node.getchildren())
                debate.add_votes(before_votes, after_votes)

        debate_list.append(debate)
        
    return debate_list

def common_empath(cat1, cat2):

    # given two empath distribution if cat1[t]!=0 and cat2[t]!=0, 
    # then count t as a common category
    common_cat = []
    for t in cat1:
        if cat1[t] != 0 and cat2 != 0:
            common_cat.append(t)

    return common_cat 

def build_concreteness_dict(dict_file):
    concreteness_dict = {}

    fd = open(dict_file, "r")
    header = fd.readline()
    for line in fd:
        info = line.split()
        word = None
        conc = None
        if len(info) == 9:
            word = info[0]
            conc = float(info[2])
        elif len(info) == 10:
            word = info[0] + " " + info[1]
            conc = float(info[3])

        if conc is not None:
            concreteness_dict[word] = conc


    fd.close()
    return concreteness_dict

def build_affective_dict(dict_file):

    affective_dict = {}

    with open(dict_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            w = row['Word']
            valence = float(row['V.Mean.Sum'])
            arousal = float(row['A.Mean.Sum'])
            dominance = float(row['D.Mean.Sum'])
            affective_dict[w] = (valence, arousal, dominance)

    return affective_dict

if __name__ == "__main__":
    data_fname = sys.argv[1]
    data_list = read_data(data_fname) 

    CONCRETENESS = build_concreteness_dict('Concreteness_ratings_Brysbaert_et_al_BRM.txt')
    AFFECTIVE = build_affective_dict('Ratings_Warriner_et_al.csv')

    basename_ftrs = os.path.splitext(data_fname)[0]
    fname_ftrs = "%s_%d.csv" % (basename_ftrs, A)

    fd_ftrs = open(fname_ftrs, "w")
    data_list[0].write_header(fd_ftrs)

    increase = {}
    for d in data_list:
        t0 = time.time()
        d.extract_features()
        # d.extract_empath_cat()
        t1 = time.time()
        d.write_features(fd_ftrs)
        print("Extracting features...",t1-t0,"s")
        print()


    fd_ftrs.close()
    print("Number of features: ", len(data_list[0].get_ftrs_names()))
    print(len(data_list))
