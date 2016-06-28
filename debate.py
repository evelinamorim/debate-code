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

FSTPERSON_PRO = ["i", "we", "me", "us", "my", "mine", "our", "ours"]
FSTPERSON_PRO_PLURAL = ["we", "us", "our", "ours"]
SNDPERSON_PRO = ["you", "your", "yours"]

POS_TAG_SET = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', '$', "''", '(', ')', ',', '--', '.', ':']

stopwords_english = stopwords.words('english')

lexicon = Empath()
wordnet_lemmatizer = WordNetLemmatizer()
CONCRETENESS = {}

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

        return self.debate_ftrs_names() + self.arguments_ftrs_names()

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
        ftrs_names.append("nsentarg")
        ftrs_names.append("nsentcounter")
        ftrs_names.append("nwordsarg")
        ftrs_names.append("nwordscounter")

        ftrs_names.append("ndefarg")
        ftrs_names.append("firstpersonarg")
        ftrs_names.append("firstpersonpluralarg")
        ftrs_names.append("indefartarg")
        ftrs_names.append("sndpersonarg")
        ftrs_names.append("concretenessarg")

        for tt in self.ttoken_arg_count:
            if tt == ',':
                ftrs_names.append("virg_arg")
            elif tt == "''":
                ftrs_names.append("quote_arg")
            else:
                ftrs_names.append("%s_arg" % tt)

        ftrs_names.append("ndefcounter")
        ftrs_names.append("firstpersoncounter")
        ftrs_names.append("firstpersonpluralcounter")
        ftrs_names.append("indefartcounter")
        ftrs_names.append("sndpersoncounter")
        ftrs_names.append("concretenesscounter")

        for tt in self.ttoken_counter_count:
            if tt == ",":
                ftrs_names.append("virg_counter")
            elif tt == "''":
                ftrs_names.append("quote_counter")
            else:
                ftrs_names.append("%s_counter" % tt)

        ftrs_names.append("nwords_neg_arg")
        ftrs_names.append("nwords_pos_arg")
        ftrs_names.append("fracwords_pos_arg")

        ftrs_names.append("nwords_neg_counter")
        ftrs_names.append("nwords_pos_counter")
        ftrs_names.append("fracwords_pos_counter")

        ftrs_names.append("support_arg")
        ftrs_names.append("causal_arg")
        ftrs_names.append("complementary_arg")
        # ftrs_names.append("conclusion_arg")
        ftrs_names.append("conflict_arg")

        ftrs_names.append("support_counter")
        ftrs_names.append("causal_counter")
        ftrs_names.append("complementary_counter")
        # ftrs_names.append("conclusion_counter")
        ftrs_names.append("conflict_counter")

        ftrs_names.append("common_empath_cat")
        ftrs_names.append("ncommon_stopwords")
        ftrs_names.append("ncommon_voc")
        ftrs_names.append("jaccard_stopwords")
        ftrs_names.append("jaccard_words")

        ftrs_names.append("class")
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

             # number of sentences in arguments
             ftrs_arg.append(len(sent))
             # number of sentences in counterarguments
             ftrs_arg.append(len(sent_counter))

             #  number of words in arguments 
             ftrs_arg.append(len(arg_tags))
             #  number of words in couterarguments 
             ftrs_arg.append(len(counter_tags))


             # pos tag features for arguments
             ndef = 0 # definite article
             firstperson = 0
             firstpersonplural = 0
             indefart = 0 # indefinite article
             sndperson = 0
             type_tokens = {}
             concreteness_arg = 0
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
             t1 = time.time()
             # print("Extracting pos tag features argument", t1-t0, "s")

             concreteness_arg = concreteness_arg / len(arg_tags)

             ftrs_arg.append(ndef)
             ftrs_arg.append(firstperson)
             ftrs_arg.append(firstpersonplural)
             ftrs_arg.append(indefart)
             ftrs_arg.append(sndperson)
             ftrs_arg.append(concreteness_arg)

             t0 = time.time()
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

             t1 = time.time()
             # print("Extracting pos features counterargument", t1-t0, "s")
             concreteness_counter = concreteness_counter / len(counter_tags)

             ftrs_arg.append(ndef)
             ftrs_arg.append(firstperson)
             ftrs_arg.append(firstpersonplural)
             ftrs_arg.append(indefart)
             ftrs_arg.append(sndperson)
             ftrs_arg.append(concreteness_counter)

             t0 = time.time()
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

             # number of negative words in args
             ftrs_arg.append(cat_arg['negative_emotion'])
             # number of positive words in args 
             ftrs_arg.append(cat_arg['positive_emotion happiness'])
             # frac of positive words in args 
             ftrs_arg.append(cat_arg['positive_emotion happiness']/ len(arg_tags))

             # number of negative words in counter
             ftrs_arg.append(cat_counter['negative_emotion'])
             # number of positive words in counter
             ftrs_arg.append(cat_counter['positive_emotion happiness'])
             # frac of positive words in counter
             ftrs_arg.append(cat_counter['positive_emotion happiness']/ len(counter_tags))

             # Dicourse elemnts by Empath categories to argument 
             #: support, causal_argument, 
             # complementary, conclusion, conflict, support
             ftrs_arg.append(cat_arg['support'])
             ftrs_arg.append(cat_arg['causal_argument'])
             ftrs_arg.append(cat_arg['complementary'])
             # ftrs_arg.append(cat_arg['conclusion'])
             ftrs_arg.append(cat_arg['conflict'])

             # Dicourse elemnts by Empath categories to counterargument 
             #: support, causal_argument, 
             # complementary, conclusion, conflict, support
             ftrs_arg.append(cat_counter['support'])
             ftrs_arg.append(cat_counter['causal_argument'])
             ftrs_arg.append(cat_counter['complementary'])
             # ftrs_arg.append(cat_counter['conclusion'])
             ftrs_arg.append(cat_counter['conflict'])

             # common in empath
             common_cat = common_empath(cat_arg, cat_counter)
             ftrs_arg.append(len(common_cat))

             # common in stop words
             ncommon_stopwords = len(stopwords_arg.intersection(stopwords_counter))
             ftrs_arg.append(ncommon_stopwords)

             # common in content
             ncommon_words = len(voc_arg.intersection(voc_counter))
             ftrs_arg.append(ncommon_words)

             # jaccard stop words
             jaccard_stopwords = ncommon_stopwords / len(stopwords_counter.union(stopwords_counter))
             ftrs_arg.append(jaccard_stopwords)

             # jacaard in content
             jaccard_words = ncommon_words / len(voc_arg.union(voc_counter))
             ftrs_arg.append(jaccard_words)
             self.add_increase_for_class(ftrs_arg, typeteam)

             features.append(ftrs_arg)

        return features


    def extract_features(self):

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

        ftrs_args_pro = [ ftrs_debate + x for x in ftrs_args_pro]
        ftrs_args_against = [ ftrs_debate + x for x in ftrs_args_against]
        self.features_names = self.features_names + self.arguments_ftrs_names()
  
        self.features = ftrs_args_pro + ftrs_args_against

    def write_features(self, fd_ftrs):

        for arg in self.features:
            ftrs_line = functools.reduce(lambda x,y: str(x) + ',' + str(y), arg)
            fd_ftrs.write("%s\n" % ftrs_line)

    def write_header(self, fd_ftrs):
        ftrs_names = self.get_ftrs_names()
        header = functools.reduce(lambda x,y: x + ',' + y, ftrs_names)
        fd_ftrs.write("%s\n" % header)

    def add_increase_for_class(self, ftrs_arg, typeteam):

        before_for = 0        
        before_against = 0
        for (type_vote, p) in self.votes["before"]:
            if type_vote == "mildlyfor" or type_vote == "stronglyfor":
                before_for = int(p) + before_for
            if type_vote == "mildlyagainst" or type_vote == "stronglyagainst":
                before_against = int(p) + before_against

        after_for = 0        
        after_against = 0

        for (type_vote, p) in self.votes["after"]:
            if type_vote == "mildlyfor" or type_vote == "stronglyfor":
                after_for = int(p) + after_for
            if type_vote == "mildlyagainst" or type_vote == "stronglyagainst":
                after_against = int(p) + after_against

        if typeteam == "pro":
            if (after_for - before_for) > A:
                ftrs_arg.append(1)
                self.increase["pro"] = after_for-before_for
            else:
                ftrs_arg.append(0)

        if typeteam == "against":
            if (after_against - before_against) > A:
                ftrs_arg.append(1)
                self.increase["against"] = after_against-before_against
            else:
                ftrs_arg.append(0)

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

if __name__ == "__main__":
    data_fname = sys.argv[1]
    data_list = read_data(data_fname) 

    CONCRETENESS = build_concreteness_dict('Concreteness_ratings_Brysbaert_et_al_BRM.txt')

    basename_ftrs = os.path.splitext(data_fname)[0]
    fname_ftrs = "%s_%d.csv" % (basename_ftrs, A)

    fd_ftrs = open(fname_ftrs, "w")
    data_list[0].write_header(fd_ftrs)

    increase = {}
    for d in data_list:
        t0 = time.time()
        d.extract_features()
        t1 = time.time()
        d.write_features(fd_ftrs)
        print("Extracting features...",t1-t0,"s")
        print()

    fd_ftrs.close()
    print("Number of features: ", len(data_list[0].get_ftrs_names()))
    # print(len(data_list))
