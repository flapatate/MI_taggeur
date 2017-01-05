# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 08:47:11 2015

freqDist NLTK utilise collections.Counter qui retourne de facon arbitraire

Modification du fichier sequential.py dans le NLTK, ligne 195:

            #hack#gets rid of the arbitrary choice between equals
            
            bt = 0
            for c in fd[context]:
                if fd[context][c] == fd[context][best_tag]:
                    bt = bt + 1
                      
            if bt==1:
                hits = fd[context][best_tag]
                if hits > cutoff:
                    self._context_to_tag[context] = best_tag
                    hit_count += hits
            else:
                print("Tie breaker : %s,  %d" % (best_tag, bt))
               

            #hack#


@author: flap
"""

import nltk
from nltk.corpus import TaggedCorpusReader
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word
from nltk.tag import RegexpTagger, BrillTaggerTrainer
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.util import unique_list
import nltk.tag.hmm
import os, re, sys
from nltk.tag import brill
from pickle import dump
from nltk.tag import CRFTagger

##parametres##

corpus_root = 'C:\\Users\\Flap\\Dropbox\\these\\'
#corpus_root = 'C:\\Users\\Utrilio\\Dropbox\\these\\'

#nom_corpus_source = '.\\corpus_total_sacres_test.txt'
#nom_corpus_source = '.\\corpus_total_sacres_PRO.txt'
#nom_corpus_source = '.\\corpus_total_sacres_DE_S.txt'
#nom_corpus_source = '.\\corpus_total_sacres_A.txt'
#nom_corpus_source = '.\\corpus_total_regarde.txt'

nom_corpus_source = '.\\corpus_total_rendu.txt'
#nom_corpus_source = '.\\corpus_total_rendu_P_M.txt'

#nom_corpus_source = '.\\corpus_total_rendu_1_j.txt'

#nom_corpus_source = '.\\corpus_total_du_tout.txt'
#nom_corpus_source = '.\\corpus_total_phrasemes2.txt'
#nom_corpus_source = '.\\corpus_total_y_pro.txt'
#nom_corpus_source = '.\\corpus_total_rendu.txt_pre_VOUS_PRO'

result = open('result.txt', "w", encoding='utf-8')


brill_value = False
substitution_formes = True
liste_signifiants = False

ngram = True
HMM = False
CRF = False
nb_tranche = 10
ajout_corpus_neutre = False


#unite = 'hein'
#unite = 'heille'
#unite = 'regarde'
#unite = 'ostie'
#unite = 'voyons'
#unite = 'écoute'
#unite = 'crisse'
#unite = 'mon_dieu'
#unite = 'wow'
#unite = 'tabarnaque'
#unite = 'câlisse'
#unite = 'vraiment'
#unite = 'tiens'
#unite = 'sérieux'
#unite = 'ayoye'
#unite = 'mets-en'
#unite = 'ark'
#unite = 'seigneur'
#unite = 'ouf'
#unite = 'crif'
#unite = 'pour_vrai'
#unite = 'je_comprends'
#unite = 'câline'
#unite = 'envoye'
#unite = 'une_chance'
#unite = 'oups'
#unite = 'wô'
#unite = 'arrête'
#unite = 'c_est_clair'
#unite = 'cibole'
#unite = 'coudon'
#unite = 'crime'
#unite = 'tabarnouche'
#unite = 'franchement'
#unite = 'oupelaille'
#unite = 'pantoute'
#unite = 'pas_du_tout'
#unite = 'adjectifs'
#unite = 'eh_boy'
#unite = 'câlique'
#unite = 'maudit'
#unite = 'calvaire'
#unite = 'chut'
#unite = 'câlisse'
#unite = 'go'
#unite = 'ciboire'
#unite = 'pas_vraiment'
#unite = 'bof'
#unite = 'fiou'
#unite = 'vraiment_pas'
#unite = 'merde'
#unite = 'ostifie'
#unite = 'super'
#unite = 'tabarnache'
#unite = 'simonaque' # bug#
#unite = 'cool'
#unite = 'ostique'
#unite = 'baptême'
#unite = 'bien_sûr' #bug#
#unite = 'de_la_marde'
#unite = 'sacrifice'
#unite = 'du_tout'
#unite = 'regarde_donc' #bug#
#unite = 'aïe' #bug#
#unite = 'calvince' #bug#
#unite = 'cristie'
#unite = 'sacre'
#unite = 'sacrement' #bug#
#unite = 'aïe_aïe_aïe'
#unite = 'câlif'
#unite = 'tellement'
#unite = 'malade'
#unite = 'mautadit'
#unite = 'bateau'

#unite = 'maudit_baptême'
#unite = 'baptême_ostique'
#unite = 'super_cool'
#unite = 'malade_cool'
#unite = 'du_marde'
#unite = 'bateau_coudon'
#unite = 'mon_dieu_seigneur'
#unite = 'maudit_mautadit'

#unite = 'tous_sacres'
#unite = 'verbes'
#unite = 'adverbes'

#unite = 'infirmatif'
#unite = 'affirmatifs'
#unite = 'adjectifs'

unite = 'sacre_sacrifice'


####

#liste_signifiants = ['super','malade','cool']

#liste_signifiants = ['pour_vrai','vraiment', 'franchement', 'tellement']

#liste_signifiants = ['ostie','crisse','tabarnaque']

#liste_signifiants = ['hein','heille','wow',
 #                    'ark', 'ouf', 'oups', 'wô',
#                    'fiou', 'aïe',
#                    'zut', 'bof']

#liste_signifiants = ['envoye', 'regarde',
#                     'écoute', 'arrête']

#liste_signifiants = ['pas_du_tout', 'pantoute', 'pas_vraiment', 'vraiment_pas',
#                     'du_tout', 'c_est_encore_drole']

  
def modifie(text):
    #### Uniformisation des formes ###

    if unite == 'ostie':
        sub = ['ostie', 'ostique','ostifie', 'ostine']
     
    if unite == 'crisse':
        sub = ['crif','crime', 'cristie', 'crisse']
        
    if unite == 'câlisse':
        sub = ['câlique','câline'] # 'câlif']
        
    if unite == 'tabarnaque':
        sub = ['tabarnache','tabarnouche'] # 'tabarnique']

    if unite == 'baptême':
        sub = ['batinse'] #bateau

    if unite == 'tous_sacres':
        sub = ['ostie', 'ostique','ostifie', 'ostine',
                'crisse', 'crif','crime', 'cristie',
                'câlisse', 'câlique','câline', 'câlif',
                'tabarnaque', 'tabarnache','tabarnouche', 'tabarnique',
                'calvaire', 'calvince', 'ciboire', 'cibole',
                'viarge', 'sacrement', 'sacre', 'sacrifice',
                'simonaque',
                'baptême', 'torieu', 'mautadit',
                'batinse'] #bateau

    if unite == 'sacres_frequents':
        sub = ['ostie', 'crisse', 'tabarnaque', 'câlisse', 'baptême']
  

    if unite == 'maudit_baptême':
        sub = ['maudit', 'baptême']
        
    if unite == 'baptême_ostique':
        sub = ['baptême', 'ostique']
                
    if unite == 'verbes':
        sub = ['écoute', 'écoutez',
               'regarde', 'regardez', 'arrête', 'arrêtez', 'tiens']   

    if unite == 'adjectifs':
        sub = ['super', 'cool', 'malade', 'sérieux']   
            
    if unite == 'malade_cool':
        sub = ['malade', 'cool']
        
    if unite == 'du_marde':
        sub = ['du_tout', 'de_la_marde']
        
    if unite == 'adverbes':
        sub = ['tellement', 'franchement', 'vraiment', 'pour_vrai']   
 
    if unite == 'maudit_mautadit':
        sub = ['maudit', 'mautadit']

    if unite == 'pv_s':
        sub = ['pour_vrai', 'sérieux']
        
    if unite == 'regarde':
        sub = ['regarde', 'regardez']
        
    if unite == 'écoute':
        sub = ['écoute', 'écoutez']
        
    if unite == 'arrête':
        sub = ['arrête', 'arrêtez']
        
    if unite == 'infirmatif':
        sub = ['pas_du_tout', 'pantoute', 'pas_vraiment', 'vraiment_pas', 'du_tout', 'c_est_encore_drole']

        
    if unite == 'affirmatifs':
        sub = ['je_comprends', 'une_chance', 'c_est_clair']
        
    if unite == 'bateau_coudon':
        sub = ['coudon', 'bateau']
        
    if unite == 'mon_dieu_seigneur':
        sub = ['mon_dieu', 'seigneur']
          
          
    if unite == 'sacre_sacrifice':
        sub = ['sacre', 'sacrement']
#'maudit', 'mautadit'                
                
  #  os.remove('corpus_modifie.txt')
    corpus_modifie = open ('.\\corpus_modifie.txt', "w", encoding='utf-8')       

    for line in text:
        for forme in sub:
            line = re.sub (' ' + forme + '\/', ' ' + unite + '/', line)
        corpus_modifie.write(line)
    corpus_modifie.close()

    corpus_modifie = open ('.\\corpus_modifie.txt', "r", encoding='utf-8')   
    text = corpus_modifie.readlines()
    corpus_modifie.close()
    return text    



def main(unite):

    score = 0
    erreur = 0
    total_M = 0        
    M_reperes = 0
    M_reperes_corrects = 0
    
    #separation du corpus neutre#
    
    #text = PlaintextCorpusReader(corpus_root, nom_corpus_total)
    
    nom_corpus = '.\/corpus.txt'
    nom_corpus_neutre = '.\/corpus_neutre.txt'
    
  #  os.remove(nom_corpus)
    corpus = open (nom_corpus, "w", encoding='utf-8')
    
 #   os.remove(nom_corpus_neutre)
    corpus_neutre = open (nom_corpus_neutre, "w", encoding='utf-8')
   
            
    ### Lecture du corpus ###
    text = None
    file = open (nom_corpus_source, "r", encoding='utf-8')    
    text = file.readlines()
    file.close()
    ###
    
    
    if substitution_formes:
        text = modifie(text)       
    
    for line in text:
        if re.search(' ' + unite + '\/', line) or re.match('^' + unite + '\/', line):
            corpus.write(line)
        else:
            corpus_neutre.write(line)
    
    
    corpus.close()
    corpus_neutre.close()
    

    
    for tranche in range(nb_tranche):
        
        corpus = open(nom_corpus, "r", encoding='utf-8')        
        
        print ( '\nTranche ' + str(tranche+1) + ' Correct \t= \tTag\n')        
        
        nom_ent = 'resultats\/corpus_entrainement' + str(tranche+1) + '.txt'
        nom_tes = 'resultats\/corpus_test' + str(tranche+1) + '.txt'
        nom_res = 'resultats\/resultats' + str(tranche+1) + '.txt'
        corpus_entrainement = open(nom_ent, "w", encoding='utf-8')       
        corpus_test = open(nom_tes, "w", encoding='utf-8')
        fichier_sortie = open (nom_res, "w", encoding='utf-8')

        line_nb = tranche
       
        for line in corpus:
            line_nb = line_nb + 1
            if line_nb%nb_tranche == 0:
                corpus_test.write(line)
            else:
                corpus_entrainement.write(line)

        if ajout_corpus_neutre is True:
            corpus_neutre = open(nom_corpus_neutre, "r", encoding='utf-8')
            for line in corpus_neutre:
                corpus_entrainement.write(line)
   

        corpus_entrainement.close()
        corpus_test.close() 
   
        corpus_entrainement_tuple = TaggedCorpusReader(corpus_root, nom_ent)
        corpus_test_tuple = TaggedCorpusReader(corpus_root, nom_tes)

        train_sents = corpus_entrainement_tuple.tagged_sents()
#        test_sents = corpus_test_tuple.tagged_sents()

        tagger = None
        tagger = create_tagger(train_sents)
        
        """
        #Pour archiver les taggers#
        output = open('resultats\/tagger' + str(tranche+1) + '.txt', 'wb')
        dump(tagger, output, -1)
        output.close()     
        """     
        
        sents_corrects = corpus_test_tuple.tagged_sents()
        sents_tagges = tagger.tag_sents(corpus_test_tuple.sents())
        
        sortie = sents_tagges
        sortie = "\n".join(str(x) for x in sortie)
        fichier_sortie.write(sortie)
        
        
        # mots_corrects sents_tagges
        
        nb = 0
        resultat = ""
        
        for sent_correct, sent_tagge in zip(sents_corrects, sents_tagges):
            for mot_correct, mot_tagge in zip(sent_correct, sent_tagge) :
                    if mot_correct[0] == unite:
                        nb = nb+1
                        resultat += str(nb) + '-\t' + mot_correct[1] + "\t=\t" + mot_tagge[1] + "\n"
                        
                        if mot_correct[1] == 'M':
                            total_M = total_M+1                        
                        
                        if mot_tagge[1] == 'M':
                            M_reperes = M_reperes+1
                            
                            if mot_correct[1] == mot_tagge[1]:
                                M_reperes_corrects = M_reperes_corrects+1
                        
                        if mot_correct[1] == mot_tagge[1]:
                            score = score+1
                        else:
                            erreur = erreur+1
                            resultat += str(sent_tagge) + "\n"
  
        print (resultat)
        
        unites = score+erreur        
        
        print ("\nUnites = " + str(unites))
        print ("Taux réussite = " + str(score/unites))
        print ("\nScore = " + str(score))
        print ("Erreur = " + str(erreur))

        print ('M totaux =' + str(total_M))
        print ('M reperes_corrects =' + str(M_reperes_corrects))        
        print ('M reperes =' + str(M_reperes))        
        
        
        if total_M:
            if M_reperes:
                if M_reperes_corrects:
                    
                    precision = M_reperes_corrects / M_reperes
                    rappel = M_reperes_corrects / total_M
                    fmesure = 2*(precision*rappel)/(precision+rappel)        
        
                    print ('Precision = ' + str(precision))
                    print ('Rappel = ' + str(rappel))
                    print ('f-mesure = ' + str(fmesure))
        
        
                    print ("\n" + str(sys.getsizeof(tagger)))
                    print (tagger)   
                    
                    if tranche == 9:
                        result.write("\n\n" + unite + "  "
                        + str(brill_value)
                        + "  " + str(substitution_formes)
                        + "  " + str(liste_signifiants)  
                        + "\nFormes=" + str(unites)  
                        + "\t M=" + str(total_M) + "\tS=" + str(unites-total_M)
                        + "\t  " + str(total_M/unites)
                        + "\n"
                        + "M_reperes=" + str(M_reperes) + "\tM_bons=" + str(M_reperes_corrects)
                        + "  Precision=" + str(precision)
                        + "  Rappel=" + str(rappel)
                        + "  f_mesure=" + str(fmesure))

        
        fichier_sortie.close()  
        corpus_entrainement.close()
        corpus_test.close()   
            
        corpus.close()  

def create_tagger(train_sents):

    if HMM is True: 
        
        corpustotal = TaggedCorpusReader(corpus_root, 'corpus_total_sacres_ecoute.txt', encoding='utf8')
        
     
        corpustagge = corpustotal.tagged_sents()
        
        
        tag_set = unique_list(tag for sent in corpustagge for (word,tag) in sent)
        symbols = unique_list(word for sent in corpustagge for (word,tag) in sent)
        t5 = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)

        tagger = t5.train_supervised(train_sents)

    if ngram is True:
        
        
        t0 = None
        t1 = None
        t2 = None
        t3 = None
        t4 = None        
        
        t0 = nltk.DefaultTagger('S')
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.BigramTagger(train_sents, backoff=t1)
        t3 = nltk.TrigramTagger(train_sents, backoff=t2)
            
        # key=str.lower 
            
        if brill_value is True:
            Template._cleartemplates()
        
            """
            templates = [ #BIG#
                    Template(Word([0]), Word([1]), Word([2])),
                    Template(Word([-1]), Word([0]), Word([1])),
                    Template(Word([0]), Word([-1])),
                    Template(Word([0]), Word([1])),
                    Template(Word([0]), Word([2])),
                    Template(Word([0]), Word([-2])),
                    Template(Word([1, 2])),
                    Template(Word([-2, -1])),
                    Template(Word([1, 2, 3])),
                    Template(Word([-3, -2, -1])),
                    Template(Word([0]), Pos([2])),
                    Template(Word([0]), Pos([-2])),
                    Template(Word([0]), Pos([1])),
                    Template(Word([0]), Pos([-1])),
                    
                    Template(Word([0])),
                    Template(Word([-2])),
                    Template(Word([2])),
                    Template(Word([1])),
                    Template(Word([-1])),
                                  
                    Template(Pos([-1]), Pos([1])),
                    Template(Pos([1]), Pos([2])),
                    Template(Pos([-1]), Pos([-2])),
                    Template(Pos([1])),
                    Template(Pos([-1])),
                    Template(Pos([-2])),
                    Template(Pos([2])),
                    Template(Pos([1, 2, 3])),
                    Template(Pos([1, 2])),
                    Template(Pos([-3, -2, -1])),
                    Template(Pos([-2, -1])),
                    Template(Pos([1]), Word([0]), Word([1])),
                    Template(Pos([1]), Word([0]), Word([-1])),
                    Template(Pos([-1]), Word([-1]), Word([0])),
                    Template(Pos([-1]), Word([0]), Word([1])),
                    Template(Pos([-2]), Pos([-1])),
                    Template(Pos([1]), Pos([2])),
                    Template(Pos([1]), Pos([2]), Word([1]))                 
                    ]

            """
            
            templates = [ #REDUIT#
            #        Template(Word([0]), Word([1]), Word([2])),
            #        Template(Word([-1]), Word([0]), Word([1])),
                    Template(Word([0]), Word([-1])),
                    Template(Word([0]), Word([1])),
               #     Template(Word([0]), Pos([1])),
             #       Template(Word([0]), Word([2])),
            #        Template(Word([0]), Word([-2])),
             #       Template(Word([1, 2])),
             #       Template(Word([-2, -1])),
             #       Template(Word([1, 2, 3])),
             #       Template(Word([-3, -2, -1])),
              #      Template(Word([0]), Pos([2])),
            #        Template(Word([0]), Pos([-2])),
            #        Template(Word([0]), Pos([1])),
             #       Template(Word([0]), Pos([-1])),
                    
            #        Template(Word([0])),
            #        Template(Word([-2])),
            #        Template(Word([2])),
            #        Template(Word([1])),
            #        Template(Word([-1])),
                                  
            #        Template(Pos([-1]), Pos([1])),
            #        Template(Pos([1]), Pos([2])),
            #        Template(Pos([-1]), Pos([-2])),
            #        Template(Pos([1])),
           #         Template(Pos([-1])),
           #         Template(Pos([-2])),
           #         Template(Pos([2])),
            #        Template(Pos([1, 2, 3])),
            #        Template(Pos([1, 2])),
            #        Template(Pos([-3, -2, -1])),
           #        Template(Pos([-2, -1])),
            #        Template(Pos([1]), Word([0]), Word([1])),
            #        Template(Pos([1]), Word([0]), Word([-1])),
            #        Template(Pos([-1]), Word([-1]), Word([0])),
            #        Template(Pos([-1]), Word([0]), Word([1])),
             #       Template(Pos([-2]), Pos([-1])),
             #       Template(Pos([1]), Pos([2])),
             #       Template(Pos([1]), Pos([2]), Word([1]))                 
                    ]
            
            t4 = BrillTaggerTrainer(t3, templates, trace=3)

            
            tagger = t4.train(train_sents, max_rules=20, min_score=0, min_acc=None)
        else:
            tagger = t3
    
    if CRF is True:
        ct = CRFTagger()
        ct.train(train_sents,'model.crf.tagger')
        tagger = ct
    
    
    return tagger
    
if liste_signifiants:
    for unite in liste_signifiants:
        main(unite)
else:
    main(unite)
    brill_value = True
    main(unite)

result.close()