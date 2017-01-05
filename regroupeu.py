# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:49:11 2016

@author: flap
"""

#librairies#
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


#PARAMÈTRES#
corpus_root = 'C:\\Users\\Flap\\Dropbox\\these\\'
nom_corpus_source = '.\\corpus_total_rendu.txt'
#result = open('result.txt', "w", encoding='utf-8')

brill_value = False
substitution_formes = True


ngram = True
nb_tranche = 10
ajout_corpus_neutre = False

unite = "MI"

#les 81 MI
"""                     
liste_signifiants = ['hein', 'heille', 'regarde', 'ostie', 'voyons',
                     'écoute', 'mon_dieu', 'crisse', 'wow', 'vraiment',
                     'tiens', 'sérieux', 'ayoye', 'mets-en', 'ark',
                     'seigneur', 'crif', 'ouf', 'pour_vrai', 'je_comprends',
                     'tabarnaque', 'câline', 'envoye', 'une_chance', 'oups',
                     'wô', 'arrête', 'c_est_clair', 'cibole', 'coudon', 
                     'crime', 'tabarnouche', 'pas_du_tout', 'franchement', 'oupellaille',
                     'pantoute', 'eh_boy', 'câlique', 'maudit', 'calvaire',
                     'chut', 'câlisse', 'go', 'ciboire', 'pas_vraiment',
                     'bof', 'fiou', 'vraiment_pas', 'merde', 'ostifie',
                     'super', 'tabarnache', 'simonaque', 'cool', 'ostique',
                     'baptême', 'bien_sûr', 'de_la_marde', 'sacrifice', 'du_tout',
                     'regarde_donc', 'aïe', 'calvince', 'cristie', 'sacre',
                     'sacrement', 'aïe_aïe_aïe', 'c_est_encore_drôle', 'câlif', 'tellement',
                     'torieu', 'malade', 'mautadit', 'tabarnique', 'viarge',
                     'bateau', 'let_s_go', 'ostine', 'youpi', 'zut',
                     'batinse', 'go_go_go', 'écoutez',  'regardez', 'arrêtez']
"""


#frequents 
"""
liste_signifiants = ['hein', 'heille', 'vraiment', 'regarde', 'ostie',
                     'tellement', 'écoute', 'voyons', 'super', 'crisse',
                     'mon_dieu', 'arrête', 'je_comprends', 'sérieux',
                     'pantoute', 'vraiment_pas', 'tiens', 'wow', 'bateau',
                     'pas_vraiment', 'pour_vrai', 'malade', 'cool', 'ayoye',
                     'mets-en', 'ark', 'crif', 'seigneur', 'maudit', 'ouf',
                     'tabarnaque', 'c_est_clair', 'une_chance', 'du_tout',
                     'envoye', 'câline', 'oups', 'tabarnouche', 'wô', 'pas_du_tout',
                     'de_la_marde', 'crime', 'cibole', 'franchement', 'coudon',
                     'oupellaille', 'câlisse', 'eh_boy', 'câlique', 'calvaire',
                     'chut', 'go', 'sacre', 'ciboire', 'merde', 'baptême', 'bof',
                     'fiou', 'ostifie', 'tabarnache', 'ostique', 'sacrifice']
"""                  

#sacres frequents
"""                     
liste_signifiants = ['ostie', 'crisse', 'bateau', 'crif', 'maudit',
                     'tabarnaque', 'câline', 'tabarnouche', 
                     'crime', 'cibole', 'câlisse', 'câlique', 'calvaire',
                     'sacre', 'ciboire', 'baptême', 'ostifie', 'tabarnache',
                     'ostique', 'sacrifice']
"""

# fréquents non sacres moins fiou
"""
liste_signifiants = ['hein', 'heille', 'vraiment', 'regarde', 
                     'tellement', 'écoute', 'voyons', 'super', 
                     'mon_dieu', 'arrête', 'je_comprends', 'sérieux',
                     'pantoute', 'vraiment_pas', 'tiens', 'wow', 
                     'pas_vraiment', 'pour_vrai', 'malade', 'cool', 'ayoye',
                     'mets-en', 'ark',  'seigneur',  'ouf',
                     'c_est_clair', 'une_chance', 'du_tout',
                     'envoye', 'oups', 'wô', 'pas_du_tout',
                     'de_la_marde', 'franchement', 'coudon',
                     'oupellaille', 'eh_boy', 
                     'chut', 'go', 'merde', 'bof'] # fiou bug
"""   

#frequents moins homos
"""
liste_signifiants = ['ostie', 'crisse', 'crif', 'tabarnaque', 'tabarnouche',
                     'crime', 'maudit', 'câlique', 'câlisse', 'ostifie', 'tabarnache',
                     'ostique', 'baptême', 'sacrifice', 'sacre',  'bateau',
                     'vraiment', 'tellement', 'regarde', 'super', 'arrête',
                     'écoute', 'vraiment_pas', 'je_comprends', 'malade',
                     'pantoute', 'pas_vraiment', 'cool', 'du_tout', 'de_la_marde',
                     'sérieux', 'pour_vrai', 'c_est_clair', 'tiens', 'pas_du_tout',
                     'une_chance', 'franchement', 'merde', 'envoye', 'seigneur']
"""

#sacres frequents moins homos
"""
liste_signifiants = ['ostie', 'crisse', 'crif', 'tabarnaque', 'tabarnouche',
                     'crime', 'maudit', 'câlique', 'câlisse', 'ostifie', 'tabarnache',
                     'ostique', 'baptême', 'sacrifice', 'sacre',  'bateau']
"""

#pas_Sacres_frequent moins homos
"""
liste_signifiants = ['vraiment', 'tellement', 'regarde', 'super', 'arrête',
                     'écoute', 'vraiment_pas', 'je_comprends', 'malade',
                     'pantoute', 'pas_vraiment', 'cool', 'du_tout', 'de_la_marde',
                     'sérieux', 'pour_vrai', 'c_est_clair', 'tiens', 'pas_du_tout',
                     'une_chance', 'franchement', 'merde', 'envoye', 'seigneur']
"""

#tous sacres
"""
liste_signifiants = ['ostie', 'ostique', 'ostifie', 'ostine',
                     'crisse', 'crif', 'crime', 'cristie',
                     'câlisse', 'câlique','câline', 'câlif',
                     'tabarnaque', 'tabarnache','tabarnouche', 'tabarnique',
                     'calvaire', 'calvince', 'ciboire', 'cibole',
                     'viarge', 'sacrement', 'sacre', 'sacrifice',
                     'simonaque', 'maudit', 'mautadit',
                     'batinse', 'baptême', 'bateau', 'torieu']
"""

#81 - sacres
"""                     
liste_signifiants = ['hein', 'heille', 'regarde', 'voyons',
                     'écoute', 'mon_dieu', 'wow', 'vraiment',
                     'tiens', 'sérieux', 'ayoye', 'mets-en', 'ark',
                     'seigneur', 'ouf', 'pour_vrai', 'je_comprends',
                     'envoye', 'une_chance', 'oups',
                     'wô', 'arrête', 'c_est_clair', 'coudon', 
                     'pas_du_tout', 'franchement', 'oupellaille',
                     'pantoute', 'eh_boy', 
                     'chut', 'go', 'pas_vraiment',
                     'bof', 'fiou', 'vraiment_pas', 'merde',
                     'super',  'cool',
                     'bien_sûr', 'de_la_marde', 'du_tout',
                     'regarde_donc', 'aïe',
                     'aïe_aïe_aïe', 'c_est_encore_drôle', 'tellement',
                     'malade', 'let_s_go','youpi', 'zut',
                     'go_go_go', 'écoutez',  'regardez', 'arrêtez']
"""




#liste_signifiants = ['envoye', 'écoute', 'écoutez',
#                    'regarde', 'regardez', 'arrête', 'arrêtez', 'tiens']


#liste_signifiants = ['envoye', 'écoute',
 #                   'regarde', 'arrête', 'tiens']

liste_signifiants = ['tellement', 'franchement', 'vraiment', 'pour_vrai', 'sérieux']
         
#liste_signifiants = ['je_comprends', 'une_chance', 'c_est_clair', 'mets-en', 'vraiment']
   
#liste_signifiants = ['tellement', 'franchement', 'vraiment', 
 #                    'pas_du_tout', 'pantoute', 'pas_vraiment', 'vraiment_pas', 'du_tout']

#liste_signifiants = ['pas_du_tout', 'pantoute', 'pas_vraiment', 'vraiment_pas', 'du_tout', 'bof']

#liste_signifiants = ['sacre', 'sacrement', 'sacrifice']

#liste_signifiants = ['malade','cool', 'ayoye', 'wow', 'eh boy', 'super', 'mon_dieu']

#liste_signifiants =  ['envoye', 'écoute', 'écoutez',
 #                    'regarde', 'regardez', 'arrête', 'arrêtez', 'tiens',
  #                   'tellement', 'franchement', 'vraiment', 'pour_vrai',
   #                  'pas_du_tout', 'pantoute', 'pas_vraiment', 'vraiment_pas', 'du_tout',
    #                 'je_comprends', 'une_chance', 'c_est_clair',
     #                'super','malade','cool']



liste_regroupements = []
liste_fmesure = []
liste_M_rep_corrects = []
liste_M_rep = []
liste_M_tot = []


def modifie(text, sub):


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


def regroupeur(liste_signifiants, liste_regroupements):
    
    liste_signifiants = list(liste_signifiants)    
    
    if liste_signifiants:
        liste_regroupements.append(list(liste_signifiants))

    
    try:
        liste_signifiants.pop()
    except:
        return 0

    regroupeur(liste_signifiants, liste_regroupements)

def create_tagger(train_sents):

    if ngram is True:
        
        
        t0 = nltk.DefaultTagger('S')
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.BigramTagger(train_sents, backoff=t1)
        t3 = nltk.TrigramTagger(train_sents, backoff=t2)
            
        # key=str.lower 
            
        if brill_value is True:
            Template._cleartemplates()
            
            templates = [ #REDUIT#
                    Template(Word([0]), Word([-1])),
                    Template(Word([0]), Word([1])),
                    ]
            
            t4 = BrillTaggerTrainer(t3, templates, trace=3)

            
            tagger = t4.train(train_sents, max_rules=20, min_score=0, min_acc=None)
        else:
            tagger = t3
    
    return tagger

def main(unite):

    score = 0
    erreur = 0
    total_M = 0        
    M_reperes = 0
    M_reperes_corrects = 0
    
    #separation du corpus neutre#
    
     
    nom_corpus = '.\/corpus.txt'
    nom_corpus_neutre = '.\/corpus_neutre.txt'
    corpus = open (nom_corpus, "w", encoding='utf-8')
    corpus_neutre = open (nom_corpus_neutre, "w", encoding='utf-8')
   
            
    ### Lecture du corpus ###
    text = None
    file = open (nom_corpus_source, "r", encoding='utf-8')    
    text = file.readlines()
    file.close()
    ###
    
    text = modifie(text, sub)       
    
    for line in text:
        if re.search(' ' + unite + '\/', line) or re.match('^' + unite + '\/', line):
            corpus.write(line)
        else:
            corpus_neutre.write(line)
    
    
    corpus.close()
    corpus_neutre.close()
    

    
    for tranche in range(nb_tranche):
        
        corpus = open(nom_corpus, "r", encoding='utf-8')        
        
#        print ( '\nTranche ' + str(tranche+1) + ' Correct \t= \tTag\n')        
        
        nom_ent = 'resultats\/corpus_entrainement' + str(tranche+1) + '.txt'
        nom_tes = 'resultats\/corpus_test' + str(tranche+1) + '.txt'
   #     nom_res = 'resultats\/resultats' + str(tranche+1) + '.txt'
        corpus_entrainement = open(nom_ent, "w", encoding='utf-8')       
        corpus_test = open(nom_tes, "w", encoding='utf-8')
#        fichier_sortie = open (nom_res, "w", encoding='utf-8')

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
        
#        sortie = sents_tagges
#        sortie = "\n".join(str(x) for x in sortie)
#        fichier_sortie.write(sortie)
        
        
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
  
        #print (resultat)
        
   #     unites = score+erreur        
        """   
        print ("\nUnites = " + str(unites))
        print ("Taux réussite = " + str(score/unites))
        print ("\nScore = " + str(score))
        print ("Erreur = " + str(erreur))

        print ('M totaux =' + str(total_M))
        print ('M reperes_corrects =' + str(M_reperes_corrects))        
        print ('M reperes =' + str(M_reperes))        
        """
        
        fmesure = 0        
        
        if total_M:
            if M_reperes:
                if M_reperes_corrects:
                    
                    precision = M_reperes_corrects / M_reperes
                    rappel = M_reperes_corrects / total_M
                    fmesure = 2*(precision*rappel)/(precision+rappel)        
                    
                    #print ('Precision = ' + str(precision))
                    #print ('Rappel = ' + str(rappel))
                  #  print ('f-mesure = ' + str(fmesure))
        
                   
                    
                    if tranche == 9:
                        """
                        result.write("\n\n" + str(sub) + "  "
                        + str(brill_value)
                        + "  " + str(substitution_formes)
                        + "\nFormes=" + str(unites)  
                        + "\t M=" + str(total_M) + "\tS=" + str(unites-total_M)
                        + "\t  " + str(total_M/unites)
                        + "\n"
                        + "M_reperes=" + str(M_reperes) + "\tM_bons=" + str(M_reperes_corrects)
                        + "  Precision=" + str(precision)
                        + "  Rappel=" + str(rappel)
                        + "  f_mesure=" + str(fmesure))
                        """
                        print ('f-mesure = ' + str(fmesure))

        if tranche == 9:
            liste_fmesure.append(fmesure)
            liste_M_rep_corrects.append(M_reperes_corrects)
            liste_M_rep.append(M_reperes)
            liste_M_tot.append(total_M)
                    
        
#        fichier_sortie.close()  
        corpus_entrainement.close()
        corpus_test.close()   
            
        corpus.close()  
        


for signifiant in liste_signifiants:
    
    liste_signifiants = list(liste_signifiants)
    
    regroupeur(liste_signifiants, liste_regroupements)
    liste_signifiants.remove(signifiant)


print ("Fin")
print (liste_regroupements)

for regroupement in liste_regroupements:
    if regroupement:
        sub = regroupement
        try:
            main(unite)
        except:
            print("erreur")
            liste_fmesure.append(0)
            liste_M_rep_corrects.append(0)
            liste_M_rep.append(0)
            liste_M_tot.append(0)
#result.close()

tuple_scores = list(zip(liste_regroupements, liste_fmesure, liste_M_rep_corrects, liste_M_rep, liste_M_tot))

tuple_scores.sort(key = lambda row: row[1])

dict_simplets = {}
list_winners = []

for couple in tuple_scores:
    if len(couple[0]) == 1:
        print(couple[0][0])
        add = {couple[0][0] :  [couple[1], couple[2], couple[3], couple[4]]}
        dict_simplets.update(add)
    
print(dict_simplets.items())



for couple in tuple_scores:
    valeur_vocable = 0
    valeur_groupe = 0   
    total_M_cor = 0
    total_M_rep = 0
    total_M_tot = 0    
    
    for vocable in couple[0]:
        total_M_cor = total_M_cor + dict_simplets[vocable][1]
        total_M_rep = total_M_rep + dict_simplets[vocable][2]
        total_M_tot = total_M_tot + dict_simplets[vocable][3]
    
    try:
                    
        precision_groupe = total_M_cor / total_M_rep
        rappel_groupe = total_M_cor / total_M_tot
        
        if precision_groupe and rappel_groupe:
            valeur_groupe = 2*(precision_groupe*rappel_groupe)/(precision_groupe+rappel_groupe)
        
        if couple[1] > valeur_groupe:
            print(couple[0],valeur_groupe, " vs ", couple[1], "winner by", couple[1]-valeur_groupe)
            list_winners.append([couple[0], couple[1]-valeur_groupe])
    except:
        print("erreur")
        

list_winners.sort(key = lambda row: row[1])
print("LISTE WINNERS", list_winners)
            
            

print("fin")
print(tuple_scores)
