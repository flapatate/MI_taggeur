# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 08:01:33 2017

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
import re, sys
from nltk.tag import brill
from pickle import dump
from nltk.tag import CRFTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

#PARAMETRES#

#FICHIERS#
dossier_racine = 'C:\\Users\\Flap\\Dropbox\\these\\'
fichier_corpus = '.\\corpus_total_rendu.txt'

#fichier_unites = '.\\liste_signifiants.txt'
#fichier_unites = '.\\liste_signifiants_fam_crisse.txt'
#fichier_unites = '.\\liste_fam_ostie.txt'
#fichier_unites = '.\\liste_pourvrai.txt'
#fichier_unites = '.\\liste_vraiment.txt'
#fichier_unites = '.\\liste_random.txt'
#fichier_unites = '.\\liste_brill.txt'


#fichier_unites_train = '.\\liste_signifiants.txt'
fichier_unites_train = '.\\liste_ambigus.txt'


fichier_unites_test = '.\\liste_ambigus.txt'
#fichier_unites_test = '.\\liste_signifiants.txt'


fichier_rapport = open('rapport.txt', "w", encoding='utf-8')

#GLOBAUX#
tranches = 10
scores_ngram = []
scores_SVM = []
scores_total = {}

#Parametres des dictionnaires a vectoriser#
signifiant = True #neutre

ngram_tag = True #bon
mot_suivant = True #bon
tag_suivant = True #bon
tag_precedent = True #bon

intonation_apres = False #inutile?
de_apres = False  #inutile?
la_apres = False  #inutile?

mot_precedent = False #nefaste?
mot_precedent2 = False #nefaste?
mot_precedent3 = False #nefaste?

tag_2_avant = False #nefaste?
tag_3_avant = False #nefaste?



def main():
    
    ititialisation_des_scores()
    
    circonscrire()
    
    #for tranche in range(tranches):
    for tranche in range(10):
        
        message('Tranche ' + str(tranche+1))
                     
        division_corpus(tranche)
        
        #analyse_ngram(tranche)
        #message(calcul_ngram())
    
        analyse_SVM(tranche)
        message(calcul_SVM())
        
    #message("\nScores_ngram")
    #message(calcul(scores_ngram))
    
    #message("\nScores_SVM")
    #message(calcul(scores_SVM))
    

def message(texte):
    print("\n" + texte)
    fichier_rapport.write("\n" + texte)

    
def ititialisation_des_scores():
    unites = open (fichier_unites_test, "r", encoding='utf-8')
    liste_unites = unites.readlines()
    
    for unite in liste_unites:
        
        entree = {'signifiant': unite.rstrip(),
                  'total_signifiant':0,
                  'total_MI':0,
                  'MI_reperes':0,
                  'MI_corrects':0
                 }
        
        scores_ngram.append(entree)
        scores_SVM.append(entree)
    
    unites.close()
    
def circonscrire():

    message("*****Début de la circonscription.")

    corpus = open (fichier_corpus, "r", encoding='utf-8')   

    unites = open (fichier_unites_train, "r", encoding='utf-8')
    liste_unites = unites.readlines()
    
    corpus_circonscrit = open ('resultats\/corpus_circonscrit.txt', "w", encoding='utf-8')
    

    
    for line in corpus:
        ligne_pertinente = False

        for unite in liste_unites:
            if re.search(' ' + unite.rstrip() + '\/', line):
                ligne_pertinente = True
               
        if ligne_pertinente:
            corpus_circonscrit.write(line)
    
            
    corpus.close()
    unites.close()
    corpus_circonscrit.close()
            
    message("Fin de la circonscription.*****")
    
    
def division_corpus(tranche):
    
    message("*****Début de la division du corpus.")
    
    corpus = open('resultats\/corpus_circonscrit.txt', "r", encoding='utf-8')
    corpus_entrainement = open('resultats\/corpus_entrainement' + str(tranche+1) + '.txt', "w", encoding='utf-8')       
    corpus_test = open('resultats\/corpus_test' + str(tranche+1) + '.txt', "w", encoding='utf-8')
    
    line_nb = tranche
       
    for line in corpus:
        line_nb = line_nb + 1
        if line_nb%tranches == 0:
            corpus_test.write(line)
        else:
            corpus_entrainement.write(line)
    
    corpus.close()        
    corpus_entrainement.close()
    corpus_test.close() 
    
    message("Fin de la division du corpus.*****")
    

def analyse_ngram(tranche):
    
    message("*****Début de l'analyse NGRAM.")
       
    corpus_entrainement_tuple = TaggedCorpusReader(dossier_racine, 'resultats\/corpus_entrainement' + str(tranche+1) + '.txt')
    corpus_test_tuple = TaggedCorpusReader(dossier_racine, 'resultats\/corpus_test' + str(tranche+1) + '.txt')

    
    train_sents = corpus_entrainement_tuple.tagged_sents()
    
    tagger = None
    tagger = create_tagger(train_sents)
    
    sents_corrects = corpus_test_tuple.tagged_sents()
    sents_tagges = tagger.tag_sents(corpus_test_tuple.sents())
    
    for sent_correct, sent_tagge in zip(sents_corrects, sents_tagges):
        phrase_combine = [(mot_correct, mot_tagge) for mot_correct, mot_tagge in zip(sent_correct, sent_tagge)]
        
        for couple in phrase_combine:
            
            for MI in scores_ngram:
                if MI['signifiant'] == couple[0][0]:
                    MI['total_signifiant'] += 1

                    if couple[0][1] == 'M':
                        MI['total_MI'] += 1

                    if couple[1][1] == 'M':
                        MI['MI_reperes'] += 1

                        if couple[1][1] == couple [0][1]:
                            MI['MI_corrects'] += 1

                    #if couple[0][1] != couple[1][1]:
                        #print(phrase_combine)
            
    
    message("Fin de l'analyse NGRAM.*****")
    
    
def analyse_SVM(tranche):
    
    message("*****Début de l'analyse SVM.")

    global scores_SVM
                
    
    #nb_MI = 0
    #MI_tagges = 0
    #MI_trouves = 0
    
 
    ###Preparation des dicts de features###
    #On va chercher les resustats 
    corpus_entrainement_tuple = TaggedCorpusReader(dossier_racine, 'resultats\/corpus_entrainement' + str(tranche+1) + '.txt')
    
    
    train_sents = corpus_entrainement_tuple.tagged_sents()      
    tagger = None
    tagger = create_tagger(train_sents)

    liste_dictionnaires = []    
    liste_y = []    
    
    ###CONSTRUCTION DU DICTIONNAIRE ENTRAINEMENT###
    corpus_test_tuple = TaggedCorpusReader(dossier_racine, 'resultats\/corpus_entrainement' + str(tranche+1) + '.txt') #sert a identifier le feature tag#
    
    sents_corrects = corpus_test_tuple.tagged_sents()
    
    sents_tagges = tagger.tag_sents(corpus_test_tuple.sents())

    for sent_correct, sent_tagge in zip(sents_corrects, sents_tagges):
        
        phrase_combine = [(mot_correct, mot_tagge) for mot_correct, mot_tagge in zip(sent_correct, sent_tagge)]
        #print(phrase_combine)                        

        indice = 0
        
        for couple in phrase_combine:  
            
            #print("waaaa" + str(couple))
            
            for MI in scores_SVM:
                
                #print(MI)
                
                if couple[0][0] == MI['signifiant']:
    
                    liste_dictionnaires.append(create_dict(phrase_combine, indice))
                    #print(couple[0][1])
                    if couple[0][1] == 'M':
                        liste_y.append(1)
                    else:
                        liste_y.append(0)
                    
                    #print("Mot entr")
                    #print(dict_mot)
                    #print('\n')
    
            indice += 1


    
     ###CONSTRUCTION DU DICTIONNAIRE TEST####
    
    #corpus_test_tuple = TaggedCorpusReader(dossier_racine, nom_tes)
    corpus_test_tuple = TaggedCorpusReader(dossier_racine, 'resultats\/corpus_test' + str(tranche+1) + '.txt') #sert a identifier le feature tag#
    sents_corrects = corpus_test_tuple.tagged_sents()
    sents_tagges = tagger.tag_sents(corpus_test_tuple.sents())
    
    liste_dictionnaires_test = []
    liste_y_test = []

    for sent_correct, sent_tagge in zip(sents_corrects, sents_tagges):
        
        phrase_combine = [(mot_correct, mot_tagge) for mot_correct, mot_tagge in zip(sent_correct, sent_tagge)]
        #print(phrase_combine)                        

        indice = 0
            
        for couple in phrase_combine:
            
            for MI in scores_SVM:
                

                if couple[0][0] == MI['signifiant']:
    
                    liste_dictionnaires_test.append(create_dict(phrase_combine, indice))
                    if couple[0][1] == 'M':
                        liste_y_test.append(1)
                    else:
                        liste_y_test.append(0)
                    
                    #print("Mot test")
                    #print(dict_mot)
                    #print('\n')
    
            indice += 1

    #vectoriation des dictionnaires###
    vec = DictVectorizer()
    listes_colles = liste_dictionnaires + liste_dictionnaires_test

    vecteur_x_ent_plus_test = vec.fit_transform(listes_colles).toarray()
    
    #print(vec.get_feature_names())
    #print(vecteur_x_ent_plus_test)
    
    vecteur_x_entrainement = vecteur_x_ent_plus_test[:len(liste_dictionnaires)]
    vecteur_x_test = vecteur_x_ent_plus_test[len(liste_dictionnaires):]
    
    #entrainement
    #clf = svm.SVC(kernel='linear', C=7, class_weight={1: 7}) #OG
    #SMALL TEST SMALL TRAIN Total signifiants 5203, Fmesure obtenue 0.9341
    #SMALL TEST BIG TRAIN Total signifiants 5203, Fmesure obtenue 0.9355
    #BIG TEST BIG TRAIN   Total signifiants 9740, Fmesure obtenue 0.9769

                  
    #clf = svm.SVC(kernel='linear', C=5, class_weight={1: 5}) # best
    #SMALL TEST SMALL TRAIN Total signifiants 5203, Fmesure obtenue 0.9385
    #SMALL TEST BIG TRAIN Total signifiants 5203, Fmesure obtenue 0.9381
    #BIG TEST BIG TRAIN Total signifiants 9740, Fmesure obtenue 0.9776

    #clf = svm.SVC(kernel='linear', C=2, class_weight={1: 8})
    #SMALL TEST SMALL TRAIN 
    #SMALL TEST BIG TRAIN 
    #BIG TEST BIG TRAIN Total signifiants 9740, Fmesure obtenue 0.9767

    #clf = svm.SVC(kernel='linear', C=6, class_weight={1: 5})
    #SMALL TEST SMALL TRAIN Total signifiants 5203, Fmesure obtenue 0.9383
    #SMALL TEST BIG TRAIN Total signifiants 5203, Fmesure obtenue 0.9377
    #BIG TEST BIG TRAIN Total Total signifiants 9740, Fmesure obtenue 0.9777
    
    clf = svm.SVC(kernel='linear', C=6, class_weight={1: 5})
    
        
    print(clf.get_params())
    clf.fit(vecteur_x_entrainement, liste_y)
    
    
    #        liste_prediction = []
    
    #print(vecteur_x_test)

    prediction = clf.predict(vecteur_x_test)
    
    #print(liste_y_test)
    #print(prediction)
    
    double_y = zip(liste_y_test, prediction)
    
    """#pour utiliser sans signifiant dans dict
    scores_total = {'signifiant': "toute",
          'total_signifiant':0,
          'total_MI':0,
          'MI_reperes':0,
          'MI_corrects':0
         }
    """
    
    for unite, couple_reponse in zip(liste_dictionnaires_test, double_y):
        #print(unite)
        #print(couple_reponse)
        
               
        for M in scores_SVM:
            #print(MI)
            if M['signifiant'] == unite['signifiant']:
                M['total_signifiant'] += 1
                
                if couple_reponse[0] == 1:
                    M['total_MI'] += 1
                        
                if couple_reponse[1] == 1:
                    M['MI_reperes'] += 1

                    if couple_reponse[0] == couple_reponse[1]:
                        M['MI_corrects'] += 1
                
        
        
        """#pour utiliser sans signifiant dans dict      
        scores_total['total_signifiant'] += 1
        
        if couple_reponse[0] == 1:
            scores_total['total_MI'] += 1
                
        if couple_reponse[1] == 1:
            scores_total['MI_reperes'] += 1

            if couple_reponse[0] == couple_reponse[1]:
                scores_total['MI_corrects'] += 1
        """
    
    #scores_SVM.append(scores_total)
    message("Fin de l'analyse SVM.*****")

    
def create_dict(phrase_combine, indice):

    dict_mot = {}


    if signifiant:
        dict_mot['signifiant'] = phrase_combine[indice][0][0]
    """
    if ngram_tag:
        dict_mot['tag'] = phrase_combine[indice][1][1]
    """
    if ngram_tag:
        if phrase_combine[indice][1][1] == 'M':
            dict_mot['tag'] = 0
        else:
            dict_mot['tag'] = 1
            

    if la_apres:
        try:
            if re.match('là$', phrase_combine[indice +1][0][0]):
                dict_mot['la_apres'] = 1
            else:
                dict_mot['la_apres'] = 0
        except IndexError:
            dict_mot['la_apres'] = 'fin_enonce'

    if de_apres:
        try:
            if re.match('de', phrase_combine[indice +1][0][0]):
                dict_mot['de_apres'] = 1
            else:
                dict_mot['de_apres'] = 0
        except IndexError:
            dict_mot['de_apres'] = 'fin_enonce'
            
    if intonation_apres:
        try:
            if re.match('_i', phrase_combine[indice +1][0][0]):
                dict_mot['intonation_apres'] = 1
            else:
                dict_mot['intonation_apres'] = 0
        except IndexError:
            dict_mot['intonation_apres'] = 'fin_enonce'

    if mot_suivant:
        try:
            dict_mot['mot_suivant'] = phrase_combine[indice + 1][0][0]
                
        except IndexError:
            dict_mot['mot_suivant'] = 'fin_enonce'

    if mot_precedent:
        try:
            dict_mot['mot_precedent'] = phrase_combine[indice - 1][0][0]
        except IndexError:
            dict_mot['mot_precedent'] = 'deb_enonce'

    if mot_precedent2:
        try:
            dict_mot['mot_precedent2'] = phrase_combine[indice - 2][0][0]
        except IndexError:
            dict_mot['mot_precedent2'] = 'deb_enonce'
            
    if mot_precedent3:
        try:
            dict_mot['mot_precedent3'] = phrase_combine[indice - 3][0][0]
        except IndexError:
            dict_mot['mot_precedent3'] = 'deb_enonce'


    ###TAG###
    if tag_suivant:
        try:
            dict_mot['tag_suivant'] = phrase_combine[indice + 1][0][1]
                
        except IndexError:
            dict_mot['tag_suivant'] = 'fin_enonce'

    if tag_precedent:
        try:
            dict_mot['tag_precedent'] = phrase_combine[indice - 1][0][1]
        except IndexError:
            dict_mot['tag_precedent'] = 'deb_enonce'

    if tag_2_avant:    
        try:
            dict_mot['tag_2_avant'] = phrase_combine[indice - 2][0][1]
                
        except IndexError:
            dict_mot['tag_2_avant'] = 'deb_enonce'

    if tag_3_avant:
        try:
            dict_mot['tag_3_avant'] = phrase_combine[indice - 3][0][1]
        except IndexError:
            dict_mot['tag_3_avant'] = 'deb_enonce'

    #print(str(dict_mot))


    return dict_mot
    
    
def create_tagger(train_sents):
    
    t0 = nltk.DefaultTagger('S')
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    t3 = nltk.TrigramTagger(train_sents, backoff=t2)
    return t3

    
def calcul_SVM():
    
    global scores_SVM
    
    sortie = ''
    #sortie += '\nUnité\tfmesureMIN\tPrécision\tRappel\tFMesure'
    
    TMI_corrects = 0
    TMI_reperes = 0
    Ttot_S = 0
    Ttotal_MI = 0
    
    for MI in scores_SVM:
        
        precision = 0
        rappel = 0
        fmesure = 0
        fmesure_minimum = 1

        tot_S = 0
        total_MI = 0
        MI_reperes = 0
        MI_corrects = 0

        tot_S = MI['total_signifiant']
        total_MI = MI['total_MI'] 

        try:
            proportionMI = total_MI / tot_S
            fmesure_minimum = 2*(proportionMI*1)/(proportionMI+1)
            
        except ZeroDivisionError:
            proportionMI = 0
            fmesure_minimum = 0
            

        MI_reperes = MI['MI_reperes']
        MI_corrects = MI['MI_corrects']
        
        try:
            precision = MI_corrects / MI_reperes
        except ZeroDivisionError:
            precision = 0

        try:
            rappel = MI_corrects / total_MI
            fmesure = 2*(precision*rappel)/(precision+rappel)
        except ZeroDivisionError:
            rappel = 0
            fmesure = 0

        Ttot_S += MI['total_signifiant'] 
        Ttotal_MI += MI['total_MI']
        TMI_reperes += MI['MI_reperes']
        TMI_corrects += MI['MI_corrects']

        #sortie += "\n{:18} a un minimum de {:5.4f} et une fmesure de {:5.4f}".format(MI['signifiant'],fmesure_minimum, fmesure)            
        #sortie += "\nTotal signifiants {:5}, Total_MI {:5}, MI_reperes {:5}, MI_corrects {:5}".format(tot_S, total_MI, MI_reperes, MI_corrects)
        #sortie += "\n{} {} {} {:.2f} {:.2f} {} {} {:.2f}".format(MI['signifiant'], tot_S, total_MI, proportionMI*100, fmesure_minimum*100, MI_reperes, MI_corrects, fmesure*100)
        sortie += "\n{:.2f}".format(fmesure*100)
    try:
        Tprecision = TMI_corrects / TMI_reperes
        Trappel = TMI_corrects / Ttotal_MI
        Tfmesure = 2*(Tprecision*Trappel)/(Tprecision+Trappel)
    except ZeroDivisionError:
        Tfmesure = 0
    
    sortie += "\nTotal signifiants {}, Fmesure obtenue {:5.4f}".format(Ttot_S, Tfmesure)
    
    return sortie

def calcul_ngram():
    
    global scores_ngram
    
    sortie = ''
    sortie += '\nUnité\tfmesureMIN\tPrécision\tRappel\tFMesure'
    
    TMI_corrects = 0
    TMI_reperes = 0
    Ttot_S = 0
    Ttotal_MI = 0
    
    for MI in scores_ngram:
        
        precision = 0
        rappel = 0
        fmesure = 0
        fmesure_minimum = 1

        tot_S = 0
        total_MI = 0
        MI_reperes = 0
        MI_corrects = 0

        tot_S = MI['total_signifiant']
        total_MI = MI['total_MI'] 

        try:
            proportionMI = total_MI / tot_S
            fmesure_minimum = 2*(proportionMI*1)/(proportionMI+1)
            
        except ZeroDivisionError:
            proportionMI = 0
            fmesure_minimum = 0
            

        MI_reperes = MI['MI_reperes']
        MI_corrects = MI['MI_corrects']
        
        try:
            precision = MI_corrects / MI_reperes
        except ZeroDivisionError:
            precision = 0

        try:
            rappel = MI_corrects / total_MI
            fmesure = 2*(precision*rappel)/(precision+rappel)
        except ZeroDivisionError:
            rappel = 0
            fmesure = 0

        Ttot_S += MI['total_signifiant'] 
        Ttotal_MI += MI['total_MI']
        TMI_reperes += MI['MI_reperes']
        TMI_corrects += MI['MI_corrects']

        #sortie += "\n{:18} a un minimum de {:5.4f} et une fmesure de {:5.4f}".format(MI['signifiant'],fmesure_minimum, fmesure)            
        #sortie += "\nTotal signifiants {:5}, Total_MI {:5}, MI_reperes {:5}, MI_corrects {:5}".format(tot_S, total_MI, MI_reperes, MI_corrects)
        sortie += "\n{} {} {} {:.2f} {:.2f} {} {} {:.2f}".format(MI['signifiant'], tot_S, total_MI, proportionMI*100, fmesure_minimum*100, MI_reperes, MI_corrects, fmesure*100)
        #sortie += "\n{:.2f}".format(fmesure*100)        

    try:
        Tprecision = TMI_corrects / TMI_reperes
        Trappel = TMI_corrects / Ttotal_MI
        Tfmesure = 2*(Tprecision*Trappel)/(Tprecision+Trappel)
    except ZeroDivisionError:
        Tfmesure = 0
    
    
    sortie += "\nTotal signifiants {}, Fmesure obtenue {:5.4f}".format(Ttot_S, Tfmesure)
    
    return sortie
    
main()
    
