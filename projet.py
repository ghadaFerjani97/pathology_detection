import pandas as pd
import numpy as np
import utils 
from math import sqrt, log2
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

def getPrior(DB):
    dic = {}
    m =  DB['target'].mean() # la moyenne des targets
    dic['estimation'] = m
    var = m*(1-m) # la variance d'une loi binomiale de paramètre m
    shape = DB.shape
    nbElt = shape[0]
    ecart_type = sqrt(var)/sqrt(nbElt)
    dic['min5pourcent'] = m-1.96*ecart_type
    dic['max5pourcent'] = m+1.96*ecart_type
    return dic


"""
Les probablilités des differents valeurs de target calculé par la fonction getPrior
"""
p_target_1 = 0.7453874538745388
p_target_0=1-p_target_1


class APrioriClassifier(utils.AbstractClassifier):
    """
    Classifieur a priori: classifie tout individu comme etant malade
    """
    def estimClass(self, attrs): # etant donnee les probas a priori,
                                # il y a plus de 1  que de 0 donc on classifie tout à 1
        return 1
    
    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.
    
        :param df:  le dataframe à tester
        :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        """
        dic = {}
        dic['VP']=0
        dic['VN'] = 0
        dic['FP'] = 0
        dic['FN'] = 0
            
        n=df.shape[0]
        for i in range(n):
            line=utils.getNthDict(df,i)
            t=self.estimClass(line)
            if line['target']==t:
                if t==1:
                    dic['VP']+=1
                else:
                    dic['VN']+=1
            else:
                if t==1:
                    dic['FP']+=1
                else:
                    dic['FN']+=1
        dic['rappel']=dic['VP']/(dic['VP']+dic['FN'])
        dic['précision']=dic['VP']/(dic['VP']+dic['FP'])
        return dic
    

def P2D_l(df,attr):
    """
    retourne une dictionnaire à 2 dimensions des vraissamblance P(attr,target)
    """
    nb = df.groupby(['target']).size() #nb[i] est le nombre de fois ou target ==i
    dic = {}
    dic[0] = {}
    dic[1] = {}
    lst = df.groupby([attr,'target']).size()
    keys = list(df[attr].unique())
    for e in lst.iteritems():
        a,t = e[0]
        dic[t][a] = e[1]/nb[t]# dic[t][a] = nb de fois ou target=t et attr = a divise par le nb total de fois ou target = t
    for key in keys:
        if key not in dic[0]:
            dic[0][key] = 0.0
        if key not in dic[1]:
            dic[1][key] = 0.0
    return dic


def P2D_p(df,attr):
    """
    retourne une dictionnaire à deux dimensions des probabilités à posteriori P(target,attr)
    """
    nb = df.groupby([attr]).size() #nb[i] est le nombre de fois ou attr ==i
    dic={}
    lst=df.groupby([attr,'target']).size()
    A = df[attr].unique()
    for i in A:
        dic[i]={}
    for e in lst.iteritems():
        a,t = e[0]
        dic[a][t]=e[1]/nb[a] # dic[a][t] := nb de fois ou target = t et attr = a divise par le 
                                # nb total de fois ou attr = a
    for i in A:
        if 1 not in dic[i]:
            dic[i][1]=0.0
        if 0 not in dic[i]:
            dic[i][0] = 0.0
    return dic

class ML2DClassifier(APrioriClassifier):
    """
    Classifieur de maximum de vraissamblance: 
    classifie un individu en utilisant le maximum de vraissamblance de target avec un attribut attr donné en
    paramètre de constructeur
    """
    def __init__(self,df,attr):#création d'un individu à partir d'une base de données,d'un attribu donnée pour le classifier 
        self.attr = attr
        self.P2Dl=P2D_l(df,attr)
        
        
    def estimClass(self,attrs): #attrs dictionnaire 
         attr=self.attr #attribut sur lequel on effectuera la vraissemblance 
         a=attrs[attr] #attrs = une ligne pour un individu donnée , a= la valeur de attr dans la ligne attrs
         if self.P2Dl[0][a]>=self.P2Dl[1][a]:
            return 0
         else: 
            return 1
        
   
    
class MAP2DClassifier(APrioriClassifier):
    """
    Classidieur de maximum de probabilité à posteriori:
    classifie un individu en utilisant le probabilité à posteriori maximal de target avec un attribut attr
    """
    def __init__(self,df,attr):
        self.attr=attr
        self.P2D_p = P2D_p(df,attr)
       
    def estimClass(self,attrs):
        attr=self.attr
        a=attrs[attr]
        if self.P2D_p[a][0]>self.P2D_p[a][1]:
            return 0
        else:
            return 1
        
    
    
    
def nbParams(df,liste=None):
   if liste==None:
       liste = list(df)
   k=1
   l=len(liste)
   for att in liste:
       n=len(df[att].unique())
       k=k*n
   print(str(l)+' variable(s) = '+str(8*k)+' Octets')
   
def nbParamsIndep(df,liste=None):
   if liste==None:
       liste = list(df)
   k=0
   l=len(liste)
   for att in liste:
       n=len(df[att].unique())
       k=k+n
   print(str(l)+' variable(s) = '+str(8*k)+' Octets')
        
def drawNaiveBayes(df,parent):
    liste_attr = list(df)
    grph = ""
    for attr in liste_attr:
        if attr != parent:
            grph+=parent+"->"+attr+";"
    return utils.drawGraph(grph)

def nbParamsNaiveBayes(df,attr,liste=None):
    m=len(df[attr].unique())
    if liste==None:
        liste=list(df)
    if len(liste)==0:
        k=0
    else:
        k=0
    for att in liste:
        if att==attr:
            continue
        n=len(df[att].unique())
        k=k+n
    k=k*m
    k+=m
    
    print(str(len(liste))+' variable(s) = '+str(8*k)+' Octets')


def PnD_l(df,attrs=None):
    """
    Entree: un data frame, une liste d'attributs de dataframe, la liste par défaut est la liste de tous les attributs
    Sortie: une dictionnaire tel que pour tout attribut att dans la liste attrs, dic[att] = P2D_l(df,att)
    """ 
    if attrs == None:
        attrs = list(df)
    dic = {}
    for att in attrs:
        if att!='target':
            dic[att] = P2D_l(df,att)
    return dic

class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur de maximum de vraissamblance:
    On estime pour chaque individu la vraissamblance de chaque valeur possible de target avec la méthode de bayes naive
    grace aux informations donnees sur l'individu ainsi que le dictionnaire PnD_l
    """
    def __init__(self,df):
         self.PnD_l = PnD_l(df)#dictionnaires a 3 dimensions des vraissamblance calcule par PnD_l
         self.attrs = list(df)#liste des attributs sur lesquels on effectue la formule de bayes

    def estimProbas(self,attrs):
        dic={}
        dic[0]=1
        dic[1]=1
        for att in self.attrs:
            if att!='target':
                if attrs[att] not in self.PnD_l[att][0]: #pour des valeurs des attributs non existant dans 
                    dic[0] = 0.0                        # nos informations de la table PnD_l, on retourne 0 par défaut
                    dic[1] = 0.0
                    return dic
                else:
                    dic[0]=dic[0]*self.PnD_l[att][0][attrs[att]]
                    dic[1]=dic[1]*self.PnD_l[att][1][attrs[att]]
        return dic
    
    def estimClass(self,attrs):
        
        dic=self.estimProbas(attrs)
        if dic[0]>=dic[1]:
            return 0
        else:
            return 1
        

class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur de maximum de probabilité a posteriori:
    On estime la probabilité de chaque valeur possible de target avec la méthode de bayes naive
    grace aux informations donnees sur l'individu ainsi que le dictionnaire PnD_l
    """
    def __init__(self,df):
        self.PnD_l=PnD_l(df) #dictionnaires a 3 dimensions des vraissamblance calcule par PnD_l
        self.attrs = list(df) #liste des attributs sur lesquels on effectue la formule de bayes
    
    def estimProbas(self,attrs):
        dic={}
        dic[0]=p_target_0
        dic[1]=p_target_1
        for att in self.attrs:
            if att!='target':
                if attrs[att] not in self.PnD_l[att][0]:#pour des valeurs des attributs non existant dans 
                    dic[0] = 0.0                        # nos informations de la table PnD_l, on retourne 0 par défaut
                    dic[1] = 0.0
                    return dic
                else:
                    dic[0]=dic[0]*self.PnD_l[att][0][attrs[att]]
                    dic[1]=dic[1]*self.PnD_l[att][1][attrs[att]]
        if dic[1]!=0.0 or dic[0]!=0.0:
            PJointe = dic[0]+dic[1]
            dic[0]=dic[0]/PJointe
            dic[1]=dic[1]/PJointe
        return dic
    
    def estimClass(self,attrs):
        dic=self.estimProbas(attrs)
        if dic[0]>=dic[1]:
            return 0
        else:
            return 1
 
def isIndepFromTarget(df,attr,x):
    """
    Verifie l'independance de target et attr avec une tolerance de x avec les informations de df
    """
    lst = df.groupby([attr,'target']).size()
    A = df[attr].unique()
    observed = np.zeros((len(A),2))
    dic = {} # dictionnaire permettan de numeroter les differentes valeus de l'attribut attr
    for i in range(len(A)):
        dic[A[i]] = i
    for e in lst.iteritems():
        a,t = e[0]
        observed[dic[a]][t] = e[1]
    g, p, dof, expctd = chi2_contingency(observed)
    return p>x


def lstNotIndepAttrs(df,x):
    """
    Entree: un dataframe df, une vleur de tolerance
    Sortie: une liste d'attributs independants avec target dans df avec une tolerance egale à x
    """
    attrs = list(df)
    lst = []
    for attr in attrs:
        if isIndepFromTarget(df,attr,x)==False:
            lst.append(attr)
    return lst

class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    class hérité de MLNaiveBayesClassifier où seuls les attributs non independants de target avec une tolérance de x 
    sont pris en compte
    """
    def __init__(self,df,x):
        self.attrs = lstNotIndepAttrs(df,x)
        self.PnD_l = PnD_l(df,self.attrs)
        
    def draw(self):
        graph = ""
        for att in self.attrs:
            if att!="target":
                graph+="target->"+att+";"
        return utils.drawGraph(graph)
    
    
class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """
    class hérité de MAPNaiveBayesClassifier où seuls les attributs non independants de target avec une tolérance de x 
    sont pris en compte
    """
    def __init__(self,df,x):
        self.attrs = lstNotIndepAttrs(df,x)
        self.PnD_l = PnD_l(df,self.attrs)
        
    def draw(self):
        graph = ""
        for att in self.attrs:
            if att!="target":
                graph+="target->"+att+";"
        return utils.drawGraph(graph)
    
    
def mapClassifiers(dic,df):
    """
    Trace le graphe de comparaisons des différents classifieurs.
    """
    for tag in dic:
        results = dic[tag].statsOnDF(df)
        plt.scatter(results["précision"],results["rappel"],marker = 'x', color = 'red')
        plt.text(results["précision"],results["rappel"],tag)
    plt.show()
    
def MutualInformation(df,x,y):
    """
    Calcul l'information mutuelle des deux attributs x et y avec les informations du dataframe df
    I(x,x)
    """

    occ_xy = df.groupby([x,y]).size() # le nombre d'occurence des differents couples (X,Y) avec X (resp. Y) fixe ayant une valeur possible de x(resp. Y)
    occ_x = df.groupby([x]).size() # le nombre d'occurence des differents valeurs de x
    occ_y = df.groupby([y]).size() # le nombre d'occurence des differents valeurs de y
    N = occ_xy.sum() # le nombre total d'occurences possilbes pour (X,Y)
    I = 0
 
    for e in occ_xy.iteritems():
        X,Y = e[0]
        I+=e[1]*(log2((N*e[1])/(occ_x[X]*occ_y[Y])))
  
    return I/N


def ConditionalMutualInformation(df,x,y,z):
    """
    Calcul l'information mutuelle conditionnelle  avec les informations du dataframe df
    I(x,y|z)
    """
    # le nombre d'occurences des differents nuplets des attributs
    occ_xyz = df.groupby([x,y,z]).size()
    occ_xz = df.groupby([x,z]).size()
    occ_yz = df.groupby([y,z]).size()
    occ_z = df.groupby([z]).size()
    # le nombe total de nuplet possible pour chaque combinaison
    N_xyz = occ_xyz.sum()
    N_xz = occ_xz.sum()
    N_yz = occ_yz.sum()
    N_z = occ_z.sum()
    # les probabilites des differents lois jointes
    p_xyz = occ_xyz/N_xyz
    p_xz = occ_xz/N_xz
    p_yz = occ_yz/N_yz
    p_z = occ_z/N_z
    
    I = 0
    for e in p_xyz.iteritems():
        X,Y,Z = e[0]
        I += e[1]*log2((p_z[Z]*e[1])/(p_xz[X][Z]*p_yz[Y][Z]))
    return I
    
def MeanForSymetricWeights(a):
    s = 0
    l,c = a.shape
    for i in range(l):
        for j in range(i,c):
            s+=2*a[i,j]
    return s/(l*c)

def SimplifyConditionalMutualInformationMatrix(a):
    l,c = a.shape
    m = MeanForSymetricWeights(a)
    for i in range(l):
        for j in range(i,c):
            if a[i,j]<m:
                a[i,j]=0
                a[j,i]=0
    return a
    
        