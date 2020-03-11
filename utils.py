import numpy as np

from IPython.display import Image
import pydotplus

import matplotlib
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making plots with seaborn

import math  # for math


def getNthDict(df, n):
  return df[n:n + 1].to_dict(orient='records')[0]


def viewData(data, kde=True):
  """
  visualisation d'un pandas.dataframe sous la forme d'histogramme (avec uu gaussian kernel densiy estimate si demandé
  et intéressant)

  Arguments:
    data {pandas.dataFrame} -- le pandas.dataframe à visualiser

  Keyword Arguments:
    kde {bool} -- demande de l'affichage du gaussian kdf (default: {True})
  """
  x = 4
  y = math.ceil(len(data.keys()) / x)
  plt.figure(figsize=(x * 4, y * 2))
  for i, k in enumerate(data.keys()):
    ax = plt.subplot(x, y, i + 1, xticklabels=[])
    ax.set_title("Distribution of '{0}': {1} in [{2},{3}]".format(k, len(data[k].unique()), data[k].min(), data[k].max()))
    ax = sns.distplot(data[k], kde=kde and len(data[k].unique()) > 5)
    ax.set_xlabel("")


def discretizeData(data):
  """Discrétisation automatique utilisant numpy

  Arguments:
    data {pandas.dataframe} -- le dataframe dont certaines colonnes seront à discrétisées

  Returns:
    pandas.dataframe -- le nouveau dataframe discréité
  """
  newData = data.copy()
  for k in newData.keys():
    if len(newData[k].unique()) > 5:
      newData[k] = data.apply(lambda row: np.digitize(row[k], np.histogram_bin_edges(newData[k], bins="fd")),
                              axis=1)
  return newData


class AbstractClassifier:
  """
  Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs. Il propose aussi comme service
  de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
  """

  def ___init__(self):
    pass

  def estimClass(self, attrs):
    """
    à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

    :param attrs: le  dictionnaire nom-valeur des attributs
    :return: la classe 0 ou 1 estimée
    """
    raise NotImplementedError

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
        line=getNthDict(df,i)
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

__GRAPHPREAMBULE='digraph{margin="0,0";node [style=filled, color = black, fillcolor=lightgrey,fontsize=10,shape=box,margin=0.05,width=0,height=0];'
def drawGraphHorizontal(arcs):
  """
  Dessine un graph (horizontalement) à partir d'une chaîne décrivant ses arcs (et noeuds)  (par exemple 'A->B;C->A')"
  :param arcs: la chaîne contenant les arcs
  :return: l'image représentant le graphe
  """
  graph = pydotplus.graph_from_dot_data(__GRAPHPREAMBULE+'rankdir=LR;' + arcs + '}')
  return Image(graph.create_png())

def drawGraph(arcs):
  """
  Dessine un graph à partir d'une chaîne décrivant ses arcs (et noeuds)  (par exemple 'A->B;C->A')"
  :param arcs: la chaîne contenant les arcs
  :return: l'image représentant le graphe
  """
  graph = pydotplus.graph_from_dot_data(__GRAPHPREAMBULE + arcs + '}')
  return Image(graph.create_png())
