import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,10))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]]]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='12', ha='left',ma='left',va='top',linespacing=400,rotation=label_rotation,weight=600,color="sienna", alpha=1)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='None', edgecolor='grey')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='-')
            plt.plot([0, 0], [-1, 1], color='grey', ls='-')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None,lims=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(10,10))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])
        
            # palette de couleur
            cmap = plt.cm.get_cmap("tab20").copy()
            cmap.set_under("black")

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    x=X_projected[selected, d1]
                    y=X_projected[selected, d2]
                    plt.scatter(x=x, y=y, alpha=alpha, label=value,marker='.',cmap=cmap)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            
            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()

def FillRate(data):
    filled_data = data.count().sum()
    row,col=data.shape
    fill_rate=filled_data/(row*col)*100   
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title("Taux de remplissage", fontsize=25) 
    ax.axis("equal") 
    ax.pie([fill_rate, 100 - fill_rate], labels=["Taux de remplissage", "Taux de valeurs manquantes"],autopct='%1.2f%%',explode=(0,0.1),radius=1)
    plt.legend(["Taux de remplissage", "Taux de valeurs manquantes"])

def DataStructure(data):
    type_of_variable = data.dtypes.value_counts()

    Nombre_variables_numeriques =type_of_variable[type_of_variable.index=='float64'][0]

    dict_structure = {'Nombre de lignes':data.shape[0], 'Nombre de colonnes':int(data.shape[1]), 
                      'Nombre de variables catégorielles':int(type_of_variable[type_of_variable.index=='object'][0]),
                      'Nombre de variables numériques ':int(Nombre_variables_numeriques),
                      'Pourcentage de données manquantes':int((data.isnull().sum()/len(data)*100).mean()),
                      'Nombre de doublons':int(len(data[data.duplicated()]))}

    structure_data=pd.DataFrame(list(dict_structure.items()),columns=['Caractéristiques','Valeurs'])
    return structure_data

def TraitementData(data):
    for col in data.columns:
        data[col] = data[col].astype(str).str.lower()
        data[col] = data[col].astype(str).str.replace(r'[-]', ' ', regex=True)
        data[col] = data[col].astype(str).str.replace('en:','')
        data[col] = data[col].astype(str).str.replace('fr:','')
        data[col] = data[col].astype(str).str.replace('de:','')
        data[col] = data[col].astype(str).str.replace('fi:','')
        data[col] = data[col].astype(str).str.replace('it:','')
        data[col] = data[col].astype(str).str.replace('pt:','')
        data[col] = data[col].astype(str).str.replace('ç','c')
        data[col] = data[col].astype(str).str.replace('œ','oe')
        data[col] = data[col].astype(str).str.replace('æ','ae')
        data[col] = data[col].astype(str).str.replace('ï','i')
        data[col] = data[col].astype(str).str.replace('î','i')
        data[col] = data[col].astype(str).str.replace('pomme de terre','patate')
        data[col] = data[col].astype(str).str.replace('pommes de terre','patate')
        data[col] = data[col].astype(str).str.title()

def MajNan(data):
    for col in data.columns:
        data[col] = data[col].replace('nan',np.NaN)
        data[col] = data[col].replace('NaN',np.NaN)
        data[col] = data[col].replace('none',np.NaN)
        data[col] = data[col].replace('None',np.NaN)
        data[col] = data[col].replace('unknown',np.NaN)
        data[col] = data[col].replace('Unknown',np.NaN)
        
def ImputeEnergy(data):
    
    energy_exception =['polyols_100g','fiber_100g', 'alcohol_100g']

    if data[energy_exception].any in list(data.columns):
        data['energy_100g'].fillna(data['fat_100g']*37.76 + data['carbohydrates_100g']*16.744 + data['proteins_100g']*16.744 + data['alcohol_100g']*29.300\
                                 + data['fiber_100g']*8.370+ data['polyols_100g']*1.040, inplace=True)
    else :
        data['energy_100g'].fillna(data['fat_100g']*37.674 + data['carbohydrates_100g']*16.744 + data['proteins_100g']*16.744 , inplace=True)
        
    return data

def CalculNutriScore(row):
    #Energy
    if row["energy_100g"] <= 335:
        a = 0
    elif ((row["energy_100g"] > 335) & (row["energy_100g"] <= 1675)):
        a = 5
    else:
        a = 10 
    #Sugar
    if row["sugars_100g"] <= 4.5:
        b = 0
    elif ((row["sugars_100g"] > 4.5) & (row["sugars_100g"] <= 22.5)):
        b = 5
    else:
        b = 10
    #saturated-fat
    if row["saturated_fat_100g"] <= 1:
        c = 0
    elif ((row["saturated_fat_100g"] > 1) & (row["saturated_fat_100g"] <= 5)):
        c = 5
    else:
        c = 10
    #sodium
    if (row["sodium_100g"]/1000) <= 90:
        d = 0
    elif (((row["sodium_100g"]/1000) > 90) & ((row["sodium_100g"]/1000) <= 450)):
        d = 5
    else:
        d = 10
    #fruits-vegetables-rate
    if row["fruits_vegetables_rate_100g"] <= 40:
        e = 0
    elif ((row["fruits_vegetables_rate_100g"] > 40) & (row["fruits_vegetables_rate_100g"] <= 80)):
        e = -2
    else:
        e = -5
    #fiber
    if row["fiber_100g"] <= 0.7:
        f = 0
    elif ((row["fiber_100g"] > 0.7) & (row["fiber_100g"] <= 3.5)):
        f = -2
    else:
        f = -5
    #proteins
    if row["proteins_100g"] <= 1.6:
        g = 0
    elif ((row["proteins_100g"] > 1.6) & (row["proteins_100g"] <= 8)):
        g = -2
    else:
        g = -5
    
    #Global_score
    global_score = a+b+c+d+e+f+g
    
    return global_score

def CalculNutriGrade(row):
    if row["CalculNutriScore"] < 0 :
        nutriscore = "a"
    elif ((row["CalculNutriScore"] >= 0) & (row["CalculNutriScore"] < 5)) :
        nutriscore = "b"
    elif ((row["CalculNutriScore"] >= 5) & (row["CalculNutriScore"] < 10)) :
        nutriscore = "c"
    elif ((row["CalculNutriScore"] >= 10) & (row["CalculNutriScore"] < 20)) :
        nutriscore = "d"
    else:
        nutriscore = "e"
        
    return nutriscore

def convertion_alpha_score(lettre):
    if lettre =="a":
        score=1
    elif lettre == "b":
        score=2
    elif lettre == "c":
        score=3
    elif lettre == "d":
        score=4
    elif lettre == "e":
        score=5
    else:
        score=6
    return score