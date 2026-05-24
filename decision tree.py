import numpy as np
from collections import Counter




class Node :
    def __init__(self, Feature = None, threshold = None, data_left = None, data_right = None, information_gain = None, leaf_value = None):
        self.Feature = Feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.information_gain = information_gain
        self.leaf_value = leaf_value



class DecisionTreeClassifier :
    def __init__(self,min_samples_split = 2, max_depth = 5):    #hyperparameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None


    def _entropy(self, s): #  calculates the impurity of an input vector s
        counts = np.unique(s, return_counts=True)[
            1]  # uso l indice 1 in quanto unique restituisce una tupla di array ed io voglio solo il secondo
        probabilities = counts / len(array)

        ent = np.sum(-probabilities * np.log2(probabilities))
        return ent


    def _information_gain(self, parent, sx_node, rx_node): #calculates the information gain value of a split between a parent and two children
        left_value = len(sx_node) / len(parent)
        right_value = len(rx_node) / len(parent)
        sx_node_ent = self._entropy(sx_node)
        rx_node_ent = self._entropy(rx_node)
        parent_ent = self._entropy(parent)
        Gain = parent_ent - ((left_value * sx_node_ent) + (right_value * rx_node_ent))
        return Gain

    def _best_split(X, y): #function calculates the best splitting parameters for input features X and a target variable y. It does so by iterating over every column in X and every threshold value in every column to find the optimal split using information gain
        best_split = {}
        best_info_gain = -1 #lo stiamo solo inizializzando per ora
        n_rows, n_columns = X.shape
        for f_idx in range(n_columns):
            X_current = X[:,f_idx]
            for threshold in np.unique(X_current):
                df = np.concatenate(X,y.reshape(1,-1).T, axis = 1)
                df_left = np.array(row for rows in df if df[f_idx] <= threshold)
                df_right = np.array(row for rows in df if df[f_idx > threshold])


                y = df[:,-1]
                y_left = df_left[:,-1]
                y_right = df_right[:,-1]

                gain = self._information_gain(y,y_left,y_right)
                if gain > best_info_gain:
                    best_split = {
                        'feature_index': f_idx,
                        'threshold': threshold,
                        'df_left': df_left,
                        'df_right': df_right,
                        'gain': gain
                    }
                    best_info_gain = gain
        return best_split


    def _build(self,X, y, depth=0): #function recursively builds a decision tree until stopping criteria is met (hyperparameters in the constructor)
        n_rows,n_columns = X.shape

        #ora voglio controllare se il nodo sia un leaf node utilizzando gli stopping criteria
        if n_rows >= self.min_samples_split and depth<=self.max_depth
            best = self._best_split(X,y)
            if best['gain'] > 0:

                #costruisco un albero a sinistra
                left = self._build(X=best['df_left'][:,:-1],y=best['df_left'][:,-1], depth= depth +1) #usando :,:-1 indico che prendo tutte le colonne tranne l ultima, usando :,-1 indico che prendo solo l ultima
                #costruisco un albero a destra
                right = self._build(X=best['df_right'][:,:-1],y=best['df_right'][:,-1], depth= depth +1)

        #gestiamo ora il nodo leaf
        return Node(
            leaf_value=Counter(y).most_common(1)[0][0] #counter(y) conta gli elementi e restituisce un dizionario con l'elemento e le sue occorrenze, (1) prende il solo vincitore quindi quello con più occorrenze, [0] estra la tupla dalla lista e infine l ultimo [0] estrae il vincitore
        )
    def fit(self,X, y): #function calls the _build() function and stores the built tree to the constructor
        self.root = self._build(X,y) #chiama la funzione ricorsiva per costruire l'albero

    def _predict(x): #function traverses the tree to classify a single instance

        if tree.leaf_value != None:
            return tree.leaf_value
        feature_value = x[tree.feature]

        if feature_value < tree.thresold:
            return self._predict(x=x,tree.data_left)
        if feature_value > tree.thresold:
            return self._predict(x=x,tree.data_right)


    def predict(self,X): #function applies the _predict() function to every instance in matrix X.

        return[self._predict(x,self.root) for x in X]






