import numpy as np
from collections import Counter




class Node :
    def __init__(self, Feature = None, threshold = None, data_left = None, data_right = None, information_gain = None, leaf_value = None, probs = None):
        self.Feature = Feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.information_gain = information_gain
        self.leaf_value = leaf_value
        self.probs = probs



class DecisionTreeClassifier :
    def __init__(self, min_samples_split=2, max_depth=5, max_features="sqrt", min_information_gain=0.0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_information_gain = min_information_gain
        self.root = None
        self.classes_ = None


    def _entropy(self, s): #  calculates the impurity of an input vector s
        counts = np.unique(s, return_counts=True)[
            1]  # uso l indice 1 in quanto unique restituisce una tupla di array ed io voglio solo il secondo
        probabilities = counts / len(s)

        ent = np.sum(-probabilities * np.log2(probabilities))
        return ent


    def _information_gain(self, parent_ent, parent, sx_node, rx_node): #calculates the information gain value of a split between a parent and two children
        left_value = len(sx_node) / len(parent)
        right_value = len(rx_node) / len(parent)
        sx_node_ent = self._entropy(sx_node)
        rx_node_ent = self._entropy(rx_node)
        Gain = parent_ent - ((left_value * sx_node_ent) + (right_value * rx_node_ent))
        return Gain

    def _label_probs(self, y): #calcola il vettore di probabilità delle classi allineato a self.classes_
        counter = Counter(y)
        total = len(y)
        probs = np.zeros(len(self.classes_), dtype=float)
        for cls, count in counter.items():
            cls_idx = np.where(self.classes_ == cls)[0][0]
            probs[cls_idx] = count / total
        return probs

    def _best_split(self, X,y):  # function calculates the best splitting parameters for input features X and a target variable y. It does so by iterating over every column in X and every threshold value in every column to find the optimal split using information gain
        best_split = {}
        best_info_gain = -1  # lo stiamo solo inizializzando per ora
        n_rows, n_columns = X.shape

        # --- SELEZIONE CASUALE DELLE FEATURE PER RANDOM FOREST ---
        if self.max_features == "sqrt":
            n_features_to_split = int(np.sqrt(n_columns))
        elif self.max_features == "log":
            n_features_to_split = int(np.log2(n_columns))
        elif isinstance(self.max_features, int):
            n_features_to_split = self.max_features
        else:
            n_features_to_split = n_columns

        n_features_to_split = max(1, min(n_columns, n_features_to_split))

        # Selezioniamo gli indici delle colonne a caso e senza ripetizioni
        sub_features_idx = np.random.choice(n_columns, size=n_features_to_split, replace=False)
        # --- FINE SELEZIONE CASUALE ---

        df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
        parent_ent = self._entropy(y) #calcolato una sola volta fuori dai due for

        # Sostituito range(n_columns) con la nostra lista di feature estratte a caso
        for f_idx in sub_features_idx:
            X_current = X[:, f_idx]
            # uso i percentili invece di np.unique per ridurre drasticamente i threshold candidati
            thresholds = np.percentile(X_current, q=np.arange(25, 100, 25))
            for threshold in thresholds:
                df_left = df[df[:, f_idx] <= threshold]
                df_right = df[df[:, f_idx] > threshold]

                # Controllo di sicurezza: se uno dei due rami è vuoto, lo split non è valido
                if len(df_left) == 0 or len(df_right) == 0:
                    continue

                y_left = df_left[:, -1]
                y_right = df_right[:, -1]

                gain = self._information_gain(parent_ent, y, y_left, y_right)
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

        # calcolo le probs del nodo corrente sempre, così anche i nodi interni le hanno
        node_probs = self._label_probs(y)
        node_leaf_value = Counter(y).most_common(1)[0][0] #counter(y) conta gli elementi e restituisce un dizionario con l'elemento e le sue occorrenze, (1) prende il solo vincitore quindi quello con più occorrenze, [0] estra la tupla dalla lista e infine l ultimo [0] estrae il vincitore

        #ora voglio controllare se il nodo sia un leaf node utilizzando gli stopping criteria
        if n_rows >= self.min_samples_split and depth<=self.max_depth:
            best = self._best_split(X,y)
            if best and best['gain'] > self.min_information_gain:

                #costruisco un albero a sinistra
                left = self._build(X=best['df_left'][:,:-1],y=best['df_left'][:,-1], depth= depth +1) #usando :,:-1 indico che prendo tutte le colonne tranne l ultima, usando :,-1 indico che prendo solo l ultima
                #costruisco un albero a destra
                right = self._build(X=best['df_right'][:,:-1],y=best['df_right'][:,-1], depth= depth +1)

                return Node(
                    Feature=best["feature_index"],
                    threshold=best["threshold"],
                    data_left=left,
                    data_right=right,
                    information_gain=best["gain"],
                    leaf_value=node_leaf_value,
                    probs=node_probs
                )

        #gestiamo ora il nodo leaf
        return Node(
            leaf_value=node_leaf_value,
            probs=node_probs
        )
    def fit(self,X, y): #function calls the _build() function and stores the built tree to the constructor
        self.classes_ = np.unique(y)
        self.root = self._build(X,y) #chiama la funzione ricorsiva per costruire l'albero

    def _predict(self,x,tree): #function traverses the tree to classify a single instance

        if tree.data_left is None and tree.data_right is None:
            return tree.leaf_value
        feature_value = x[tree.Feature]

        if feature_value <= tree.threshold:
            return self._predict(x,tree.data_left)
        else:
            return self._predict(x,tree.data_right)

    def _predict_proba_single(self, x, tree):

        if tree.data_left is None and tree.data_right is None:
            return tree.probs
        feature_value = x[tree.Feature]

        if feature_value <= tree.threshold:
            return self._predict_proba_single(x, tree.data_left)
        else:
            return self._predict_proba_single(x, tree.data_right)


    def predict(self,X): #function applies the _predict() function to every instance in matrix X.

        return[self._predict(x,self.root) for x in X]

    def predict_proba(self, X):

        return np.array([self._predict_proba_single(x, self.root) for x in X])



    #roberto riccio
