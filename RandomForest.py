import numpy as np
from DecisionTree import Node,DecisionTreeClassifier


class Random_forest_classifier:
    '''
    passo al costruttore tutti gli iperparametri dell'algoritmo
    '''

    # Cambia questo nell'__init__ di Random_forest_classifier
    def __init__(self, n_base_learner=10, max_depth=5, min_samples_split=2,
                 max_features="sqrt", min_information_gain=0.0,
                 bootstrap_sample_size=None) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_information_gain = min_information_gain
        self.bootstrap_sample_size = bootstrap_sample_size


    def BootStrapping(self,X,y):

        Bootstrap_samples_x = []
        Bootstrap_sample_y = []

        sample_size = self.bootstrap_sample_size if self.bootstrap_sample_size else X.shape[0] #gli sto dicendo che se non viene impostata una grandezza dell'array, questa sarà uguale alle dimensioni della matrice X

        for i in range(self.n_base_learner):

            samples_index = np.random.choice(X.shape[0],size=sample_size,replace=True) #indico che deve pescare tutte le righe di x, generare un array di indici della grandezza di bootstrap_sample_size e di reinserire numeri già usati
            Bootstrap_samples_x.append(X[samples_index])

            Bootstrap_sample_y.append(y[samples_index])

        return Bootstrap_samples_x,Bootstrap_sample_y


    def Train(self,X_train : np.array, y_train : np.array) -> None:

        Bootstrap_samples_x, Bootstrap_sample_y = self.BootStrapping(X_train,y_train)

        self.base_learner_list = []


        for base_learner_index in range(self.n_base_learner):


            base_learner = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_information_gain=self.min_information_gain
            )

            # Passiamo il set X e y estratto per questo specifico albero tramite .fit() creata nell'altro file

            base_learner.fit(
                Bootstrap_samples_x[base_learner_index],
                Bootstrap_sample_y[base_learner_index]
            )

            # Salviamo l'albero nella nostra lista
            self.base_learner_list.append(base_learner)


    def _predict_probs_with_base_learners(self,X_set:np.array) -> list:
        #creo una lista di predizioni per ognuno degli alberi del tipo [0.1, 0.5]
        pred_prob_list = []
        for base_learners in (self.base_learner_list):
            pred_prob_list.append(base_learners.predict_proba(X_set))
        return pred_prob_list

    def predict_proba(self,X_set:np.array) -> list:

        pred_probs=[]
        base_learners_predicted_probs = self._predict_probs_with_base_learners(X_set)

        for obs in range(X_set.shape[0]):
            base_learner_for_obs = [a[obs] for a in base_learners_predicted_probs]
            obs_average_pred_probs = np.mean(base_learner_for_obs, axis = 0) #list cohmprension
            pred_probs.append(obs_average_pred_probs)

        return pred_probs


    def predict(self,X_set:np.array) -> np.array:

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis = 1) #argomento maggiore di ogni riga di probabilità
        return preds


    #roberto riccio