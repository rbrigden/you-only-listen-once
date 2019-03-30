import torch
import numpy as np
import training.speaker_verification.eer as eer
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import training.speaker_verification.model as models

class SpeakerClassificationProcessor:

    def __init__(self):

        self.weights = []
        self.labeldict = defaultdict(list)
        self.traindict = defaultdict(list)
        self.helddict = defaultdict(list)
        self.threshold_dict = defaultdict(list)

    def add_speaker(self, embeddings, id_):
        """ Add speaker to the classification set. Modify self.weights which is (K, D) matrix with weights for each of the K speakers in the database

        :param embeddings: (N, D) matrix where N is the number of samples and D is the embedding dimensionality.
        :parem id_: ID for speaker to be added

        :return: None
        """

        for index in embeddings:
            self.labeldict[id_].append(embeddings[index])

        # Split Training/Held-Out 80-20
        train_index = int(len(self.labeldict[id_]) * 0.8)
        self.traindict[id_] = self.labeldict[id_][:train_index]
        self.helddict[id_] = self.labeldict[id_][train_index:]
      
        self.modeldict = defaultdict(list)
        # Train Logistic Regression Parameters for Each Speaker  
        for label in self.traindict:
            target = self.traindict[label]
            del self.traindict[label]
            self.modeldict[label]  = self.getLogisticRegressionParams(target)
            self.traindict[label] = target

        # Get EER and Threshold for Each Speaker
        for label in self.modeldict:
            eer_labels, scores = self.get_eer_inputs(self.modeldict[label], label)
        
            # Calculate EER and Probability-Threshold for Each Speaker
            self.threshold_dict[label] = float((eer.EER(eer_labels, scores)[1]).tolist())

    def classify_speaker(self, embedding):
        """ Classify speech query

        :param embedding: D-dimensional embedding vector from speech query

        :return: user_id if query has positive result or None for failed identification.
        """
        
        
        targets = self.get_target(embedding)
        if targets == []:
            return None
        bestlabel = self.get_argmax_target(embedding, targets)
        return bestlabel
        

    def getLogisticRegressionParams(self, target):
        """ Train for Each Speaker vs Rest-Of-Speakers. One vs Rest Binary Classification

        :param target: (Z, D) matrix which contains Z D-dimensional embeddings for Z utterances ofpostive-Labeled speaker that is to be classified 
        :return: Logistic Regression model for target-speaker
        """

        positiveLabels = np.ones(len(target))
        target = np.asarray(target)
        negatives = []
        for label in self.traindict:
            negatives.append(self.traindict[label])
        n = np.asarray(negatives[0])
        for i in range(1, len(negatives)):
            n = np.concatenate((n, negatives[i]), axis = 0)
        negativeLabels = np.zeros(len(n))
        XLab = np.concatenate((target, n), axis = 0)
        YLab = np.concatenate((postiveLabels, negativesLabels))
        f = LogisticRegression(solver = 'liblinear', penalty = 'l2', class_weight = 'balanced').fit(XLab, YLab)
        return f 

    def get_err_inputs(self, model, target_label):
        """ For a particular model, get list of EER labels and scores to pass into EER function

        :param model: Logistic Regression model for target-Speaker
        :param target_label: Unique Speaker_ID for target-Speker
        
        Returns:
        :list eer_labels: 0/1 depending on if target-Speaker or not
        :list scores: Probability Scores for all Speakers on target-Model

        """
        eer_labels = scores = []
        for label in self.helddict:
            res = list(map(lambda x: model.predict_proba([x])[0][1], self.helddict[label]))
            scores.extend(res)
            if label == target_label:
                eer_labels.extend([1] * len(res))
            else:
                eer_labels.extend([0] * len(res))
        return eer_labels, scores

    
    def get_target(embedding):
       """ Get all target labels that pass the EER_threshold by calculating probability using the Logistic Regression model for a particular embedding

        :param embedding: D-dimensional embedding vector from speech query
        :return targets: List of possible target labels for a particular embedding
        """ 
        targets = []
        for label in self.modeldict:
            model = modeldict[label]
            prob = model.predict_proba([embedding])[0][1]
            if (prob > self.threshold_dict[label]):
                targets.append(label)
        return targets

    
    def get_argmax_target(embedding, targets):
       """ Gets best target-Label amongst all possible targets that passed the threshold

        :param embedding: D-dimensional embedding vector from speech query
        :param targets: Possible target labels to get argmax probability from

        :return max_label: Target Label with best Argmax Probability
        """ 

        if len(targets) == 1:
            return targets[0]
        maximum = -1.0
        max_label = None
    
        every = dict()
        for label in targets:
            model = self.modeldict[label]
            every[label] = model.predict_proba([embedding])[0][1]

        for label in every:
            numerator = every[label]
            del every[label]
            denom = sum(every.values.())/len(every)
            arg = numerator/denom
            if (maximum == -1.0 or arg > maximum):
                maximum = arg
                max_label = label
            every[label] = numerator
        return max_label
        


        
