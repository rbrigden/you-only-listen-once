import torch
import numpy as np
import training.speaker_verification.eer as eer
import sklearn.linear_model
from collections import defaultdict
import training.speaker_verification.model as models
import redis
from io import BytesIO
import processor.db as db_core
import logging
import gin


@gin.configurable
class SpeakerClassificationProcessor:

    def __init__(self, mode='lr', decision_mode=False, fixed_thresh=None):
        self.mode = mode
        self.decision_mode = decision_mode
        self.redis_conn = redis.Redis()
        self.fixed_thresh = fixed_thresh
        self.logger = logging.getLogger('SpeakerClassificationProcessor')


    def update_speakers(self):
        """ Add speaker to the classification set. Modify self.weights which is (K, D) matrix with weights for each of the K speakers in the database

        :param embeddings: (N, D) matrix where N is the number of samples and D is the embedding dimensionality.
        :parem id_: ID for speaker to be added

        :return: None
        """

        # TODO: Uses stats.npy to normalize the embeddings

        internal_emb = defaultdict(list)


        # Load external embeddings
        external_embeddings = self.redis_conn.get('external')
        external_embeddings = np.load(BytesIO(external_embeddings))

        # Split the external embeddings
        held_out_prop = int(0.2 * len(external_embeddings))
        external_train_idxs = np.arange(len(external_embeddings))
        np.random.shuffle(external_train_idxs)
        external_embeddings_train = external_embeddings[external_train_idxs[held_out_prop:]]
        external_embeddings_val = external_embeddings[external_train_idxs[:held_out_prop]]

        for user in db_core.User.select():
            for embedding in user.embeddings:
                internal_emb[user.id].append(db_core.load_embedding_data(embedding, dtype=np.float64))

        self.modeldict = defaultdict(list)

        for internal_id, internal_embeddings in internal_emb.items():
            neg_embeddings = [other_embeddings for other_id, other_embeddings in internal_emb.items() if
                              internal_id != other_id]
            pos_embeddings = np.stack(internal_embeddings)

            if len(neg_embeddings) >= 5:
                neg_embeddings = np.concatenate([np.stack(x) for x in neg_embeddings], axis=0)

                # Split the internal embeddings
                held_out_prop = int(0.1 * len(neg_embeddings))
                internal_train_idxs = np.arange(len(neg_embeddings))
                np.random.shuffle(internal_train_idxs)
                internal_embeddings_train = neg_embeddings[internal_train_idxs[held_out_prop:]]
                internal_embeddings_val = neg_embeddings[internal_train_idxs[:held_out_prop]]
                embeddings_train = np.concatenate((internal_embeddings_train, external_embeddings_train), axis=0)
                # embeddings_train = external_embeddings_train
                embeddings_val = np.concatenate((external_embeddings_val, internal_embeddings_val), axis=0)
            else:
                embeddings_train = external_embeddings_train
                embeddings_val = external_embeddings_val

            lr_model = self.getLogisticRegressionParams(pos_embeddings, embeddings_train)
            svm_model = self.getSVMParams(pos_embeddings, embeddings_train)

            lr_eer_labels, lr_scores = self.get_eer_inputs(lr_model, embeddings_val, pos_embeddings)
            svm_eer_labels, svm_scores = self.get_eer_inputs(svm_model, embeddings_val, pos_embeddings)

            # Calculate EER and Probability-Threshold for Each Speaker
            lr_eer, lr_threshold = eer.EER(lr_eer_labels, lr_scores)
            svm_eer, svm_threshold = eer.EER(svm_eer_labels, svm_scores)


            internal_user = db_core.User.get(db_core.User.id == internal_id)
            self.logger.info("LR Speaker Model for {}. EER: {}, Thresh: {}".format(internal_user.username, float(lr_eer), float(lr_threshold)))
            self.logger.info("SVM Speaker Model for {}. EER: {}, Thresh: {}".format(internal_user.username, float(svm_eer), float(svm_threshold)))

            db_core.write_speaker_model(internal_user, lr_model, float(lr_threshold))
            db_core.write_speaker_model_svm(internal_user, svm_model, float(svm_threshold))


    def classify_speaker(self, embedding):
        """ Classify speech query

        :param embedding: D-dimensional embedding vector from speech query

        :return: user_id if query has positive result or None for failed identification.
        """

        users = [user for user in db_core.User.select()]

        if self.mode == 'lr':
            speaker_models, thresholds = zip(*[db_core.load_speaker_model(user, sklearn.linear_model.LogisticRegression(solver='liblinear', class_weight='balanced')) for user in users])
        elif self.mode == 'svm':
            speaker_models, thresholds = zip(*[db_core.load_speaker_model_svm(user) for user in users])
        else:
            raise ValueError("Invalid mode")

        user_ids = [user.id for user in users]

        targets, decisions = self.get_target(user_ids, speaker_models, thresholds, embedding)
        labels, raw_decisions = zip(*decisions)


        if self.decision_mode and self.mode == 'svm':
            if sum(raw_decisions) > 1:
                decision = max(targets, key=lambda x: x[1])[0]
            elif sum(raw_decisions) == 1:
                decision = max(decisions, key=lambda item: item[1])[0]
            else:
                decision = None
        else:
            decision = None if len(targets) == 0 else max(targets, key=lambda x: x[1])[0]

        return decision

        # bestlabel = self.get_argmax_target(targets)
        # return bestlabel

    def getLogisticRegressionParams(self, positives, negatives):
        """ Train for Each Speaker vs Rest-Of-Speakers. One vs Rest Binary Classification

        :param target: (Z, D) matrix which contains Z D-dimensional embeddings for Z utterances ofpostive-Labeled speaker that is to be classified 
        :return: Logistic Regression model for target-speaker
        """

        positiveLabels = np.ones(len(positives))
        negativeLabels = np.zeros(len(negatives))
        XLab = np.concatenate((positives, negatives), axis=0)
        YLab = np.concatenate((positiveLabels, negativeLabels))
        f = sklearn.linear_model.LogisticRegression(solver='liblinear', class_weight=None).fit(XLab, YLab)
        return f

    def getSVMParams(self, positives, negatives):
        """ Train for Each Speaker vs Rest-Of-Speakers. One vs Rest Binary Classification

        :param target: (Z, D) matrix which contains Z D-dimensional embeddings for Z utterances ofpostive-Labeled speaker that is to be classified
        :return: Logistic Regression model for target-speaker
        """

        positiveLabels = np.ones(len(positives))
        negativeLabels = np.zeros(len(negatives))
        XLab = np.concatenate((positives, negatives), axis=0)
        YLab = np.concatenate((positiveLabels, negativeLabels))
        f = sklearn.svm.SVC(gamma='scale', probability=True).fit(XLab, YLab)
        return f


    def get_eer_inputs(self, model, embeddings_val, pos_embeddings):
        """ For a particular model, get list of EER labels and scores to pass into EER function

        :param model: Logistic Regression model for target-Speaker
        :param target_label: Unique Speaker_ID for target-Speker
        
        Returns:
        :list eer_labels: 0/1 depending on if target-Speaker or not
        :list scores: Probability Scores for all Speakers on target-Model

        """
        neg_probs = list(map(lambda x: model.predict_proba([x])[0][1], embeddings_val))
        neg_labels = np.zeros_like(neg_probs)

        pos_probs = list(map(lambda x: model.predict_proba([x])[0][1], pos_embeddings))
        pos_labels = np.ones_like(pos_probs)
        return np.concatenate([neg_labels, pos_labels]), np.concatenate([neg_probs, pos_probs])

    def get_target(self, user_ids, speaker_models, thresholds, embedding):
        """ Get all target labels that pass the EER_threshold by calculating probability using the Logistic Regression model for a particular embedding

         :param embedding: D-dimensional embedding vector from speech query
         :return targets: List of possible target labels for a particular embedding
         """
        targets = []
        decisions = []
        for label, model, threshold in zip(user_ids, speaker_models, thresholds):
            prob = model.predict_proba([embedding])[0][1]
            self.logger.info("P({}) {}".format(db_core.User.get(db_core.User.id == label).username, prob))
            if self.mode == 'lr':
                threshold = self.fixed_thresh if self.fixed_thresh else threshold
                if prob > threshold:
                    targets.append((label, prob))
            elif self.mode == 'svm':
                decision = model.predict([embedding])[0]
                decisions.append((label, decision))
                if self.fixed_thresh and prob > self.fixed_thresh:
                    targets.append((label, prob))
        return targets, decisions


    def get_argmax_target(self, targets):
        """ Gets best target-Label amongst all possible targets that passed the threshold

         :param embedding: D-dimensional embedding vector from speech query
         :param targets: Possible target labels to get argmax probability from

         :return max_label: Target Label with best Argmax Probability
         """

        if len(targets) == 1:
            return targets[0][0]

        labels, probs = [np.array(x) for x in zip(*targets)]

        new_scores = np.zeros_like(probs)
        for i in range(probs.shape[0]):
            mask = np.ones_like(probs)
            mask[i] = 0
            mu = (probs * mask).sum() / mask.sum()
            new_scores[i] = probs[i] / mu

        self.logger.info("Probs: {}".format([(db_core.User.get(db_core.User.id == labels[idx]).username, probs[idx]) for idx in range(probs.shape[0])]))
        self.logger.info("Scores: {}".format([(db_core.User.get(db_core.User.id == labels[idx]).username, new_scores[idx]) for idx in range(new_scores.shape[0])]))


        return labels[np.argmax(new_scores)]

@gin.configurable
def main(mode):
    if mode == "plot_embeddings":
        from inference.demo import plot_embeddings
        import matplotlib.pyplot as plt
        embeddings = []
        labels = []
        for user in db_core.User.select():
            for embedding in user.embeddings:
                embeddings.append(db_core.load_embedding_data(embedding, dtype=np.float64))
                labels.append(user.username)
        embeddings = np.stack(embeddings)
        plot_embeddings(embeddings, labels)
        plt.savefig("internal_embeddings.png")
    elif mode == "retrain_speaker_models":
        sc = SpeakerClassificationProcessor()
        sc.update_speakers()
    else:
        raise ValueError("Invalid mode specified")



if __name__ == '__main__':
    import sklearn.linear_model

    import processor.db as db_core
    import processor.utils as U
    import sklearn.linear_model
    import os


    gin.external_configurable(redis.Redis, module="redis")
    gin.external_configurable(sklearn.linear_model.LogisticRegression)

    gin.parse_config_file("processor/config/speaker_classification.gin")
    # Set up processor logging
    logging.basicConfig(filename='yolo_processor.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


    db = db_core.get_db_conn()
    db.connect()
    db.create_tables([db_core.User, db_core.Embedding, db_core.Audio, db_core.SpeakerModel, db_core.SpeakerModelSVM])
    main()