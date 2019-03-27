

class SpeakerClassificationProcessor:

    def __init__(self, ):
        self.x = 9

    def add_speaker(self, embeddings):
        """ Add speaker to the classification set

        :param embeddings: (N, D) matrix where N is the number of samples and D is the embedding dimensionality.
        :return: None
        """
        raise NotImplementedError

    def classify_speaker(self, embedding):
        """ Classify speech query
        :param embedding: D-dimensional embedding vector from speech query
        :return: user_id if query has positive result or None for failed identification.
        """
        raise NotImplementedError