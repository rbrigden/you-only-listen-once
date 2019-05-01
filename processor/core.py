import gin
import redis
import json
import torch
import time
from processor.speaker_classification_processor import SpeakerClassificationProcessor
from processor.speaker_embedding_processor import SpeakerEmbeddingProcessor
from processor.speaker_embedding_processor import SpeakerEmbeddingInference
from processor.speech_rec_processor import SpeechRecognitionProcessor
from processor.external import load_voxceleb_embeddings
from processor.audio_processor import AudioProcessor
import processor.db as db_core
import processor.utils as U
import sklearn.linear_model
import os

from multiprocessing import Process
import logging
from peewee import SqliteDatabase
import numpy as np
from io import BytesIO

@gin.configurable
class YoloProcessor:

    def __init__(self,
                 registration_split=3,
                 load_external=False,
                 load_fixtures=False):
        self.registration_split = registration_split
        self.load_external = load_external
        self.speaker_classification = SpeakerClassificationProcessor()
        self.embedding_processor = SpeakerEmbeddingProcessor()
        self.audio_processing = AudioProcessor()
        self.redis_conn = redis.Redis()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Set up processor logging
        logging.basicConfig(
                            filename='yolo_processor.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info("Yolo Processor")
        self.logger = logging.getLogger('yoloProcessor')

        # database
        self.db = self._init_db()

        # setup
        self._setup()

        # demo fixtures
        if load_fixtures:
            db_core.clear_all_db_records()
            self._add_fixtures("internal_data/")


    def _add_fixtures(self, fixtures_path):

        for dirname in os.listdir(fixtures_path):
            speaker_path = os.path.join(fixtures_path, dirname)

            # Add user to the database
            user = db_core.User(username=dirname)
            user.save()

            all_processed_utterances = []
            rec_ids = []
            for filename in os.listdir(speaker_path):
                rec_ids.append(filename)
                filepath = os.path.join(speaker_path, filename)
                self.logger.info("Get spectrogram for {}".format(filename))
                processed_utterances = self.audio_processing.from_file(filepath)
                all_processed_utterances.extend(processed_utterances)

            self.logger.info("Get embeddings for all audio fixtures")
            embeddings = self.embedding_processor(all_processed_utterances)

            embeddings = embeddings.numpy()

            for i, rec_id in zip(range(embeddings.shape[0]), rec_ids):
                embedding_data = embeddings[i]
                db_core.create_embedding_record(user=user, embedding=embedding_data, rec_id=rec_id)

        self.speaker_classification.update_speakers()



    def run(self):

        while True:
            request = self.redis_conn.blpop(["queue:requests"], 30)

            # TODO: we should be doing something smarter here
            if not request:
                continue

            result = self._process(json.loads(request[1].decode('utf-8')))

    def _setup(self):
        # Load external dataset embeddings

        if not self.redis_conn.exists('external') or self.load_external:
            external_embeddings = []
            external_embeddings.append(load_voxceleb_embeddings())
            external_embeddings = np.concatenate(external_embeddings, axis=0)

            with BytesIO() as b:
                np.save(b, external_embeddings)
                serialized_embeddings = b.getvalue()
            self.redis_conn.set('external', serialized_embeddings)

    def _init_db(self):
        db = db_core.get_db_conn()
        db.connect()
        db.create_tables([db_core.User, db_core.Embedding, db_core.Audio, db_core.SpeakerModel])
        return db

    def _process(self, request):

        # Parse request data
        request_id = request["id"]
        request_type = request["type"]
        self.logger.log(logging.INFO, "{} request {} received".format(request_type, request_id))

        if request_type == "register":
            username = request['name']
            self._register(request_id, username)
        elif request_type == "authenticate":
            self._authenticate(request_id)

        return request

    def _authenticate(self, id_):
        audio_bytes = self.redis_conn.get('audio:{}'.format(id_))
        #U.play_audio(audio_bytes)
        processed_utterance = self.audio_processing(audio_bytes)
        embeddings = self.embedding_processor(processed_utterance)
        id_decision = self.speaker_classification.classify_speaker(embeddings.squeeze(0).numpy())

        if id_decision is None:
            user = None
        else:
            user = db_core.User.get(db_core.User.id == id_decision)

        # Send the result to the client
        self.redis_conn.set("result:{}".format(id_), "Unknown" if user is None else user.username)

        self.logger.log(logging.INFO, "Decision is: {}".format("Unknown" if user is None else user.username))
        self.logger.log(logging.INFO, "Authentication complete for request {}".format(id_))
        return user


    def _register(self, request_id, username):
        # Add user to the database
        user = db_core.User(username=username)
        user.save()

        audio_bytes = self.redis_conn.get('audio:{}'.format(request_id))
        processed_utterances = self.audio_processing(audio_bytes, split=4)
        embeddings = self.embedding_processor(processed_utterances)
        embeddings = embeddings.numpy()

        for i in range(embeddings.shape[0]):
            embedding_data = embeddings[i]
            db_core.create_embedding_record(user=user, embedding=embedding_data, rec_id="{}:{}".format(request_id, i))

        self.speaker_classification.update_speakers()

        self.logger.log(logging.INFO, "Registration complete for request {}".format(request_id))


if __name__ == "__main__":
    gin.external_configurable(redis.Redis, module="redis")
    gin.external_configurable(sklearn.linear_model.LogisticRegression, module="sklearn.linear_model")

    gin.parse_config_file("processor/config/prod.gin")
    processor = YoloProcessor()

    try:
        processor.run()
    except KeyboardInterrupt as e:
        processor.db.close()

    processor.db.close()
