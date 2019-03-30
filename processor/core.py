import gin
import redis
import json
import time
# from processor.speaker_classification_processor import SpeakerClassificationProcessor
from processor.speaker_embedding_processor import SpeakerEmbeddingProcessor
from processor.speaker_embedding_processor import SpeakerEmbeddingInference
from processor.speech_rec_processor import SpeechRecognitionProcessor
from processor.audio_processor import AudioProcessor
import processor.db as db_core
import processor.utils as U
from multiprocessing import Process
import logging
from peewee import SqliteDatabase



class YoloProcessor:

    def __init__(self):
        # self.speaker_classification = SpeakerClassificationProcessor()
        self.embedding_processor = SpeakerEmbeddingProcessor()
        self.audio_processing = AudioProcessor()
        self.redis_conn = redis.Redis()

        # Set up processor logging
        logging.basicConfig(filename='yolo_processor.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info("Yolo Processor")
        self.logger = logging.getLogger('yoloProcessor')

        # database
        self.db = self._init_db()



    def run(self):

        while True:
            request = self.redis_conn.blpop(["queue:requests"], 30)

            # TODO: we should be doing something smarter here
            if not request:
                continue

            result = self._process(json.loads(request[1].decode('utf-8')))


    def _init_db(self):
        db = db_core.get_db_conn()
        db.connect()
        db.create_tables([db_core.User, db_core.Embedding, db_core.Audio])
        return db

    def _process(self, request):

        # Parse request data
        id_ = request["id"]
        request_type = request["type"]
        self.logger.log(logging.INFO, "{} request {} received".format(request_type, id_))

        if request_type == "register":
            self._register(id_)
        elif request_type == "authenticate":
            self._authenticate(id_)

        return request

    def _authenticate(self, id_):
        audio_bytes = self.redis_conn.get('audio:{}'.format(id_))
        U.play_audio(audio_bytes)
        processed_utterance = self.audio_processing(audio_bytes)
        embeddings = self.embedding_processor([processed_utterance])
        self.logger.log(logging.INFO, "Authentication complete for request {}".format(id_))

    def _register(self, id_):
        audio_bytes = self.redis_conn.get('audio:{}'.format(id_))
        U.play_audio(audio_bytes)
        processed_utterances = self.audio_processing(audio_bytes)
        embeddings = self.embedding_processor([processed_utterances])




        self.logger.log(logging.INFO, "Registration complete for request {}".format(id_))


if __name__ == "__main__":
    gin.external_configurable(redis.Redis, module="redis")

    gin.parse_config_file("processor/config/prod.gin")
    processor = YoloProcessor()
    processor.run()
