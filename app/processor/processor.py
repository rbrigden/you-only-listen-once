import gin
import redis
import json
import time
from app.processor.speaker_classification_processor import SpeakerClassificationProcessor
from app.processor.speaker_embedding_processor import SpeakerEmbeddingProcessor
from app.processor.speech_rec_processor import SpeechRecognitionProcessor


import app.processor.utils as U
from multiprocessing import Process


def process(conn, request):
    """ Load a request and execute all subprocessors in parallel """
    print("Request is: {}".format(request))
    id_ = request["id"]
    audio_blob = conn.get('audio:{}'.format(id_))
    return request

def write_result(result):
    pass


class YoloProcessor:

    def __init__(self):
        self.speaker_classification =




    def run(self):
        conn = redis.Redis()

        while True:
            request = conn.blpop(["queue:requests"], 30)

            # TODO: we should be doing something smarter here
            if not request:
                continue

            result = process(conn, json.loads(request[1].decode('utf-8')))
            write_result(result)




if __name__ == "__main__":
    gin.external_configurable(redis.Redis, module="redis")
    gin.parse_config_file("app/processor/config/prod.gin")


    processor = YoloProcessor()
    main_loop()

