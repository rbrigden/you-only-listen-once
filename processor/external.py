from processor.speaker_embedding_processor import SpeakerEmbeddingProcessor
from processor.audio_processor import AudioProcessor

def embeddings_from_wav_set(wav_file_paths):
    get_embeddings = SpeakerEmbeddingProcessor()
    get_spects = AudioProcessor()
