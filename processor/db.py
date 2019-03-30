from peewee import *
import datetime
import numpy as np

_db = None


def get_db_conn():
    global _db
    if _db is None:
        _db = SqliteDatabase("demo.db")
        return _db
    else:
        return _db


class BaseModel(Model):
    class Meta:
        database = get_db_conn()


class User(BaseModel):
    username = CharField(unique=True)


class Embedding(BaseModel):
    data = BlobField()
    user = ForeignKeyField(User, backref='embeddings')


class Audio(BaseModel):
    rec_id = CharField(unique=True)
    embedding = ForeignKeyField(Embedding, unique=True)


def create_embedding(user, embedding, rec_id):
    data = embedding.tostring()
    embedding = Embedding(data=data, user=user)
    embedding.save()
    audio = Audio(rec_id=rec_id, embedding=embedding)
    audio.save()


def load_embedding_data(embedding, dtype=np.float32):
    return np.fromstring(embedding.data).astype(dtype)

