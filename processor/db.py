from peewee import *
import datetime
import numpy as np

_db = None


def get_db_conn():
    global _db
    if _db is None:
        _db = SqliteDatabase("demo1.db")
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

class SpeakerModel(BaseModel):
    user = ForeignKeyField(Embedding, unique=True)
    coef = BlobField()
    intercept = BlobField()
    n_iter = BlobField()
    threshold = FloatField()

class Audio(BaseModel):
    rec_id = CharField(unique=True)
    embedding = ForeignKeyField(Embedding, unique=True)


def write_speaker_model(user, lr_model, threshold):
    coef_data = lr_model.coef_.astype(np.float64).tostring()
    intercept_data = lr_model.intercept_.astype(np.float64).tostring()
    n_iter_data = lr_model.n_iter_.astype(np.float64).tostring()

    q = SpeakerModel.select().where(SpeakerModel.user == user)
    if q.exists():
        q = SpeakerModel.update({SpeakerModel.coef: coef_data,
                  SpeakerModel.intercept: intercept_data,
                  SpeakerModel.n_iter: n_iter_data,
                  SpeakerModel.threshold: threshold}).where(SpeakerModel.user == user)
        q.execute()
    else:
        sm = SpeakerModel(user=user,
                          coef=coef_data,
                          intercept=intercept_data,
                          n_iter=n_iter_data,
                          threshold=threshold)
        sm.save()


def load_speaker_model(user, lr_model):
    sm = SpeakerModel.get(user == user)
    coef_data = np.fromstring(sm.coef).astype(np.float32)
    intercept_data = np.fromstring(sm.intercept).astype(np.float32)
    n_iter_data = np.fromstring(sm.n_iter).astype(np.float32)
    lr_model.coef_ = coef_data
    lr_model.intercept_ = intercept_data
    lr_model.n_iter_ = n_iter_data
    return lr_model, sm.threshold


def create_embedding_record(user, embedding, rec_id):
    data = embedding.astype(np.float64).tostring()
    embedding = Embedding(data=data, user=user)
    embedding.save()
    audio = Audio(rec_id=rec_id, embedding=embedding)
    audio.save()


def load_embedding_data(embedding, dtype=np.float32):
    return np.fromstring(embedding.data).astype(dtype)


if __name__ == "__main__":
    db = get_db_conn()
    db.connect()
    db.create_tables([User, Embedding, Audio, SpeakerModel])

    # Add user
    # User.create(username="Ryan")

    user = list(User.select())
    sms = list(SpeakerModel.select())
    print("hello")
    db.close()
