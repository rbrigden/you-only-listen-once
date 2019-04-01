from peewee import *
import datetime 
from playhouse.mysql_ext import MySQLConnectorDatabase


db = MySQLConnectorDatabase('my_database', host='1.2.3.4', user='mysql')

class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    username = CharField(unique = True)
    rec_id = IntegerField()


class Embedding(BaseModel):
    rec_id = IntegerField()
    data = BlobField()
    user_id = ForeignKeyField(User, backref='embeddings')

class Audio(BaseModel):
    rec_id = IntegerField()
    file_id = IntegerField()
    embedding_id = IntegerField()


db.connect()
db.create_tables([User, Embedding, Audio])

table1 = db['User']
table.insert(username = 'Huey', rec_id = 3)

for obj in table1:
    print (obj) 
