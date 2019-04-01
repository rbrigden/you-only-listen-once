from processor.db import *
import argparse

def clear_all_db_records():
    tables = [SpeakerModel, User, Embedding, Audio]
    for table in tables:
        for x in table.select():
            x.delete_instance()

def create_fixtures(fixtures_path):
    raise NotImplementedError




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-path", type=str)





