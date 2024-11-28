import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASETS_DIR = os.path.join(BASE_DIR, '../data/datasets')
    LOGS_DIR = os.path.join(BASE_DIR, '../data/logs')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
