import pymongo


class MongoDB:
    def __init__(self, host: str, database: str, collection: str):
        self.host = host
        self.database = database
        self.client = pymongo.MongoClient(host)
        self.db = self.client[database]
        self.collection = self.db[collection]

    def add_data(self, data: dict):
        self.collection.insert_one(data)
