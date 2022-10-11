import os
from deta import Deta
from dotenv import load_dotenv

load_dotenv("datakey.env")

DETA_KEY = os.getenv("DETA_KEY")
deta = Deta(DETA_KEY)
db = deta.Base("user_db")

def insert_user(username , name , password):
        """ Returns the user on a successful user creation , otherwise raises an error """
        return db.put({"key": username, "name": name, "password": password})

def fetch_all_users():
    res = db.fetch()
    return res.items
print(fetch_all_users())

def get_user(username):
    return db.get(username)

def update_user(username , updates):
    return db.update(updates , username)

def delete_user(username):
    return db.delete(username)


