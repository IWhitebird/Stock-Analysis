import streamlit_authenticator as stauth
import database as db

usernames =["pparker","shreyas"]
names = ["Peter Parker","Shreyas Patange"]
passwords = ["abc123","def123"]
hashed_passwords = stauth.Hasher(passwords).generate()

for (username , name , hash_password) in zip(usernames , names , hashed_passwords):
    db.insert_user(username, name, hash_password)