from tinydb import TinyDB, Query

# Initialize the database
db = TinyDB('db.json')

# Query the database
Fruit = Query()
result = db.search(Fruit.type == 'apple')
print(result)
