import sys
import sqlite3
db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
cur.execute("ANALYZE")
cur.execute("VACUUM")
db.commit()
db.close()