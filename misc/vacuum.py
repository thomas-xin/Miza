import sys
import sqlite3
db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
try:
    cur.execute("ANALYZE")
except Exception:
    from traceback import format_exc
    print(f"{sys.argv[1]}:\n{format_exc()}")
    raise SystemExit(1)
cur.execute("VACUUM")
db.commit()
db.close()