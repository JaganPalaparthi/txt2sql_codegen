import random, sqlite3
from datetime import datetime, timedelta

DB = "financial_data.db"
conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS financial_data (
  id INTEGER PRIMARY KEY,
  trade_date TEXT,
  symbol TEXT,
  open REAL,
  close REAL,
  volume INTEGER
)
""")
c.execute("DELETE FROM financial_data")  # wipe any old data

start = datetime(2023, 1, 1)
symbols = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA"]
rows = []
for i in range(100):
    d = start + timedelta(days=i)
    sym = random.choice(symbols)
    # simulate some price movement
    o = round(random.uniform(100, 500), 2)
    c_ = round(o * random.uniform(0.95, 1.05), 2)
    vol = random.randint(1_000_000, 10_000_000)
    rows.append((d.strftime("%Y-%m-%d"), sym, o, c_, vol))

c.executemany("""
INSERT INTO financial_data (trade_date, symbol, open, close, volume)
VALUES (?, ?, ?, ?, ?)
""", rows)
conn.commit()
conn.close()
print(f"Populated {len(rows)} rows in {DB}")
