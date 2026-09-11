"""
creer_base_donnees.py
----------------------
Construit la base de donnees SQLite (conteneurs.db) a partir de
outputs/conteneurs_features.csv. A executer UNE FOIS (ou a chaque fois
que les donnees changent).
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHEMIN_CSV = BASE_DIR.parent.parent / "outputs" / "conteneurs_features.csv"
CHEMIN_DB = BASE_DIR / "conteneurs.db"

print("[1/4] Chargement du CSV...")
df = pd.read_csv(CHEMIN_CSV, sep=";")
print(f"[OK] {len(df)} lignes chargees")

print("[2/4] Creation de la base SQLite...")
conn = sqlite3.connect(CHEMIN_DB)
df.to_sql("lots", conn, if_exists="replace", index=False)

print("[3/4] Creation des index (pour des recherches rapides)...")
conn.execute("CREATE INDEX IF NOT EXISTS idx_lot ON lots(ID_LOT)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_importateur ON lots(ID_IMPORTATEUR)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_marchandise ON lots(ID_MARCHANDISE)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_pays ON lots(ID_PAYS_CHARGEMENT)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_port ON lots(ID_PORT_CHARGEMENT)")

print("[4/4] Creation de la table d'historique des predictions...")
conn.execute("""
CREATE TABLE IF NOT EXISTS predictions_historique (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_prediction TEXT,
    id_marchandise INTEGER,
    id_port_chargement INTEGER,
    id_pays_chargement INTEGER,
    id_importateur INTEGER,
    lmd_lms INTEGER,
    id_terminal INTEGER,
    date_validation_bad TEXT,
    date_pointage TEXT,
    delai_estime_jours REAL,
    modele_utilise TEXT
)
""")
conn.commit()
conn.close()

print(f"\n[OK] Base de donnees creee : {CHEMIN_DB}")
