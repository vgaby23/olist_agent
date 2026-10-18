"""
sql_tools.py - "Pintu aman" ke database olist.db untuk SQL Agent.

Dua lapis pengaman:
1. Database dibuka dalam mode READ-ONLY (tidak mungkin menulis/menghapus).
2. Query dicek dulu: hanya satu perintah SELECT (atau WITH ... SELECT) yang boleh.
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "olist.db"
MAX_ROWS = 50  # batas baris hasil supaya jawaban tidak terlalu panjang

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def get_connection():
    """Buka database dalam mode read-only."""
    uri = DB_PATH.as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def validate_query(sql):
    """Pastikan query aman. Mengembalikan query yang sudah dirapikan, atau memunculkan ValueError."""
    query = sql.strip().rstrip(";").strip()
    if ";" in query:
        raise ValueError("Hanya satu query yang diizinkan.")
    if not re.match(r"^(select|with)\b", query, re.IGNORECASE):
        raise ValueError("Hanya query SELECT yang diizinkan.")
    if FORBIDDEN.search(query):
        raise ValueError("Query mengandung perintah yang tidak diizinkan.")
    return query


def run_query(sql):
    """Jalankan query yang aman, kembalikan hasil sebagai DataFrame (maksimal MAX_ROWS baris)."""
    query = validate_query(sql)
    conn = get_connection()
    try:
        cur = conn.execute(query)
        rows = cur.fetchmany(MAX_ROWS)
        columns = [d[0] for d in cur.description]
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


def get_schema():
    """Daftar tabel dan kolomnya dalam bentuk teks (nanti dimasukkan ke prompt agent)."""
    conn = get_connection()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        lines = []
        for t in tables:
            cols = [f"{c[1]} ({c[2]})" for c in conn.execute(f"PRAGMA table_info({t})")]
            lines.append(f"Tabel {t}: " + ", ".join(cols))
    finally:
        conn.close()
    return "\n".join(lines)


if __name__ == "__main__":
    # Pengujian cepat: python scripts/sql_tools.py
    print("=== SKEMA DATABASE ===")
    print(get_schema())

    print("\n=== TES 1: query SELECT yang valid ===")
    print(run_query("SELECT COUNT(*) AS total_order FROM orders;").to_string(index=False))

    print("\n=== TES 2: query dengan WITH (CTE) ===")
    print(run_query("WITH x AS (SELECT order_status FROM orders) SELECT COUNT(*) AS n FROM x").to_string(index=False))

    print("\n=== TES 3: query berbahaya harus DITOLAK ===")
    for bad in ["DELETE FROM orders",
                "DROP TABLE orders",
                "SELECT 1; DROP TABLE orders",
                "UPDATE orders SET order_status='x'"]:
        try:
            run_query(bad)
            print("GAGAL (seharusnya ditolak):", bad)
        except ValueError as e:
            print(f"Ditolak: {bad!r} -> {e}")

    print("\n=== TES 4: database memang read-only ===")
    conn = get_connection()
    try:
        conn.execute("CREATE TABLE tes (a INT)")
        print("GAGAL: database bisa ditulis!")
    except sqlite3.OperationalError as e:
        print("Benar, penulisan ditolak oleh database:", e)
    finally:
        conn.close()
