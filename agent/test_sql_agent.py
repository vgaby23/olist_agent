"""
test_sql_agent.py - Menjalankan SQL Agent pada beberapa pertanyaan contoh.

"""
from agent.sql_agent import ask_sql_agent, get_llm

# (pertanyaan, role, jawaban acuan yang dihitung manual dari database)
TESTS = [
    ("Berapa total pendapatan dari semua order yang sudah terkirim?", "company", "sekitar 13.220.248,93"),
    ("Berapa rata-rata waktu pengiriman dalam hari?", "company", "sekitar 12,6 hari"),
    ("Berapa persen order yang tiba tepat waktu?", "company", "sekitar 91,9%"),
    ("Metode pembayaran apa yang paling banyak dipakai?", "seller", "credit_card (74.579 pembayaran, order terkirim saja)"),
    ("Di negara bagian mana pelanggan terbanyak?", "seller", "SP (39.149 pelanggan unik dengan order terkirim)"),
    ("Bulan apa pendapatan tertinggi di tahun 2017-2018?", "seller", "2017-11 (sekitar 987.648)"),
    ("Siapa seller dengan pendapatan tertinggi?", "company", "seller_id 4869f7a5... (sekitar 226.987,93)"),
    ("Berapa pelanggan yang membeli lebih dari satu kali?", "company", "2.801 pelanggan unik (order terkirim saja)"),
    ("Berapa profit toko bulan lalu?", "company", "harus menolak: tidak ada data profit"),
]

if __name__ == "__main__":
    llm = get_llm()  # buat sekali supaya koneksi dipakai ulang
    for i, (question, role, expected) in enumerate(TESTS, start=1):
        print("=" * 70)
        print(f"[{i}] ({role}) {question}")
        print(f"    Jawaban acuan : {expected}")
        result = ask_sql_agent(question, role=role, llm=llm, verbose=False)
        print(f"    SQL           : {result['sql']}")
        print(f"    Jawaban agent : {result['answer']}")
