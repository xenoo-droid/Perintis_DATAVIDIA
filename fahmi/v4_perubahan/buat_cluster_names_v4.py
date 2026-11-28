# Membuat cluster_names.json (format sama dengan v3) dari lembar nama v4, dan melengkapi meta.json untuk notebook Seno.
import json, pandas as pd
LEMBAR = "lembar_usulan_claude_v2.csv"     # ganti ke "lembar_bersama.csv" setelah tim meninjau & menyalin
lb = pd.read_csv(LEMBAR, encoding="utf-8-sig", dtype=str, keep_default_na=False)
names = {str(int(r.event_id)): {"kelas": r.kelas_final, "yakin": r.yakin or "ragu", "alasan": r.alasan,
                                "penilai": ["AI (Claude)"], "sumber": LEMBAR} for r in lb.itertuples()}
json.dump(names, open("cluster_names.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
m = json.load(open("meta.json", encoding="utf-8"))
m["input"]["fitur"] = "hafizh/v3.1"                                  # format yang dicek notebook Seno
m["pengaturan"]["blok_z"] = ["changed"]                              # ruang Z B v4 = dino_changed_embed
m["nama"] = {"sumber": LEMBAR, "penilai": "AI (Claude), cek warna grayscale", "kosong = bobot 0": [k for k, v in names.items() if not v["kelas"]]}
json.dump(m, open("meta.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("cluster_names.json:", len(names), "cluster |", sum(1 for v in names.values() if not v["kelas"]), "kosong")
