# Kamus nama cluster v3-7 (dibuat dari cluster_names.json). Salin ke notebook final/01_model.ipynb (FAQ no. 7: kamus eksplisit).
# Nama kelas ditentukan dari montase tiap cluster (penilai: AI Claude, tanpa tinjauan tim). Bukan dari leaderboard.
# ID 0-39 = cluster Spectral K=40 (kecuali induk campur 2, 10, 29, 39 yang dipecah); ID 40-49 = sub-cluster hasil pemecahan.
NAMA_CLUSTER = {
    0: "Volcano",
    1: "Hurricane",
    3: "Volcano",
    4: "Tsunami",
    5: "Earthquake",
    6: "Volcano",
    7: "Hurricane",
    8: "Tsunami",
    9: "Fire",
    11: "Tsunami",
    12: "Fire",
    13: "Fire",
    14: "Earthquake",
    15: "Fire",
    16: "Hurricane",
    17: "Fire",
    18: "Hurricane",
    19: "Fire",
    20: "Fire",
    21: "Hurricane",
    22: "Fire",
    23: "Hurricane",
    24: "Hurricane",
    25: "Earthquake",
    26: "Fire",
    27: "Earthquake",
    28: "Fire",
    30: "Earthquake",
    31: "Earthquake",
    32: "Fire",
    33: "Earthquake",
    34: "Hurricane",
    35: "Hurricane",
    36: "Fire",
    37: "Hurricane",
    38: "Hurricane",
    40: "Hurricane",  # sub 2a
    41: "Tsunami",  # sub 2b
    42: "Hurricane",  # sub 10a
    43: "Volcano",  # sub 10b
    44: "Hurricane",  # sub 29a
    45: "Hurricane",  # sub 29b
    46: "Earthquake",  # sub 29c
    47: "Hurricane",  # sub 39a
    48: "Hurricane",  # sub 39b
    49: "Hurricane",  # sub 39c
}
YAKIN_CLUSTER = {
    0: "yakin",
    1: "yakin",
    3: "ragu",
    4: "yakin",
    5: "yakin",
    6: "ragu",
    7: "yakin",
    8: "ragu",
    9: "ragu",
    11: "ragu",
    12: "ragu",
    13: "ragu",
    14: "yakin",
    15: "yakin",
    16: "ragu",
    17: "ragu",
    18: "yakin",
    19: "yakin",
    20: "ragu",
    21: "yakin",
    22: "ragu",
    23: "yakin",
    24: "ragu",
    25: "ragu",
    26: "yakin",
    27: "ragu",
    28: "yakin",
    30: "ragu",
    31: "yakin",
    32: "ragu",
    33: "ragu",
    34: "yakin",
    35: "ragu",
    36: "yakin",
    37: "yakin",
    38: "yakin",
    40: "ragu",
    41: "yakin",
    42: "ragu",
    43: "ragu",
    44: "ragu",
    45: "ragu",
    46: "ragu",
    47: "ragu",
    48: "ragu",
    49: "ragu",
}

INDUK_DIPECAH = {2: [40, 41], 10: [42, 43], 29: [44, 45, 46], 39: [47, 48, 49]}
EVENT_ID_SHA1_40CLUSTER = "9517d553f3ad1ea535c7175b7a60f3c85bba2652"  # hash assignment 40-cluster asli (event_id induk, int32)
