# Bagian C untuk cluster v4 (sama dengan seno/run7: seed -> LabelSpreading -> Scaler -> PCA -> LogReg)
# Perbedaan: ruang Z = ruang B v4 (dino_changed_embed), label = lembar_usulan_claude_v2.csv, fitur classifier + blok "changed".
import json, numpy as np, pandas as pd, joblib
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

FITUR_DIR = "../hafizh/v3.1"; B_DIR = "v4_perubahan"; LEMBAR = f"{B_DIR}/lembar_usulan_claude_v2.csv"; OUT = "run_v4"
SEED = 42; SEED_KEDEKATAN_MIN = 0.5; BOBOT_YAKIN = {"yakin": 1.0, "ragu": 0.5}; BOBOT_TANPA_NAMA = 0.0
LS_KNN, LS_ALPHA = 15, 0.2; C_PCA, LR_C = 128, 1.0
FITUR_BLOK = ["color_pre", "color_post", "changed"]
KELAS = ["Earthquake", "Explosion", "Fire", "Hurricane", "Tsunami", "Volcano"]
FILE = {"color_pre": "dino_color_pre", "color_post": "dino_color_post", "changed": "dino_changed_embed"}

ev = pd.read_parquet(f"{B_DIR}/events.parquet")
acuan = pd.read_parquet("v3_revisi2/events.parquet")            # urutan Id = sample_submission
assert (ev.Id.values == acuan.Id.values).all()
ids = ev.Id.values; N = len(ids)
lb = pd.read_csv(LEMBAR, encoding="utf-8-sig", dtype=str, keep_default_na=False)
NAMA = {int(r.event_id): (r.kelas_final or None) for r in lb.itertuples()}
YAKIN = {int(r.event_id): (r.yakin or None) for r in lb.itertuples()}
kelas_c = ev.event_id.map(NAMA).to_numpy(); yakin_c = ev.event_id.map(YAKIN).to_numpy()

l2 = lambda a: (a.astype(np.float64) / np.linalg.norm(a.astype(np.float64), axis=1, keepdims=True))
emb = {b: l2(np.load(f"{FITUR_DIR}/{FILE[b]}.npy")) for b in FILE}
Z = np.round(joblib.load(f"{B_DIR}/pca.joblib").transform(emb["changed"] - np.load(f"{B_DIR}/mean_x.npy")), 4)

ked = ev.kedekatan.to_numpy(float); punya = pd.notna(kelas_c)
is_seed = (ked >= SEED_KEDEKATAN_MIN) & punya
bobot = ked * np.where(punya, pd.Series(yakin_c).map(BOBOT_YAKIN).fillna(0).to_numpy(float), BOBOT_TANPA_NAMA)
y = np.full(N, -1); y[is_seed] = pd.Categorical(kelas_c[is_seed], categories=KELAS).codes
ls = LabelSpreading(kernel="knn", n_neighbors=LS_KNN, alpha=LS_ALPHA, max_iter=1000).fit(Z, y)
y_latih = np.array(KELAS)[ls.classes_][ls.label_distributions_.argmax(axis=1)]

X = np.hstack([emb[b] for b in FITUR_BLOK])
pipe = lambda: Pipeline([("scaler", StandardScaler()), ("pca", PCA(C_PCA, random_state=SEED)),
                         ("clf", LogisticRegression(class_weight="balanced", C=LR_C, max_iter=5000))])
latih = bobot > 0
cv = cross_val_predict(pipe(), X[latih], y_latih[latih], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                       params={"clf__sample_weight": bobot[latih]})
model = pipe().fit(X, y_latih, clf__sample_weight=bobot)
pred = model.predict(X)

run7 = pd.read_csv("../seno/run7_revisi2/submission_run7_revisi2.csv")
sub = pd.DataFrame({"Id": ids, "Phenomenon": pred}); assert (sub.Id.values == run7.Id.values).all()
sub.to_csv(f"{OUT}/submission_run_v4.csv", index=False)
pd.DataFrame({"Id": ids, "Phenomenon": pred}).to_csv(f"{OUT}/pred_train.csv", index=False)
joblib.dump({"pipeline": model, "fitur_blok": FITUR_BLOK, "kelas": list(model.classes_)}, f"{OUT}/model.joblib")
meta = {"versi": "run_v4", "oleh": "Fahmi", "label": LEMBAR, "clustering": B_DIR, "fitur_blok": FITUR_BLOK,
        "n_seed": int(is_seed.sum()), "foto_cluster_kosong": int((~punya).sum()),
        "cv_vs_label_latih": {"acc": round(accuracy_score(y_latih[latih], cv), 4), "macro_f1": round(f1_score(y_latih[latih], cv, average="macro"), 4)},
        "distribusi_prediksi": pd.Series(pred).value_counts().reindex(KELAS).fillna(0).astype(int).to_dict(),
        "sama_run7": round(float((pred == run7.Phenomenon.values).mean()), 4)}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(meta, indent=2, ensure_ascii=False))
