import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
np.set_printoptions(suppress=True, precision=4)

DATA = "placement_predict_50k Dataset.csv"   # <-- put the CSV next to this notebook
df = pd.read_csv(DATA)
print(df.shape)
df.head()

NUM = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "Salary Package"]
df[NUM].agg(["min", "max", "mean", "std"]).T.round(3)

# Euclidean distance between student 1 and student 2 on two raw features
a = df.loc[0, ["CGPA", "AttendancePercent"]].values.astype(float)
b = df.loc[1, ["CGPA", "AttendancePercent"]].values.astype(float)
contrib = (a - b) ** 2
print("A =", a, "  B =", b)
print("(CGPA diff)^2   =", round(contrib[0], 4))
print("(Attend diff)^2 =", round(contrib[1], 4))
print("euclidean       =", round(np.sqrt(contrib.sum()), 4))
print(f"attendance share of the distance = {100*contrib[1]/contrib.sum():.2f} %")

cg = df["CGPA"]
mu, sd_pop, sd_samp = cg.mean(), cg.std(ddof=0), cg.std(ddof=1)
print(f"n     = {len(cg)}")
print(f"mu    = {mu:.6f}")
print(f"sigma = {sd_pop:.6f}   (ddof=0, what sklearn uses)")
print(f"sigma = {sd_samp:.6f}   (ddof=1, what pandas uses)")

# hand calculation for the first five students
hand = df.loc[:4, ["StudentID", "CGPA"]].copy()
hand["z = (x-mu)/sigma"] = ((hand["CGPA"] - mu) / sd_pop).round(4)
hand

scaler = StandardScaler()
scaler.fit(df[["CGPA"]])              # LEARN mu and sigma
z = scaler.transform(df[["CGPA"]])    # APPLY the formula

print("mean_ :", scaler.mean_)
print("scale_:", scaler.scale_)
print("var_  :", scaler.var_)
print("first 5:", z[:5].ravel())
print(f"after -> mean = {z.mean():.10f}   std = {z.std():.10f}")
print(f"range -> min  = {z.min():.4f}   max = {z.max():.4f}")

# all five columns at once
X = df[NUM].fillna(df[NUM].median())
Z = StandardScaler().fit_transform(X)
print(pd.DataFrame(Z[:5], columns=NUM).round(4).to_string(index=False))
print("\nmeans after:", np.round(Z.mean(0), 12))
print("stds  after:", np.round(Z.std(0), 6))

xmin, xmax = cg.min(), cg.max()
print(f"min = {xmin}   max = {xmax}   range = {xmax - xmin}")

hand = df.loc[:4, ["StudentID", "CGPA"]].copy()
hand["x_scaled"] = ((hand["CGPA"] - xmin) / (xmax - xmin)).round(4)
print(hand.to_string(index=False))

mm = MinMaxScaler().fit(df[["CGPA"]])
m = mm.transform(df[["CGPA"]])
print("data_min_ :", mm.data_min_, "  data_max_ :", mm.data_max_)
print("scale_    :", mm.scale_,    "  min_      :", mm.min_)
print("first 5   :", m[:5].ravel())
print(f"min = {m.min():.4f}  max = {m.max():.4f}  mean = {m.mean():.4f}")

m2 = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df[["CGPA"]])
print("feature_range=(-1,1) first 5:", m2[:5].ravel())

s  = df["Salary Package"].copy()
s2 = s.copy(); s2.iloc[7] = 2600.0          # the typo
X2 = s2.values.reshape(-1, 1)
ok = np.ones(len(s), bool); ok[7] = False   # everyone except the corrupted row

rows = []
for name, Sc in [("MinMaxScaler", MinMaxScaler), ("StandardScaler", StandardScaler), ("RobustScaler", RobustScaler)]:
    clean = Sc().fit_transform(s.values.reshape(-1, 1)).ravel()
    dirty = Sc().fit_transform(X2).ravel()
    rows.append([name,
                 round(clean[ok].max() - clean[ok].min(), 4),
                 round(dirty[ok].max() - dirty[ok].min(), 4)])
t = pd.DataFrame(rows, columns=["scaler", "clean width", "width after the typo"])
t["% of range kept"] = (100 * t["width after the typo"] / t["clean width"]).round(1)
t

X = df[["CodingTestScore"]].fillna(df["CodingTestScore"].median())
q1, q2, q3 = np.percentile(X.values, [25, 50, 75])
print(f"Q1 = {q1}   median = {q2}   Q3 = {q3}   IQR = {q3-q1:.4f}")

hand = df.loc[:4, ["StudentID", "CodingTestScore"]].copy()
hand["robust"] = ((hand["CodingTestScore"] - q2) / (q3 - q1)).round(4)
print(hand.to_string(index=False))

rb = RobustScaler().fit(X)
r = rb.transform(X)
print("center_ (median):", rb.center_)
print("scale_  (IQR)   :", rb.scale_)
print("first 5         :", r[:5].ravel())
print(f"median after = {np.median(r):.6f}   IQR after = {np.percentile(r,75)-np.percentile(r,25):.6f}")

# head-to-head on the same column
cmp = pd.DataFrame({
    "raw":      X.values.ravel(),
    "standard": StandardScaler().fit_transform(X).ravel(),
    "minmax":   MinMaxScaler().fit_transform(X).ravel(),
    "robust":   RobustScaler().fit_transform(X).ravel(),
    "maxabs":   MaxAbsScaler().fit_transform(X).ravel()})
cmp.describe().T.round(4)

ma = MaxAbsScaler().fit(df[["Salary Package"]])
print("max_abs_:", ma.max_abs_)
print("raw    :", df["Salary Package"].head().values)
print("scaled :", ma.transform(df[["Salary Package"]])[:5].ravel())

rows = df.loc[:2, ["AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]].fillna(0)
print("raw rows:"); print(rows.to_string(index=False))

l2 = np.sqrt((rows.values ** 2).sum(axis=1))
print("\nL2 norms:", np.round(l2, 4))
print("L1 norms:", np.round(np.abs(rows.values).sum(axis=1), 4))

for norm in ["l2", "l1", "max"]:
    out = Normalizer(norm=norm).fit_transform(rows)
    print(f"\n{norm}-normalised:")
    print(pd.DataFrame(out, columns=rows.columns).round(4).to_string(index=False))

pub = df[["Publications"]]
pt  = PowerTransformer(method="yeo-johnson").fit(pub)
yj  = pt.transform(pub)
qt  = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit_transform(pub)

print(f"skew raw          = {df['Publications'].skew():.4f}")
print(f"skew log1p        = {np.log1p(df['Publications']).skew():.4f}")
print(f"skew Yeo-Johnson  = {pd.Series(yj.ravel()).skew():.4f}   (lambda = {pt.lambdas_[0]:.4f})")
print(f"skew Quantile     = {pd.Series(qt.ravel()).skew():.4f}")
print()
print(f"Salary Package skew = {df['Salary Package'].skew():.4f} -> log1p = {np.log1p(df['Salary Package']).skew():.4f}")

Xc = df[["CGPA"]]
Xtr, Xte = train_test_split(Xc, test_size=0.2, random_state=42)

wrong = StandardScaler().fit(Xc)     # fitted on EVERYTHING  <-- leakage
right = StandardScaler().fit(Xtr)    # fitted on train only  <-- correct

print(f"WRONG mean_ = {wrong.mean_[0]:.6f}   scale_ = {wrong.scale_[0]:.6f}")
print(f"RIGHT mean_ = {right.mean_[0]:.6f}   scale_ = {right.scale_[0]:.6f}")
print(f"difference  = {abs(wrong.mean_[0]-right.mean_[0]):.6f}")
print()
print(f"test mean after RIGHT transform = {right.transform(Xte).mean():.6f}   <- NOT exactly 0, correct")
print(f"test std  after RIGHT transform = {right.transform(Xte).std():.6f}")

feat = ["CGPA","AttendancePercent","AptitudeTestScore","CodingTestScore","SoftSkillsRating",
        "Internships","Projects","Certifications","MockInterviewScore"]
D = df[feat + ["PlacementStatus"]].dropna()
Xm, ym = D[feat], D["PlacementStatus"]
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.2, random_state=42, stratify=ym)

sub = np.random.RandomState(0).choice(len(Xtr), 8000, replace=False)   # keeps the SVM tractable
Xtr_s, ytr_s = Xtr.iloc[sub], ytr.iloc[sub]

models = {"KNN (k=5)": KNeighborsClassifier(5),
          "LogisticRegression": LogisticRegression(max_iter=1000),
          "SVM (RBF)": SVC(),
          "DecisionTree": DecisionTreeClassifier(random_state=0)}

res = []
for name, mdl in models.items():
    a_raw = accuracy_score(yte, clone(mdl).fit(Xtr_s, ytr_s).predict(Xte))
    a_std = accuracy_score(yte, Pipeline([("s", StandardScaler()), ("m", clone(mdl))]).fit(Xtr_s, ytr_s).predict(Xte))
    a_mm  = accuracy_score(yte, Pipeline([("s", MinMaxScaler()),   ("m", clone(mdl))]).fit(Xtr_s, ytr_s).predict(Xte))
    res.append([name, round(a_raw,4), round(a_std,4), round(a_mm,4), round(a_std-a_raw,4)])
pd.DataFrame(res, columns=["model","no scaling","StandardScaler","MinMaxScaler","gain"])

# the hidden benefit: comparable coefficients
lr_raw = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr_s)
lr_std = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(Xtr_s, ytr_s)
pd.DataFrame({"feature": feat,
              "coef raw": np.round(lr_raw.coef_[0], 4),
              "coef standardised": np.round(lr_std.named_steps["m"].coef_[0], 4)})
