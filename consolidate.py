import pandas as pd, numpy as np
from run_extra import block_bootstrap_diff
R="results/"
cv=pd.read_csv(R+"cv_summary.csv"); cvp=pd.read_csv(R+"cv_peaks_summary.csv",index_col=0); g=pd.read_csv(R+"gru_cv_folds.csv")
te=pd.read_csv(R+"final_test_results.csv"); tp=pd.read_csv(R+"test_peaks_results.csv",index_col=0); tg=pd.read_csv(R+"gru_test_results.csv")
def cvrow(f,m):
    r=cv[(cv.features==f)&(cv.model==m)].iloc[0]
    return r["MAE_mean"],r["MAE_std"],r["R2_mean"],r["within_0.10_mean"],r["R2_normal_mean"]
def terow(f,m):
    r=te[(te.features==f)&(te.model==m)].iloc[0]
    return r["MAE"],r["R2"],r["within_0.10"],r["R2_normal"]
rows=[]
def add(name,c,t):
    rows.append(dict(model=name,CV_MAE=c[0],CV_MAE_std=c[1],CV_R2=c[2],CV_within_0_10=c[3],CV_R2_normal=c[4],
                     Test_MAE=t[0],Test_R2=t[1],Test_within_0_10=t[2],Test_R2_normal=t[3]))
add("Mean (no skill)",cvrow("F0","mean"),terow("F0","mean"))
add("Sensors only - XGBoost",cvrow("F1","xgb_l2"),terow("F1","xgb_l2"))
add("Persistence (previous Si)",cvrow("F0+H","persistence"),terow("F0+H","persistence"))
add("Sensors + prev Si - XGBoost",cvrow("F0+H","xgb_l2"),terow("F0+H","xgb_l2"))
add("Sensors + prev Si - Ridge",cvrow("F2+H","ridge"),terow("F2+H","ridge"))
h=cvp.loc["hybrid"]
add("Sensors + prev Si - Ridge+XGB hybrid",(h["MAE_mean"],h["MAE_std"],h["R2_mean"],h["within_0.10_mean"],h["R2_normal_mean"]),
    (tp.loc["hybrid","MAE"],tp.loc["hybrid","R2"],tp.loc["hybrid","within_0.10"],tp.loc["hybrid","R2_normal"]))
add("Sensors + prev Si - GRU (3 seeds)",(g["MAE"].mean(),g["MAE"].std(),g["R2"].mean(),g["within_0.10"].mean(),g["R2_normal"].mean()),
    (tg["MAE"][0],tg["R2"][0],tg["within_0.10"][0],tg["R2_normal"][0]))
T=pd.DataFrame(rows).round(4); T.to_csv(R+"consolidated_results.csv",index=False); print(T.to_string(index=False))
import os
if os.path.exists(R+"gru_test_runs.csv"):
    gr=pd.read_csv(R+"gru_test_runs.csv"); print(f"\nGRU test over {len(gr)} independent runs: MAE {gr.MAE.mean():.4f} +- {gr.MAE.std():.4f}, R2 {gr.R2.mean():.3f} +- {gr.R2.std():.3f}")
    print("(GRU table row = mean over runs; the bootstrap below uses predictions averaged over runs)")
pf=pd.read_csv(R+"cv_peaks_folds.csv").pivot(index="fold",columns="variant",values="MAE")
print("\nper-fold MAE hybrid",pf["hybrid"].round(4).tolist()," gru",g.set_index("fold")["MAE"].round(4).tolist())
a=pd.read_csv(R+"test_peaks_predictions.csv",parse_dates=["dt"]); b=pd.read_csv(R+"gru_test_predictions.csv",parse_dates=["dt"])
assert (a.dt.values==b.dt.values).all(); y=a.Si.values
for n2,p2 in [("hybrid",a.hybrid),("persistence",a.persistence),("xgb_ref",a.xgb_ref)]:
    d,lo,hi=block_bootstrap_diff(np.abs(y-b.gru),np.abs(y-p2)); print(f"gru vs {n2}: dMAE={d:.4f} CI=({lo:.4f},{hi:.4f})","SIG" if not lo<=0<=hi else "n.s.")
m=y>1.2; print("GRU MAE on upsets (>1.2):", round(float(np.abs(y[m]-b.gru[m]).mean()),3))


# ---------------- figure: CV and test MAE of all models (overwrites the older 5-bar version) ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
labels = ["Mean", "Sensors only\n(XGBoost)", "Persistence", "Sensors+prev Si\n(XGBoost)", "Sensors+prev Si\n(Ridge)",
          "Sensors+prev Si\n(Ridge+XGB)", "Sensors+prev Si\n(GRU)"]
colors = ["gray", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3))
ax[0].bar(range(len(T)), T.CV_MAE, yerr=T.CV_MAE_std, capsize=3, color=colors)
ax[0].set_title("(a) Time-series CV (mean ± std, 5 folds)")
ax[1].bar(range(len(T)), T.Test_MAE, color=colors)
ax[1].set_title("(b) Held-out test period")
for a in ax:
    a.set_xticks(range(len(T)))
    a.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
    a.set_ylabel("MAE (wt% Si)")
fig.tight_layout()
fig.savefig(R + "fig_cv_mae.png", dpi=300)
print("saved fig_cv_mae.png")