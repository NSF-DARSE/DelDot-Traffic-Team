from _style import *
z,m=load()
fig,ax=plt.subplots(figsize=(7,4))
labels=["GEH < 5\n(good)","GEH < 10\n(acceptable)"]; vals=[m["pct_GEH_under_5"],m["pct_GEH_under_10"]]
b=ax.bar(labels,vals,color=[SIGN,AMBER],width=.55)
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+1,f"{v:.0f}%",ha="center",fontweight="bold")
ax.set_ylim(0,100); ax.set_title(f"GEH acceptance (seen series) (mean GEH = {m['GEH_mean']:.1f})"); ax.set_ylabel("% of predictions")
save(fig,"fig5_geh_breakdown.png")
