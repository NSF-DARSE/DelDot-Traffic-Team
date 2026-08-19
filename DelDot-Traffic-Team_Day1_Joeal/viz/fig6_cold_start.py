from _style import *
z,m=load()
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.5,4))
# left: cold-start WAPE, naive vs GNN blend (holdout simulation)
labels=["Naive\n(global mean)","GNN + prior\n(blended)"]; vals=[m["cold_naive_WAPE"],m["cold_sim_WAPE"]]
b=a1.bar(labels,vals,color=[GREY,SIGN],width=.6)
for r,v in zip(b,vals): a1.text(r.get_x()+r.get_width()/2,v+.01,f"{v:.3f}",ha="center",fontweight="bold")
imp=(1-m["cold_sim_WAPE"]/m["cold_naive_WAPE"])*100
a1.set_title(f"Cold-start WAPE ({m['cold_sim_stations']} held-out stations)\nGNN beats naive by {imp:.0f}%"); a1.set_ylabel("WAPE")
# right: cold GEH
gl=["GEH < 5","GEH < 10"]; gv=[m["cold_pct_GEH_under_5"],m["cold_pct_GEH_under_10"]]
b2=a2.bar(gl,gv,color=[SIGN,AMBER],width=.55)
for r,v in zip(b2,gv): a2.text(r.get_x()+r.get_width()/2,v+1,f"{v:.0f}%",ha="center",fontweight="bold")
a2.set_ylim(0,100); a2.set_title(f"Cold-start GEH (mean {m['cold_GEH_mean']:.1f})"); a2.set_ylabel("% of predictions")
save(fig,"fig6_cold_start.png")
