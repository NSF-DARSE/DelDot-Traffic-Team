from _style import *
import numpy as np
z,m=load()
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.5,4))
# MAPE seen vs cold
labels=["Seen\n(LightGBM)","Cold-start\n(GNN)"]; mp=[m["MAPE"],m["cold_sim_MAPE"]]
b=a1.bar(labels,mp,color=[SIGN,BLUE],width=.6)
for r,v in zip(b,mp): a1.text(r.get_x()+r.get_width()/2,v+2,f"{v:.0f}%",ha="center",fontweight="bold")
a1.set_title("MAPE (actuals ≥ 20 veh/hr)"); a1.set_ylabel("MAPE %")
# GEH<5 seen vs cold with the 85% microsim acceptance reference
gv=[m["pct_GEH_under_5"],m["cold_pct_GEH_under_5"]]
b2=a2.bar(labels,gv,color=[SIGN,BLUE],width=.6)
for r,v in zip(b2,gv): a2.text(r.get_x()+r.get_width()/2,v+1.5,f"{v:.0f}%",ha="center",fontweight="bold")
a2.axhline(85,color=RED,ls="--",lw=1.5,label="85% microsim-calibration standard")
a2.set_ylim(0,100); a2.set_title("GEH < 5 vs the real-world reference"); a2.set_ylabel("% of predictions"); a2.legend(frameon=False,fontsize=9)
save(fig,"fig7_realworld_standards.png")
