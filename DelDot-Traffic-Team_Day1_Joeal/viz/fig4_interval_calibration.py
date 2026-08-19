from _style import *
z,m=load()
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4))
cov=m["coverage_pct"]; tgt=m["target_pct"]
a1.bar(["Achieved","Target"],[cov,tgt],color=[SIGN,GREY],width=.55)
a1.text(0,cov+.6,f"{cov:.1f}%",ha="center",fontweight="bold"); a1.text(1,tgt+.6,f"{tgt:.0f}%",ha="center",fontweight="bold")
a1.set_ylim(80,95); a1.set_title("90% interval coverage"); a1.set_ylabel("% of actuals inside band")
sw=z["seen_widths"]; a2.hist(sw,bins=40,color=BLUE,alpha=.8)
a2.axvline(np.mean(sw),color=RED,lw=2,ls="--",label=f"mean {np.mean(sw):.0f}")
a2.set_title("Interval width (seen series)"); a2.set_xlabel("upper - lower (veh/hr)"); a2.set_ylabel("count"); a2.legend(frameon=False)
save(fig,"fig4_interval_calibration.png")
