from _style import *
z,m=load(); wd,we=z["prof_wd"],z["prof_we"]; h=range(24)
fig,ax=plt.subplots(figsize=(8,4))
ax.fill_between(h,wd,color=SIGN,alpha=.10); ax.plot(h,wd,color=SIGN,lw=2.5,marker="o",ms=4,label="Weekday")
ax.fill_between(h,we,color=AMBER,alpha=.10); ax.plot(h,we,color=AMBER,lw=2.5,marker="o",ms=4,label="Weekend")
ax.set_title("The daily rhythm of traffic (validation actuals)")
ax.set_xlabel("hour of day"); ax.set_ylabel("mean volume (veh/hr)"); ax.set_xticks(range(0,24,3)); ax.legend(frameon=False)
save(fig,"fig1_daily_profile.png")
