from _style import *
z,m=load(); a=z["one_actual"]; p=z["one_pred"]; x=range(len(a))
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(x,a,color=INK,lw=2,label="Actual")
ax.plot(x,p,color=SIGN,lw=2,ls="--",label="Day-1 forecast")
ax.set_title("One busy series, one week: forecast vs actual")
ax.set_xlabel("hour into the week"); ax.set_ylabel("volume (veh/hr)"); ax.legend(frameon=False)
save(fig,"fig3_pred_vs_actual.png")
