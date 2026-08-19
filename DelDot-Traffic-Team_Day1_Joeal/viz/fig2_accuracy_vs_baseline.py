from _style import *
z,m=load()
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.5,4))
labels=["Seasonal\nnaive","Day-1\nmodel"]; vals=[m["naive_MAE"],m["MAE"]]
b=a1.bar(labels,vals,color=[GREY,SIGN],width=.6)
for r,v in zip(b,vals): a1.text(r.get_x()+r.get_width()/2,v+4,f"{v:.0f}",ha="center",fontweight="bold")
a1.set_title("Point error (MAE, veh/hr)"); a1.set_ylabel("MAE")
w=[m["wape_naive"],m["wape_model"],m["wape_oracle"]]; wl=["Naive","Model","Oracle\nceiling"]
b2=a2.bar(wl,w,color=[GREY,SIGN,AMBER],width=.6)
for r,v in zip(b2,w): a2.text(r.get_x()+r.get_width()/2,v+.005,f"{v:.3f}",ha="center",fontweight="bold")
a2.set_title("WAPE vs the naive->oracle range"); a2.set_ylabel("WAPE")
save(fig,"fig2_accuracy_vs_baseline.png")
