# Elevator Pitch — Traffic Forecasting System

## The Pitch (30 seconds)

"We built a system that predicts how many cars will be on any Delaware road,
at any hour, for the next 30 days — including roads that just got sensors
installed last week.

It refreshes every morning, it's live on AWS for about $25 a month, and it
tells you not just *what* it predicts but *how confident* it is.

The practical use? Before you close a lane for repairs, you can look at the
dashboard and pick the time that disrupts the fewest drivers. That could be
the difference between a minor inconvenience and a 2-hour traffic jam."

---

## If they ask "how does it work?"

"Traffic is surprisingly predictable. Every road has a rhythm — rush hour,
weekday vs weekend, summer vs winter. We learned those patterns from 18 months
of data and layer them together like a recipe. Then a machine learning model
fine-tunes the details. For brand-new sensors, we use neighboring roads to
estimate what the pattern should look like."

---

## If they ask "how accurate is it?"

"93% of our predictions are within acceptable accuracy by industry standards.
The average error is about 105 cars per hour on roads that typically see 800.
And every prediction comes with a confidence range — wider when we're less
sure, tighter when the pattern is rock-solid."

---

## If they ask "what's the big deal?"

"Right now, planning road work or estimating traffic impact is mostly guesswork
and experience. This gives DOT staff an actual number they can point to — with
a confidence level — updated every morning. It turns 'I think Sundays are
quieter' into 'Sunday at 6am on Route STN_0053 will see 340 cars per hour,
65% below the Tuesday peak.' That's actionable."
