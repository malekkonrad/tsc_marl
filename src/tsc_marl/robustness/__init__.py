"""Faza 3 — zachowanie sterowników pod awarią sensorów.

`controllers` — kontrolery, które czytają TĘ SAMĄ warstwę detektorów (polityka,
MaxPressure, FixedTime); `degrade` — pętla ewaluacji zamrożonego sterownika na
liście scenariuszy awarii. Wejście procesu: `runner/degrade.py` (`tsc degrade`).
"""
