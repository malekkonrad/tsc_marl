"""Dziedzina sterowania sygnalizacją — wzory z papierów, bez frameworków.

Nic tutaj nie wie o LibSignalu, torchu ani o naszym rdzeniu: to prymitywy
liczone na słownikach i tablicach, których używa ZARÓWNO rdzeń Fazy 1
(`core/features.py`), JAK I agenci-wtyczki (`libsignal/agents/`). Dlatego
dziedzina leży niżej niż jedni i drudzy — inaczej rdzeń importowałby z katalogu
agentów cudzego frameworka, co przez chwilę faktycznie robił.
"""
