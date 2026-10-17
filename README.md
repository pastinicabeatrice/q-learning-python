# Q-learning

Un agent care învață singur, prin încercare și eroare, să iasă dintr-un labirint evitând capcanele.
Este învățare prin recompensă (reinforcement learning), implementată de la zero.

## Mediul

Labirintul este o listă de șiruri: `S` start, `G` țintă (+10), `X` capcană (-10), `#` zid, `.` liber.
Fiecare pas costă -1, deci agentul este încurajat să găsească drumul cel mai scurt.

## Algoritmul

Agentul păstrează o tabelă `Q[stare, acțiune]` cu recompensa estimată și o actualizează după fiecare pas:

```
Q[s, a] ← Q[s, a] + α · (r + γ · max Q[s', ·] - Q[s, a])
```

- `α = 0.5` - rata de învățare
- `γ = 0.9` - cât de mult contează recompensele viitoare
- `ε = 0.2` - probabilitatea de a încerca o acțiune aleatoare (explorare) în loc de cea mai bună cunoscută (exploatare)

## Ce afișează `main.py`

- cum crește recompensa medie pe parcursul antrenării
- politica învățată, ca hartă de săgeți
- drumul final al agentului

## Rulare

Necesită Python 3.8 sau mai nou, fără alte dependențe.

```
python main.py
python -m unittest
```
