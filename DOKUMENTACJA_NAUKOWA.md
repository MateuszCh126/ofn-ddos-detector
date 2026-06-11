# Detekcja ataków DDoS z użyciem Skierowanych Liczb Rozmytych (OFN)

**Dokumentacja naukowo-techniczna** · wersja modelu 2.0 (arytmetyka Kosińskiego, score znormalizowany)

---

## Streszczenie

System wykrywa wolumetryczne ataki DDoS na podstawie szeregów czasowych natężenia ruchu
obserwowanego na wielu routerach jednocześnie. Każde krótkie okno pomiarowe routera jest
przekształcane w **Skierowaną Liczbę Rozmytą** (OFN, *Ordered Fuzzy Number*) — obiekt, który
oprócz rozmytej wartości anomalii koduje **kierunek trendu** (wzrost/spadek). Dowody ze
wszystkich routerów są fuzjowane w jedną globalną OFN w algebrze Kosińskiego, defuzzyfikowane
do skalarnego wskaźnika podejrzenia i przepuszczane przez automat alarmowy z histerezą.
Parametry detektora (wagi routerów, progi, histereza) stroi algorytm genetyczny. Jakość
porównywana jest z dwoma klasycznymi detektorami referencyjnymi (próg z-score na wolumenie
globalnym oraz EWMA).

---

## 1. Skierowane Liczby Rozmyte

### 1.1 Definicja

Za Kosińskim, Prokopowiczem i Ślęzakiem, skierowaną liczbą rozmytą nazywamy uporządkowaną parę
funkcji ciągłych

$$
A = (f_A,\, g_A), \qquad f_A, g_A : [0,1] \to \mathbb{R},
$$

gdzie $f_A$ nazywamy **ramieniem wznoszącym** (*up*), a $g_A$ **ramieniem opadającym** (*down*).
Argument $y \in [0,1]$ pełni rolę poziomu przynależności; wartości $f_A(1)$ i $g_A(1)$ wyznaczają
**jądro** (core), a obraz obu funkcji — **nośnik** (support). W odróżnieniu od klasycznych liczb
rozmytych para $(f, g)$ jest *uporządkowana*: zamiana ramion daje inny obiekt, co pozwala
zakodować kierunek procesu (np. ruch rosnący vs malejący).

**Kierunek** OFN definiujemy znakiem przyrostu ramienia wznoszącego:

$$
\operatorname{dir}(A) =
\begin{cases}
+1, & f_A(1) - f_A(0) > \varepsilon, \\
-1, & f_A(1) - f_A(0) < -\varepsilon, \\
\ \ 0, & \text{w przeciwnym razie (singleton)},
\end{cases}
\qquad \varepsilon = 10^{-12}.
$$

### 1.2 Reprezentacja dyskretna

Implementacja (`pyofn/core.py`) przechowuje oba ramiona jako wektory $N$ próbek
($N = 512$ domyślnie, $N = 256$ w detektorze) na równomiernej siatce
$y_k = k/(N-1)$, $k = 0, \dots, N-1$. Wszystkie operacje są wektoryzowane w NumPy;
OFN o różnych $N$ są uzgadniane przez interpolację liniową (`resample`).

### 1.3 Arytmetyka Kosińskiego

Działania wykonywane są **po współrzędnych, ramię do ramienia** — bez krzyżowania ramion,
charakterystycznego dla przedziałowej arytmetyki Zadeha:

$$
A + B = (f_A + f_B,\; g_A + g_B), \qquad
A - B = (f_A - f_B,\; g_A - g_B),
$$

$$
c \cdot A = (c f_A,\; c g_A) \ \ \text{dla } c \in \mathbb{R}, \qquad
-A = (-f_A,\; -g_A), \qquad
A \cdot B = (f_A f_B,\; g_A g_B).
$$

Z tej definicji wynika kluczowa własność algebraiczna, której klasyczne liczby rozmyte nie mają:

$$
\boxed{\,A - A = (0, 0) = \mathbf{0}\,}
$$

— zbiór OFN z dodawaniem tworzy grupę przemienną, a z mnożeniem — strukturę pierścieniową.
Dzięki temu fuzja dowodów „za" i „przeciw" w agregatorze (rozdz. 3.4) nie rozszerza sztucznie
nośnika wyniku, jak działoby się w arytmetyce przedziałowej. Własność $A - A = 0$ oraz
$(A + B) - B = A$ są weryfikowane testami jednostkowymi (`tests/test_pyofn_core.py`).

### 1.4 Funkcja przynależności

Dla OFN „właściwej" (proper), tj. o rozłącznych w $x$ zakresach ramion, przynależność
odtwarzamy interpolacją odwrotną każdego ramienia, z plateau jądra:

$$
\mu_A(x) =
\begin{cases}
1, & x \in [\min(f_A(1), g_A(1)),\ \max(f_A(1), g_A(1))], \\[2pt]
\max\big(f_A^{-1}(x),\, g_A^{-1}(x)\big), & \text{poza jądrem, wewnątrz nośnika}, \\[2pt]
0, & \text{poza nośnikiem}.
\end{cases}
$$

### 1.5 Defuzzyfikacja

Zaimplementowane są dwa funkcjonały wyostrzające:

**(a) Środek ciężkości (COG)** — dla OFN właściwych, całkowanie po osi $x$ z wagą przynależności:

$$
\operatorname{COG}(A) = \frac{\int x\, \mu_A(x)\, dx}{\int \mu_A(x)\, dx}.
$$

Dla trójkąta $(a, b, c)$ daje to dokładnie $\tfrac{a+b+c}{3}$ (test jednostkowy).
Gdy zakresy $x$ ramion się nakładają (OFN „niewłaściwa", *improper* — naturalny produkt
odejmowania w algebrze Kosińskiego), $\mu_A$ przestaje być funkcją jednoznaczną i COG po $x$
traci sens; implementacja wykrywa nakładanie i przechodzi na wariant (b).

**(b) Średnia ramion** — całkowanie po osi $y$:

$$
\operatorname{MOA}(A) = \frac{1}{2}\left(\int_0^1 f_A(y)\, dy + \int_0^1 g_A(y)\, dy\right).
$$

Funkcjonał ten jest **liniowy** względem arytmetyki Kosińskiego:
$\operatorname{MOA}(\alpha A + \beta B) = \alpha\operatorname{MOA}(A) + \beta\operatorname{MOA}(B)$,
zawsze dobrze określony i ciągły względem deformacji kształtu — dlatego to on (a nie COG)
jest używany w agregatorze do wyliczania score (rozdz. 3.4).

### 1.6 Kształty elementarne

`pyofn/shapes.py` dostarcza konstruktory: trójkątne i trapezowe (w obu skierowaniach),
gaussowskie (ramiona z odwrotnej dystrybuanty $x = m \mp \sigma\sqrt{-2\ln y}$, obcięte do
$\pm 3\sigma$), singleton oraz ogólną OFN liniową. Trapez skierowany w prawo na parametrach
$a \le b \le c \le d$:

$$
f(y) = a + y(b - a), \qquad g(y) = c + (1-y)(d - c),
$$

wersja skierowana w lewo zamienia role ramion: $f(y) = d + y(c-d)$, $g(y) = a + y(b-a)$.

---

## 2. Odporna normalizacja ruchu

Surowe natężenie ruchu routera $r$ w chwili $t$ oznaczmy $x_r(t)$. Baseline estymowany jest
odpornie z okna historii $H$ (domyślnie 16 kroków poprzedzających bieżące okno), przez medianę
i medianowe odchylenie bezwzględne (MAD):

$$
m_r = \operatorname{med}(H), \qquad
s_r = \max\big(1.4826 \cdot \operatorname{med}\lvert H - m_r \rvert,\ s_{\min}\big),
$$

gdzie stała $1.4826 = 1/\Phi^{-1}(3/4)$ czyni MAD zgodnym estymatorem $\sigma$ dla rozkładu
normalnego, a $s_{\min} = 1$ zabezpiecza przed degeneracją skali. Znormalizowany sygnał
(z-score) jest obcinany do $\pm 8$:

$$
z_r(t) = \operatorname{clip}\!\left(\frac{x_r(t) - m_r}{s_r},\, -8,\, 8\right).
$$

Estymatory medianowe mają punkt załamania 50%, więc baseline nie „uczy się" trwającego ataku
tak szybko, jak średnia i wariancja próbkowa. Przy wielu cechach na router (tensor
`steps × routers × features`) każda cecha jest normalizowana niezależnie, a następnie składana
w kompozyt średnią ważoną wagami cech $v_j \ge 0$:

$$
z^{(c)}_r(t) = \frac{\sum_j v_j\, z_{r,j}(t)}{\sum_j v_j},
\qquad
u^{(c)}_r(t) = \frac{\sum_j v_j\, \max(z_{r,j}(t), 0)}{\sum_j v_j},
$$

gdzie $u^{(c)}$ to kompozyt **anomalii dodatniej** (część dodatnia z-score) — z niego budowany
jest kształt OFN, podczas gdy pełny (znakowy) kompozyt $z^{(c)}$ służy do estymacji kierunku.

---

## 3. Tor detekcji OFN

### 3.1 Okno i budowa trapezu

W każdej chwili $t \ge W - 1$ (rozmiar okna $W = 4$) router dostarcza okno anomalii
$u = (u_1, \dots, u_W)$. Po posortowaniu $u_{(1)} \le \dots \le u_{(4)}$ wartości te stają się
parametrami trapezu $(a, b, c, d)$. Jeżeli rozpiętość $d - a$ jest mniejsza niż `min_spread`
$= 0{,}2$, trapez jest rozszerzany symetrycznie wokół średniej okna do minimalnej szerokości
(z obcięciem $a \ge 0$, spójnym z nieujemnością $u$), a degeneracja ramienia kierunkowego jest
korygowana o margines $\delta = \max(\text{min\_spread}/6, 10^{-6})$, by kierunek OFN nie
kolapsował numerycznie do 0.

Router spoczynkowy (kierunek 0 i średnia anomalia $\le$ `min_spread`) reprezentowany jest
singletonem — nie wnosi kształtu, tylko punktową wartość.

### 3.2 Estymacja kierunku (regresja OLS)

Trend okna wyznacza nachylenie prostej najmniejszych kwadratów dopasowanej do pełnego
znormalizowanego okna $z^{(c)} = (z_1, \dots, z_W)$:

$$
\hat\beta = \frac{\sum_{i=1}^{W} (i - \bar i)(z_i - \bar z)}{\sum_{i=1}^{W} (i - \bar i)^2},
\qquad
T = \hat\beta \cdot (W - 1),
$$

czyli $T$ to łączna zmiana wzdłuż dopasowanej prostej na długości okna. Kierunek:

$$
d_r = \operatorname{sign}(T) \cdot \mathbb{1}\big[\lvert T \rvert > \varepsilon_T\big],
\qquad \varepsilon_T = 2{,}2.
$$

**Kalibracja statystyczna progu.** Dla czystego szumu z-score ($\sigma \approx 1$) i $W = 4$
odchylenie standardowe statystyki $T$ wynosi
$\sigma_T = (W-1)\,\sigma / \sqrt{\sum (i - \bar i)^2} = 3/\sqrt{5} \approx 1{,}34$.
Próg $\varepsilon_T = 2{,}2 \approx 1{,}64\,\sigma_T$ odpowiada więc dwustronnemu poziomowi
istotności $\approx 10\%$ — empirycznie mediana odsetka routerów fałszywie „kierunkowych"
w spoczynku spada z ~92% (poprzednia heurystyka różnicy brzegów z $\varepsilon_T = 0{,}15$)
do ~6,6%.

### 3.3 OFN routera

$$
A_r =
\begin{cases}
\operatorname{trapez}^{\rightarrow}(u_{(1)}, u_{(2)}, u_{(3)}, u_{(4)}), & d_r \ge 0 \ \text{(i nie-spoczynkowy)}, \\
\operatorname{trapez}^{\leftarrow}(u_{(1)}, u_{(2)}, u_{(3)}, u_{(4)}), & d_r < 0, \\
\operatorname{singleton}(\bar u), & \text{spoczynek}.
\end{cases}
$$

### 3.4 Agregacja globalna i score

Dowody routerów łączone są w algebrze Kosińskiego z wagami $w_r \ge 0$ (strojonymi przez GA):

$$
G = \frac{1}{\Omega} \left(
\sum_{r:\, d_r > 0} w_r A_r
\;-\; \sum_{r:\, d_r < 0} w_r A_r
\;+\; \kappa \sum_{r:\, d_r = 0} w_r A_r
\right),
\qquad
\Omega = \sum_{r} w_r^{\text{eff}},
$$

gdzie $\kappa = 0{,}25$ (`neutral_contribution`) tłumi wkład routerów bez wyraźnego kierunku,
a wagi efektywne $w_r^{\text{eff}} = w_r$ dla routerów kierunkowych i $\kappa\, w_r$ dla
neutralnych. Normalizacja przez $\Omega$ sprawia, że score **nie zależy od liczby routerów** —
jest średnią ważoną, nie sumą (przed tą poprawką spoczynkowy score rósł liniowo z $N$:
0,08 → 0,37 → 1,43 dla $N = 10/30/60$, generując FPR do 17%).

Wskaźnik podejrzenia to defuzzyfikacja średnią ramion (liniowość, rozdz. 1.5b):

$$
S(t) = \max\big(\operatorname{MOA}(G),\, 0\big)
= \max\!\left( \frac{1}{\Omega}\sum_r \pm\, w_r^{\text{eff}}\operatorname{MOA}(A_r),\ 0 \right).
$$

Interpretacja: $S(t)$ jest w skali „przeciętnego z-score anomalii na router", typowo
$S \in [0, 8]$ (górne ograniczenie z `anomaly_clip`).

### 3.5 Automat alarmowy z histerezą

Stan alarmu $\alpha(t) \in \{0, 1\}$ aktualizuje dwuprogowy automat ze zliczaniem serii:

- warunek alertu: $S(t) \ge \theta_a$ **i** liczba routerów dodatnich
  $n^+(t) \ge n_{\min}$ **i** $S(t) \ge S_{\min}$;
- warunek wyciszenia: $S(t) \le \theta_c$;
- strefa pośrednia $(\theta_c, \theta_a)$ zeruje obie serie (świadoma decyzja projektowa:
  oscylacje wokół progu nie wzbudzają ani nie gaszą alarmu).

Alarm włącza się po $k_a$ kolejnych spełnieniach warunku alertu, gaśnie po $k_c$ kolejnych
spełnieniach warunku wyciszenia. Wartości domyślne (skala znormalizowana, v2.0):
$\theta_a = 1{,}5$, $\theta_c = 0{,}75$, $k_a = k_c = 2$, $n_{\min} = 4$.

---

## 4. Metryki jakości

Dla etykiet $y_t \in \{0,1\}$ i predykcji $\hat y_t$ liczone są punktowo po krokach czasu:
TP, FP, TN, FN oraz

$$
\text{recall} = \frac{TP}{TP + FN}, \quad
\text{precision} = \frac{TP}{TP + FP}, \quad
F_1 = \frac{2PR}{P + R}, \quad
\text{FPR} = \frac{FP}{FP + TN},
$$

z konwencją NaN przy braku pozytywnych etykiet/predykcji. **Opóźnienie detekcji** to odstęp
od pierwszego kroku ataku $t_0$ do pierwszej predykcji pozytywnej $\ge t_0$; brak detekcji
karany jest pełnym pozostałym horyzontem $|T| - t_0$.

---

## 5. Strojenie genetyczne

Genom kandydata o długości $R + 5$ ($R$ — liczba routerów):

$$
\gamma = \big(w_1, \dots, w_R,\ \theta_a,\ \rho,\ \phi,\ k_a,\ k_c\big),
$$

dekodowany z obcięciem do kostki ograniczeń: $w_r \in [0{,}1,\, 3]$,
$\theta_a \in [0{,}5,\, 5]$, próg wyciszenia $\theta_c = \rho\,\theta_a$ z
$\rho \in [0{,}25,\, 0{,}9]$, $n_{\min} = \max(1, \lfloor \phi R \rceil)$ z
$\phi \in [0{,}05,\, 0{,}8]$, $k_a, k_c \in \{1, \dots, 5\}$.

**Funkcja kosztu** (minimalizowana, uśredniana po scenariuszach treningowych):

$$
J(\gamma) = 0{,}55\,(1 - \text{recall}) + 0{,}30\,\text{FPR}
+ 0{,}15\,\frac{\text{delay}}{|T| - t_0}.
$$

Scenariusze bez ataku wnoszą wyłącznie składnik FPR. Operatory: selekcja turniejowa
($k = 3$), krzyżowanie arytmetyczne $\gamma' = \alpha\gamma_1 + (1{-}\alpha)\gamma_2$
($p_c = 0{,}75$), mutacja gaussowska $\mathcal{N}(0, 0{,}18)$ z prawdopodobieństwem
$0{,}12$ na gen (geny całkowitoliczbowe zaokrąglane), elityzm 4 osobników,
populacja 36, 24 pokolenia, RNG z ziarnem dla powtarzalności.

---

## 6. Detektory referencyjne

**Volume threshold** — odporny z-score na globalnym wolumenie $V(t) = \sum_r x_r(t)$
względem mediany/MAD z przesuwnej historii 16 kroków, z tą samą histerezą.

**EWMA** — wykładniczo ważona średnia i wariancja w sformułowaniu Welforda:

$$
\delta_t = V(t) - \hat\mu_{t-1}, \qquad
\hat\mu_t = \hat\mu_{t-1} + \alpha\,\delta_t, \qquad
\hat\sigma^2_t = (1 - \alpha)\big(\hat\sigma^2_{t-1} + \alpha\,\delta_t^2\big),
$$

score $= \max(\delta_t / \max(\hat\sigma_{t-1}, \sigma_{\min}), 0)$ — residuum oceniane jest
względem stanu *sprzed* aktualizacji. Sformułowanie Welforda zastąpiło wcześniejszy wariant
$\hat\sigma^2_t = \alpha\delta_t^2 + (1-\alpha)\hat\sigma^2_{t-1}$, który systematycznie
zawyżał wariancję (brak czynnika $1-\alpha$ przy innowacji), zaniżając czułość baseline'u.

---

## 7. Scenariusze syntetyczne

Generator (`ddos_ofn/simulation.py`) tworzy macierz `steps × routers` z baseline'ami
$\mathcal{U}(80, 160)$ i szumem $\mathcal{N}(0, 4)$, obciętą do wartości nieujemnych:

| Scenariusz | Charakter | Etykiety ataku |
|---|---|---|
| `normal` | czysty szum | brak |
| `ddos_ramp` | liniowa rampa na 70% routerów | tak |
| `ddos_pulse` | pulsowanie 2-z-3 kroków | tak |
| `ddos_low_and_slow` | słaba, wolna rampa (0,55 amplitudy) | tak |
| `ddos_rotating` | 4 grupy routerów atakowane rotacyjnie | tak |
| `flash_crowd` | sinusoidalny przepływ legalny (25% routerów) | **nie** (test FPR) |
| `flash_cascade` | kaskada przepływów legalnych grupami | **nie** (test FPR) |

Scenariusze `flash_*` pełnią rolę negatywnych przypadków kontrolnych: poprawny detektor ma
ich **nie** klasyfikować jako ataku, mimo wzrostu wolumenu.

---

## 8. Własności, kompromisy i ograniczenia

1. **FPR vs recall.** Kalibracja v2.0 (próg kierunku $\varepsilon_T = 2{,}2$, normalizacja
   score) sprowadza FPR w spoczynku do 0% i odporność na `flash_crowd`, ale na domyślnych
   progach osłabia czułość na ataki subtelne — w szczególności `ddos_low_and_slow` wymaga
   strojenia GA lub niższych progów per-instalacja. Słaba rampa o przyroście wolniejszym niż
   $\varepsilon_T$ na długości okna $W$ z definicji nie wzbudza kierunku; remedium to dłuższe
   okno trendu, niezależne od okna kształtu.
2. **Skala punktowa.** Po normalizacji $\Omega$ score ma stałą interpretację niezależną od
   liczby routerów; modele zapisane w skali v1.0 są niekompatybilne i odrzucane przy wczytaniu
   (pole `version` w payloadzie JSON).
3. **Stacjonarność baseline'u.** Mediana/MAD z 16 kroków zakłada lokalną stacjonarność ruchu;
   silna sezonowość dobowa wymagałaby baseline'u warunkowanego czasem.
4. **Detekcja punktowa po krokach.** Metryki liczone są per-krok, nie per-epizod; długi atak
   wykryty z opóźnieniem obniża recall proporcjonalnie do opóźnienia.

---

## 9. Literatura

1. W. Kosiński, P. Prokopowicz, D. Ślęzak, *Ordered fuzzy numbers*, Bulletin of the Polish
   Academy of Sciences: Mathematics, 51(3), 2003, 327–338.
2. W. Kosiński, *On fuzzy number calculus*, International Journal of Applied Mathematics and
   Computer Science, 16(1), 2006, 51–57.
3. P. Prokopowicz i in. (red.), *Theory and Applications of Ordered Fuzzy Numbers*,
   Studies in Fuzziness and Soft Computing 356, Springer, 2017.
4. P.J. Rousseeuw, C. Croux, *Alternatives to the median absolute deviation*, JASA 88(424), 1993.
5. B.P. Welford, *Note on a method for calculating corrected sums of squares and products*,
   Technometrics 4(3), 1962.
