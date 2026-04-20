# Dokumentacja OFN DDoS Detector

## 1. Cel projektu

Projekt wykrywa anomalie typu DDoS na podstawie ruchu obserwowanego na wielu routerach lub wezlach sieci.

Rdzen pomyslu jest taki:

1. Z kazdego routera bierzemy krotkie okno pomiarowe o dlugosci `window_size` probek. Domyslnie sa to 4 kolejne pomiary, ale nie jest to stala zaszyta w logice.
2. To okno zamieniamy na skierowana liczbe rozmyta OFN.
3. Wszystkie OFN laczymy do jednego globalnego sygnalu.
4. Detektor podejmuje decyzje alarm / brak alarmu na podstawie progow i histerezy.
5. Algorytm genetyczny dostraja wagi routerow oraz parametry alarmu.

To nie jest klasyfikator oparty o siec neuronowa. To jest model analityczny i interpretowalny.

## 2. Jak system dziala krok po kroku

### 2.1. Wejscie

Wejsciem jest macierz ruchu o ksztalcie:

```text
(steps, routers)
```

czyli:

1. `steps` - kolejne kroki czasu,
2. `routers` - kolejne routery lub wezly.

Kazda kolumna to historia ruchu jednego routera.

### 2.2. Wyznaczenie baseline

Plik: [baseline.py](./ddos_ofn/baseline.py)

Dla kazdego routera system bierze historie z poprzednich krokow i liczy:

1. odporny srodek (`median`),
2. odporna skale (`MAD` przeskalowany do odpowiednika odchylenia standardowego),
3. dolna granice skali `min_baseline_scale`, zeby uniknac dzielenia przez zbyt male wartosci.

To jest potrzebne, bo surowa liczba pakietow nie jest porownywalna miedzy routerami. Duzy router i maly router moga miec inna skale ruchu, ale podobny poziom anomalii po normalizacji.

### 2.3. Budowa OFN dla routera

Plik: [ofn_builder.py](./ddos_ofn/ofn_builder.py)

Z krotkiego okna pomiarowego powstaje jedna skierowana liczba rozmyta:

1. okno jest normalizowane do baseline,
2. z-score sa symetrycznie obcinane do przedzialu `[-anomaly_clip, anomaly_clip]`, zeby pojedyncze ekstremalne probki nie dominowaly,
3. znormalizowane wartosci ujemne sa potem odcinane do zera przez `anomaly = max(normalized, 0)`, bo ksztalt OFN opisuje dodatnia czesc anomalii,
4. trend jest oceniany z roznicy pomiedzy ostatnia i pierwsza probka znormalizowanego okna,
5. `suspicion` to srednia z dodatniej czesci anomalii,
6. z tego budowany jest OFN trapezowy albo singleton.

Interpretacja:

1. kierunek dodatni oznacza, ze anomalia rosnie,
2. kierunek ujemny oznacza, ze ruch w oknie maleje,
3. kierunek neutralny oznacza slaby lub niejednoznaczny trend,
4. wartosc `suspicion` to defuzyfikowany poziom podejrzenia dla jednego routera.

Wazne: w systemie wystepuja dwa rozne "clipowania" i trzeba je rozrozniac:

1. `anomaly_clip` - symetryczne obciecie z-score do `[-clip, clip]`,
2. odciecie do czesci dodatniej - zamiana wartosci ujemnych na `0` przed budowa OFN.

### 2.4. Agregacja wielu routerow

Plik: [aggregator.py](./ddos_ofn/aggregator.py)

Kazdy router daje swoj OFN. Potem system laczy je do jednego globalnego OFN:

1. dodatnie OFN sa dodawane,
2. ujemne OFN sa odejmowane,
3. neutralne OFN sa nadal dodawane, ale z oslabiona sila `neutral_contribution`,
4. kazdy router moze miec swoja wage `w_i`.

W efekcie powstaje:

1. `raw_score` - surowy globalny wynik przed dolnym ograniczeniem,
2. `score` - wynik po obcieciu do `>= 0`,
3. liczba routerow dodatnich, ujemnych i neutralnych.

Wazne: neutralne OFN nie sa ignorowane i nie odejmuja score'u. Ich wklad jest dodatni, ale oslabiony wspolczynnikiem `neutral_contribution`.

### 2.5. Detektor alarmu

Plik: [detector.py](./ddos_ofn/detector.py)

Detektor nie odpala alarmu od razu po jednym skoku. Uzywa histerezy.

Warunek wlaczenia alarmu jest logicznym `AND` trzech warunkow:

1. `score >= alert_threshold`,
2. `positive_routers >= min_positive_routers`,
3. `score >= min_total_score`.

Dopiero jesli caly ten warunek utrzyma sie przez `alert_windows` kolejnych okien, alarm zostaje wlaczony.

Kasowanie alarmu ma osobna logike:

1. alarm moze zostac wygaszony dopiero, gdy `score <= clear_threshold`,
2. ten warunek musi utrzymac sie przez `clear_windows` kolejnych okien.

To ogranicza miganie alarmu i zmniejsza falszywe alarmy.

### 2.6. Metryki oceny

Plik: [metrics.py](./ddos_ofn/metrics.py)

Projekt liczy:

1. `recall`,
2. `precision`,
3. `f1`,
4. `false_positive_rate`,
5. `detection_delay`.

Praktyczna interpretacja:

1. `recall` mowi, ile krokow ataku zostalo wykrytych,
2. `false_positive_rate` mowi, jak czesto model podnosi alarm bez ataku,
3. `detection_delay` mowi, po ilu krokach od startu ataku pojawia sie pierwszy alarm.

Dokladna semantyka `detection_delay` jest taka:

1. jezeli w scenariuszu nie ma ataku, metryka zwraca `0.0`,
2. jezeli atak jest, a alarm pojawi sie po starcie ataku, zwracana jest liczba krokow od pierwszego kroku ataku do pierwszego alarmu,
3. jezeli atak jest, ale alarm nigdy nie pojawi sie po jego starcie, zwracana jest pozostala dlugosc scenariusza `steps - first_attack`.

Ta ostatnia regola daje skonczona kare, dzieki czemu kandydaci GA sa porownywalni bez wprowadzania `inf` albo `None`.

### 2.7. Strojenie GA

Plik: [ga_optimize.py](./ddos_ofn/ga_optimize.py)

Algorytm genetyczny nie optymalizuje OFN bezposrednio. On stroi:

1. wagi routerow,
2. `alert_threshold`,
3. relacje `clear_threshold` do `alert_threshold`,
4. minimalny udzial dodatnich routerow,
5. dlugosc histerezy alarmu i czyszczenia.

Funkcja celu karze za:

1. niskie `recall`,
2. wysokie `false_positive_rate`,
3. duzy `detection_delay`.

Obecnie koszt liczony jest jako:

```text
0.55 * recall_error + 0.30 * false_positive_rate + 0.15 * delay_term
```

gdzie:

1. `recall_error = 1.0 - recall`,
2. `delay_term = detection_delay / max(1, steps - attack_start)` dla scenariuszy z atakiem,
3. `delay_term = 0.0` dla scenariuszy bez ataku.

Wazne: `delay_term` jest znormalizowany do dlugosci pozostalego horyzontu ataku, wiec nie rosnace wprost z liczba krokow scenariusza.

## 3. Co za co odpowiada

### 3.1. BuilderConfig

Plik: [config.py](./ddos_ofn/config.py)

Parametry budowy OFN i ich domyslne wartosci:

1. `n_points = 256` - rozdzielczosc OFN.
2. `window_size = 4` - ile probek trafia do jednego OFN.
3. `history_size = 16` - ile historii jest uzywane do baseline.
4. `min_spread = 0.2` - minimalna szerokosc trapezu, zeby OFN nie degenerowal sie numerycznie.
5. `trend_epsilon = 0.15` - minimalna zmiana, od ktorej uznajemy trend za dodatni lub ujemny.
6. `anomaly_clip = 8.0` - symetryczne obciecie znormalizowanych odchylen.
7. `min_baseline_scale = 1.0` - dolna granica skali baseline.
8. `neutral_contribution = 0.25` - jak silnie neutralne routery dodatnio wplywaja na wynik globalny.

### 3.2. DetectorConfig

Parametry logiki alarmu i ich domyslne wartosci:

1. `alert_threshold = 4.0` - prog wlaczenia alarmu.
2. `clear_threshold = 2.0` - prog wygaszenia alarmu.
3. `alert_windows = 2` - ile kolejnych okien musi spelniac warunek wlaczenia.
4. `clear_windows = 2` - ile kolejnych okien musi spelniac warunek wygaszenia.
5. `min_positive_routers = 4` - minimalna liczba routerow z dodatnim kierunkiem.
6. `min_total_score = 0.0` - dodatkowy dolny prog na `score`.

`min_total_score` nie jest martwym parametrem. Jest aktywnym elementem warunku alarmu, ale przy domyslnej wartosci `0.0` pozostaje neutralny wobec juz obcietego `score >= 0`.

### 3.3. SimulationConfig

Parametry generatora danych syntetycznych i ich domyslne wartosci:

1. `routers = 30` - liczba routerow.
2. `steps = 160` - dlugosc scenariusza.
3. `seed = 7` - ziarno generatora.
4. `baseline_low = 80.0`, `baseline_high = 160.0` - zakres normalnego ruchu.
5. `noise_std = 4.0` - poziom szumu.
6. `attack_fraction = 0.7` - jaki procent routerow bierze udzial w ataku.
7. `attack_scale = 5.0` - sila narastajacego ataku.
8. `pulse_scale = 6.0` - sila ataku pulsacyjnego.
9. `flash_scale = 2.0` - sila legalnego skoku ruchu.
10. `attack_start = 80`, `attack_duration = 40` - kiedy atak sie zaczyna i jak dlugo trwa.

### 3.4. GAConfig

Parametry algorytmu genetycznego i ich domyslne wartosci:

1. `population_size = 36` - liczba kandydatow w populacji.
2. `generations = 24` - liczba generacji.
3. `mutation_rate = 0.12` - czestotliwosc mutacji.
4. `mutation_sigma = 0.18` - skala mutacji.
5. `crossover_rate = 0.75` - prawdopodobienstwo krzyzowania.
6. `tournament_k = 3` - rozmiar turnieju selekcyjnego.
7. `elite_count = 4` - ilu najlepszych przechodzi bez zmian.
8. `weight_bounds = (0.1, 3.0)` - zakres wag routerow.
9. `alert_threshold_bounds = (1.0, 10.0)` - zakres progu alarmu.
10. `clear_ratio_bounds = (0.25, 0.9)` - zakres relacji progu clear do progu alert.
11. `positive_fraction_bounds = (0.05, 0.8)` - zakres udzialu dodatnich routerow.
12. `hysteresis_bounds = (1, 5)` - dopuszczalny zakres liczby okien histerezy.
13. `seed = 13` - ziarno GA.

Wazne: GA nie stroi `min_total_score`. Ten parametr jest dziedziczony z bazowego `DetectorConfig` i pozostaje staly podczas optymalizacji.

## 4. Jak dostrajac projekt w praktyce

Najlepiej nie zaczynac od GA. Najpierw trzeba zobaczyc zachowanie baseline.

### Krok 1. Uruchom baseline

```bash
python scripts/eval_ddos.py --scenario ddos_ramp
python scripts/eval_ddos.py --scenario flash_crowd
python scripts/dashboard.py
```

Patrzysz wtedy na:

1. czy `ddos_ramp` jest wykrywany,
2. czy `flash_crowd` nie daje zbyt wielu falszywych alarmow,
3. czy opoznienie alarmu nie jest zbyt duze.

### Krok 2. Popraw logike alarmu

Jesli model za szybko alarmuje:

1. podnies `alert_threshold`,
2. podnies `min_positive_routers`,
3. zwieksz `alert_windows`,
4. w razie potrzeby podnies tez `min_total_score`, jesli chcesz wymusic silniejszy laczny score.

Jesli model gubi atak:

1. obniz `alert_threshold`,
2. zmniejsz `min_positive_routers`,
3. zmniejsz `alert_windows`,
4. sprawdz, czy `trend_epsilon` nie jest za duze.

Jesli alarm nie chce zgasnac:

1. zwieksz `clear_threshold`, aby warunek `score <= clear_threshold` byl latwiejszy do spelnienia,
2. zmniejsz `clear_windows`,
3. sprawdz, czy wagi routerow nie sa przesadnie duze.

### Krok 3. Uruchom GA

```bash
python scripts/train_ddos_ga.py
```

Albo w dashboardzie kliknij `DOSTROJ GA`.

Po treningu sprawdzasz:

1. `best_fitness`,
2. walidacje na zapisanym modelu,
3. top wagi routerow,
4. czy model nie zaczal zbyt mocno polegac na 1-2 routerach.

### Krok 4. Zweryfikuj zapisany model

Po treningu uruchamiasz jeszcze raz scenariusze w trybie `saved_tuned` i patrzysz, czy:

1. recall wzrosl,
2. FPR nie wzrosl za bardzo,
3. opoznienie alarmu spadlo albo pozostalo akceptowalne.

## 5. Jak testuje, czy to dziala

### 5.1. Testy jednostkowe

```bash
python -m pytest -q
```

Aktualne testy sprawdzaja:

1. budowe OFN z okna pomiarowego,
2. przelaczanie kierunku,
3. agregacje sygnalow routerow,
4. logike alarmu i histerezy,
5. podstawowe strojenie GA,
6. semantyke `detection_delay`.

Pliki:

1. [test_ofn_builder.py](./tests/test_ofn_builder.py)
2. [test_direction_switch.py](./tests/test_direction_switch.py)
3. [test_aggregation.py](./tests/test_aggregation.py)
4. [test_detector_rules.py](./tests/test_detector_rules.py)
5. [test_ga_optimize.py](./tests/test_ga_optimize.py)
6. [test_metrics.py](./tests/test_metrics.py)

### 5.2. Testy scenariuszowe

```bash
python scripts/eval_ddos.py --scenario normal
python scripts/eval_ddos.py --scenario ddos_ramp
python scripts/eval_ddos.py --scenario ddos_pulse
python scripts/eval_ddos.py --scenario flash_crowd
```

Oczekiwany sensowny efekt:

1. `normal` - `false_positive_rate` powinien byc bliski zera; dla MVP warto traktowac `<= 0.05` jako sensowny punkt odniesienia.
2. `ddos_ramp` - orientacyjnie `recall >= 0.85` i `detection_delay <= 5` krokow.
3. `ddos_pulse` - scenariusz trudniejszy; dla MVP sensownym minimum jest wykrywalnosc lepsza od przypadku, np. `recall >= 0.40`.
4. `flash_crowd` - detector nie powinien zachowywac sie tak agresywnie jak przy prawdziwym ataku; jako punkt odniesienia warto pilnowac `false_positive_rate <= 0.10`.

Te liczby sa proponowanymi progami walidacyjnymi dla MVP po recznym tuningu lub GA. Nie sa gwarancja, ze domyslna konfiguracja spelnia je bez strojenia.

### 5.3. Smoke test dashboardu

```bash
python scripts/dashboard.py --smoke-test
```

Ten test sprawdza, czy dashboard startuje, renderuje przebieg i nie wywala sie przy podstawowym uruchomieniu.

## 6. Jak czytac dashboard

Plik: [dashboard.py](./scripts/dashboard.py)

### Wykres 1. Globalny score OFN i alarmy

To jest najwazniejszy wykres.

1. linia `score` pokazuje laczny wynik po agregacji OFN,
2. linia `attack label` pokazuje prawde wzorcowa w scenariuszu syntetycznym,
3. linia `alarm` pokazuje decyzje detektora,
4. linia pozioma to prog alarmu.

### Wykres 2. Macierz ruchu router x czas

Pokazuje surowy ruch dla wszystkich routerow.

Sluzy do oceny:

1. czy atak obejmuje duza grupe routerow,
2. czy wzrosty sa rozproszone czy skupione,
3. czy sygnal wyglada jak atak rampowy, pulsacyjny albo legalny burst.

### Wykres 3. Kierunki routerow

Pokazuje, ile routerow:

1. wzmacnia alarm,
2. oslabia alarm,
3. jest neutralnych.

To jest dobry wykres do rozumienia, dlaczego score rosl albo malal.

### Wykres 4. Top wagi routerow

Pokazuje, ktore routery maja najwiekszy wplyw na decyzje.

Jesli 1-2 routery maja skrajnie dominujaca wage, to model moze byc nadmiernie dopasowany.

## 7. Kiedy uznac, ze model dziala dobrze

To zalezy od celu, ale dla MVP sensowny jest taki kierunek:

1. `recall >= 0.85` dla `ddos_ramp`,
2. `false_positive_rate <= 0.05` dla `normal`,
3. `false_positive_rate <= 0.10` dla `flash_crowd`,
4. `detection_delay <= 5` krokow dla scenariuszy atakowych,
5. brak dominacji pojedynczego routera w wagach po GA.

Praktycznie:

1. jesli recall jest wysokie, ale FPR tez jest wysokie, model jest zbyt agresywny,
2. jesli FPR jest niskie, ale recall slabe, model jest zbyt ostrozny,
3. jesli delay jest duzy, alarm pojawia sie za pozno.

## 8. Ograniczenia obecnego MVP

1. Dane sa syntetyczne, nie pochodza jeszcze z realnych routerow.
2. Model obserwuje glownie wolumen pakietow, a nie pelen zestaw cech sieciowych.
3. Strojenie GA jest celowo lekkie, zeby uruchamialo sie szybko na desktopie.
4. Zapisane wagi sa zwiazane z identyfikatorami routerow z danego scenariusza.
5. `flash_crowd` i `ddos_pulse` dalej sa dobrymi scenariuszami do dalszego dopracowania.

## 9. Najwazniejsze komendy

```bash
python -m pytest -q
python scripts/eval_ddos.py --scenario ddos_ramp
python scripts/train_ddos_ga.py
python scripts/dashboard.py
python scripts/dashboard.py --smoke-test
```

## 10. Najbardziej sensowny sposob dalszego rozwoju

1. dodac import realnych danych z CSV,
2. dodac kolejne cechy poza sama liczba pakietow,
3. rozbudowac suite walidacyjny o bardziej realistyczne typy ataku,
4. trzymac osobno zbiory treningowe i walidacyjne z realnych danych,
5. porownac ten model z prostym baseline progu ruchu i z klasycznym detektorem anomalii.
