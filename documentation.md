# Dokumentacja projektu

Implementacja dotyczy algorytmu odkrywania kolokacji przestrzennych opisanego w artykule:

Yan Huang, Shashi Shekhar, Hui Xiong, "Discovering Co-location Patterns from Spatial Data Sets: A General Approach", IEEE Transactions on Knowledge and Data Engineering, 16(12), 2004, s. 1472-1485.

Projekt implementuje wariant event-centric, w ktorym nie buduje sie sztucznych transakcji. Zamiast tego wykorzystuje sie relacje sasiedztwa przestrzennego oraz miary participation ratio, participation index i conditional probability.

## 1. Model danych

### Dane wejsciowe

Dane wejsciowe sa reprezentowane jako tabela zdarzen przestrzennych. Kazdy wiersz opisuje jedna instancje obiektu przestrzennego:

| Kolumna | Znaczenie |
| --- | --- |
| `instance_id` | unikalny identyfikator instancji |
| `feature_type` | typ cechy przestrzennej, np. `amenity=restaurant` |
| `x` | wspolrzedna X w metrycznym ukladzie odniesienia |
| `y` | wspolrzedna Y w metrycznym ukladzie odniesienia |

Dane przykładowe w projekcie pochodza z OpenStreetMap:

- `data/warsaw_osm_events.csv`
- `data/prague_osm_events.csv`

Skrypt `src/dataset.py` pobiera obiekty OSM, przypisuje im jedna etykiete `feature_type`, przeksztalca geometrie do projekcji metrycznej i zapisuje punkt reprezentatywny obiektu. Dzieki temu prog odleglosci `d` w algorytmie jest interpretowany w metrach.

### Zdarzenia i typy cech

Zgodnie z modelem event-centric z artykulu:

- typ cechy przestrzennej odpowiada elementowi zbioru `ET`, np. restauracja, bankomat, przystanek;
- instancja cechy odpowiada jednemu punktowi w przestrzeni;
- kolokacja jest uporzadkowana kanonicznie jako krotka typow cech, np. `("amenity=atm", "amenity=bank")`;
- jedna instancja moze miec tylko jeden typ cechy w danych przetwarzanych przez algorytm.

### Relacja sasiedztwa R

W artykule relacja sasiedztwa `R` jest parametrem algorytmu i moze byc dowolna, o ile odpowiada semantyce dziedziny. W tej implementacji przyjeto relacje euklidesowa:

```text
(i, j) nalezy do R wtedy i tylko wtedy, gdy odleglosc_euklidesowa(i, j) <= d
```

Do szybkiego wyznaczania sasiedztwa uzywany jest `scipy.spatial.cKDTree.query_pairs`. W strukturze sasiedztwa przechowywane sa tylko pary roznych instancji. Refleksyjnosc potrzebna dla kolokacji rozmiaru 1 jest obsluzona przez osobne zainicjalizowanie tablic jednoelementowych.

### Table instance i row instance

Dla kolokacji `C = {f1, f2, ..., fk}`:

- `row instance` to krotka instancji `(i1, i2, ..., ik)`, gdzie kazda instancja ma odpowiedni typ cechy i wszystkie instancje w krotce sa wzajemnie sasiednie;
- `table instance` to zbior wszystkich takich krotek dla danej kolokacji.

Przyklad dla kolokacji `("A", "B")`:

```text
table_instance(("A", "B")) = [(A.1, B.1), (A.2, B.4), (A.3, B.4)]
```

### Participation ratio i participation index

Dla kolokacji `C` i cechy `fi`:

```text
PR(C, fi) = liczba roznych instancji fi wystepujacych w table_instance(C)
            / liczba wszystkich instancji fi w danych
```

Participation index:

```text
PI(C) = min PR(C, fi) dla wszystkich fi nalezacych do C
```

`PI` jest miara prevalencji. Jest antymonotoniczna, wiec jezeli kolokacja nie spelnia progu `min_pi`, to jej nadzbiory nie musza byc rozwazane.

### Reguly kolokacji

Dla prevalent colocation `C = C1 union C2` generowana jest regula:

```text
C1 => C2
```

Warunkiem przyjecia reguly jest:

```text
conditional_probability(C1 => C2) >= min_cp
```

W implementacji:

```text
cp(C1 => C2) =
    liczba roznych projekcji C1 z table_instance(C1 union C2)
    / liczba row instances w table_instance(C1)
```

Dla poprzednika jednoelementowego mianownikiem jest liczba wszystkich instancji tej cechy.

## 2. Algorytm

### Pseudokod

```text
Wejscie:
    E - zbior instancji zdarzen: (instance_id, feature_type, x, y)
    d - prog sasiedztwa przestrzennego
    min_pi - minimalny participation index
    min_cp - minimalne conditional probability

Wyjscie:
    prevalent - zbior prevalent colocations z wartosciami PI
    rules - zbior regul kolokacji z PI i cp

Algorytm:
    1. Wczytaj wspolrzedne i typy cech.
    2. Policz liczbe instancji dla kazdego feature_type.
    3. Utworz P1:
        dla kazdego feature_type f:
            P1 zawiera kolokacje (f)
            table_instance((f)) = wszystkie instancje f
            PI((f)) = 1

    4. Zbuduj relacje sasiedztwa R:
        pairs = KDTree.query_pairs(distance=d)
        adjacency = lista sasiadow dla kazdej instancji

    5. Wygeneruj kandydatow rozmiaru 2:
        dla kazdej pary sasiadow (i, j) z pairs:
            jezeli feature(i) != feature(j):
                dodaj uporzadkowana krotke (i, j)
                do table_instance((feature_i, feature_j))

    6. Dla kazdej kolokacji rozmiaru 2:
        oblicz PR i PI
        jezeli PI >= min_pi:
            zapisz jako prevalent

    7. Ustaw k = 2.
       Dopoki istnieja prevalent colocations rozmiaru k:
            C_{k+1} = apriori_gen(P_k)

            dla kazdego kandydata c w C_{k+1}:
                p = c bez ostatniej cechy
                q = c bez przedostatniej cechy
                polacz table_instance(p) i table_instance(q)
                po wspolnym prefiksie instancji
                sprawdz warunek sasiedztwa dla dwoch nowych instancji

                jezeli powstala niepusta table_instance(c):
                    oblicz PR i PI
                    jezeli PI >= min_pi:
                        zapisz c jako prevalent

            k = k + 1

    8. Dla kazdej prevalent colocation c:
        dla kazdego niepustego wlasciwego podzbioru antecedent:
            consequent = c \ antecedent
            cp = liczba roznych projekcji antecedent z table_instance(c)
                 / liczba instancji antecedent
            jezeli cp >= min_cp:
                dodaj regule antecedent => consequent

    9. Zwroc prevalent colocations i rules.
```

### Generowanie kandydatow Apriori

Funkcja `apriori_gen` dziala zgodnie z klasycznym krokiem Apriori:

1. Laczy kolokacje rozmiaru `k`, ktore maja wspolny prefiks.
2. Tworzy kandydata rozmiaru `k + 1`.
3. Usuwa kandydata, jezeli dowolny jego podzbior rozmiaru `k` nie byl prevalent.

Dzieki antymonotonicznosci `PI` takie przycinanie jest poprawne.

### Generowanie table instances

Implementacja uzywa podejscia hybrydowego opisanego w artykule:

- dla kolokacji rozmiaru 2 stosowany jest join geometryczny na parach punktow znalezionych przez KD-tree;
- dla kolokacji rozmiaru 3 i wiekszych stosowany jest join kombinatoryczny po wspolnym prefiksie table instances, a nastepnie sprawdzany jest warunek sasiedztwa dla nowych instancji.

## 3. Roznice wzgledem algorytmu oryginalnego

Najwazniejsze roznice i uproszczenia:

| Obszar | Artykul | Implementacja |
| --- | --- | --- |
| Relacja sasiedztwa | Dowolna relacja `R` przekazana przez uzytkownika | Euklidesowa odleglosc `<= d` |
| Typ danych przestrzennych | Ogolnie obiekty przestrzenne, takze rozszerzone | Punkty reprezentatywne OSM |
| Generowanie par | Opisane jako spatial join | KD-tree `query_pairs` |
| Join dla wiekszych wzorcow | Sort-merge join na table instances | Slownik prefiksow + sprawdzenie adjacency |
| Multi-resolution pruning | Opisane jako opcjonalna optymalizacja | Nie zaimplementowano |
| Dane eksperymentalne | Syntetyczne i NASA climate dataset | OpenStreetMap dla Warszawy i Pragi |
| CRS i jednostki | Zalezne od danych | Automatyczna projekcja metryczna przez OSMnx |

Brak multi-resolution pruning nie zmienia definicji wykrywanych kolokacji. Wplywa glownie na wydajnosc dla duzych zbiorow danych. Implementacja nadal zachowuje glowna logike: generowanie kandydatow Apriori, table instances, participation index i reguly kolokacji.

## 4. Wyniki testow

### Testy jednostkowe

Uruchomienie:

```bash
uv run pytest
```

Wynik:

```text
21 passed
```

Zakres testow:

- `test_apriori_gen.py` - generowanie kandydatow i przycinanie Apriori;
- `test_build_neighbors.py` - budowa relacji sasiedztwa, w tym przypadki brzegowe;
- `test_generate_size2_geometric.py` - generowanie table instances dla par;
- `test_join_combinatorial.py` - join kombinatoryczny dla kolokacji wiekszych niz 2;
- `test_participation_index.py` - obliczanie participation ratio i participation index;
- `test_generate_rules.py` - generowanie regul i filtrowanie po `min_cp`;
- `test_smoketest.py` - test regresyjny odtwarzajacy wartosci PI z przykladu z artykulu.

Test smoke dla danych odpowiadajacych rysunkowi z artykulu sprawdza:

| Kolokacja | Oczekiwany PI |
| --- | ---: |
| `{A, B}` | 0.4 |
| `{A, C}` | 0.5 |
| `{B, C}` | 0.2 |
| `{A, B, C}` | 0.2 |

### Parametry wynikow w katalogu `results`

Aktualne pliki wynikowe w `results/` odpowiadaja parametrom:

```bash
uv run main data/warsaw_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
uv run main data/prague_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
```

### Zbiory danych

| Zbior | Liczba zdarzen | Najliczniejsze typy cech |
| --- | ---: | --- |
| Warszawa | 13235 | `highway=bus_stop`, `amenity=restaurant`, `shop=convenience`, `amenity=atm`, `amenity=cafe` |
| Praga | 11695 | `highway=bus_stop`, `amenity=restaurant`, `amenity=cafe`, `shop=convenience`, `leisure=park` |

### Wyniki liczbowe

| Zbior | Prevalent colocations | Reguly |
| --- | ---: | ---: |
| Warszawa | 19 | 12 |
| Praga | 48 | 58 |

Najsilniejsze kolokacje dla Warszawy:

| Kolokacja | Rozmiar | PI | Liczba row instances |
| --- | ---: | ---: | ---: |
| `amenity=restaurant, shop=convenience` | 2 | 0.670 | 4379 |
| `amenity=atm, amenity=pharmacy` | 2 | 0.665 | 1158 |
| `amenity=atm, amenity=restaurant` | 2 | 0.641 | 3945 |
| `amenity=cafe, amenity=restaurant` | 2 | 0.620 | 5162 |
| `amenity=atm, amenity=cafe` | 2 | 0.609 | 2560 |

Najsilniejsze kolokacje dla Pragi:

| Kolokacja | Rozmiar | PI | Liczba row instances |
| --- | ---: | ---: | ---: |
| `amenity=restaurant, shop=convenience` | 2 | 0.755 | 6860 |
| `amenity=cafe, amenity=restaurant` | 2 | 0.738 | 15004 |
| `amenity=atm, amenity=pharmacy` | 2 | 0.699 | 870 |
| `amenity=atm, amenity=cafe` | 2 | 0.699 | 3724 |
| `amenity=cafe, shop=convenience` | 2 | 0.659 | 3163 |

Najsilniejsze reguly dla Warszawy:

| Regula | Rozmiar | PI | cp |
| --- | ---: | ---: | ---: |
| `amenity=bank => amenity=atm` | 2 | 0.555 | 0.927 |
| `amenity=atm, amenity=cafe => amenity=restaurant` | 3 | 0.457 | 0.917 |
| `amenity=atm, amenity=restaurant => shop=convenience` | 3 | 0.415 | 0.866 |
| `amenity=cafe => amenity=restaurant` | 2 | 0.620 | 0.846 |
| `amenity=restaurant => highway=bus_stop` | 2 | 0.456 | 0.844 |

Najsilniejsze reguly dla Pragi:

| Regula | Rozmiar | PI | cp |
| --- | ---: | ---: | ---: |
| `amenity=atm, amenity=cafe, tourism=hotel => amenity=restaurant` | 4 | 0.432 | 0.999 |
| `amenity=atm, amenity=cafe, shop=convenience => amenity=restaurant` | 4 | 0.404 | 0.997 |
| `amenity=cafe, tourism=hotel => amenity=restaurant` | 3 | 0.517 | 0.996 |
| `amenity=atm, tourism=hotel => amenity=restaurant` | 3 | 0.455 | 0.994 |
| `shop=convenience, tourism=hotel => amenity=restaurant` | 3 | 0.412 | 0.991 |

### Wykresy

Projekt generuje dwa typy wykresow dla kazdego zbioru:

- `results/warsaw_summary.png`
- `results/warsaw_spatial_colocations.png`
- `results/prague_summary.png`
- `results/prague_spatial_colocations.png`

`*_summary.png` zawiera:

- liczbe prevalent colocations wedlug rozmiaru;
- ranking najwiekszych wartosci PI;
- rozrzut regul wzgledem `cp` i `PI`;
- ranking regul wedlug `cp`.

`*_spatial_colocations.png` pokazuje przestrzenne nalozenie instancji uczestniczacych w wybranych kolokacjach. Szare punkty stanowia tlo calego zbioru, a kolorowe markery wskazuja instancje, ktore faktycznie uczestnicza w table instances danej kolokacji.

Komentarz:

- W obu miastach bardzo wysokie PI osiagaja kolokacje laczace restauracje, sklepy convenience, kawiarnie, bankomaty i apteki. Jest to zgodne z intuicja dla centrow miejskich, gdzie uslugi konsumenckie wystepuja blisko siebie.
- Praga ma wiecej wykrytych kolokacji i regul niz Warszawa przy tych samych progach. W wynikach Pragi czesc bardzo silnych regul zawiera `tourism=hotel`, co sugeruje koncentracje hoteli w obszarach z restauracjami i kawiarniami.
- Reguly o wysokim `cp` nie zawsze oznaczaja najwyzsze `PI`. `cp` mierzy warunkowa przewidywalnosc konsekwentu z poprzednika, a `PI` dodatkowo wymaga udzialu wszystkich cech kolokacji.
- Wykresy przestrzenne nalezy interpretowac jako wizualna walidacje: punkty uczestniczace w danej kolokacji powinny tworzyc lokalne skupienia zgodne z progiem sasiedztwa `d=200 m`.

## 5. Skomentowany kod

Kod jest podzielony na moduly:

| Plik | Rola |
| --- | --- |
| `src/colocation.py` | rdzen algorytmu: kandydaci, sasiedztwo, table instances, PI, reguly |
| `src/dataset.py` | pobieranie i przygotowanie danych OSM |
| `src/main.py` | CLI do uruchamiania algorytmu i eksportu wynikow |
| `src/plot.py` | wykresy podsumowujace i przestrzenne |
| `src/visualize.py` | pomocnicza wizualizacja surowych danych |
| `tests/` | testy jednostkowe i regresyjne |

Najwazniejsze funkcje w `src/colocation.py`:

- `apriori_gen` - implementuje krok join/prune Apriori;
- `_build_neighbors` - buduje relacje sasiedztwa z KD-tree;
- `_generate_size2_geometric` - tworzy table instances dla kolokacji rozmiaru 2;
- `_join_combinatorial` - laczy table instances dla kolokacji rozmiaru 3 i wiekszych;
- `_participation_index` - liczy `PR` i `PI`;
- `_generate_rules` - generuje reguly kolokacji;
- `discover_colocations` - publiczny punkt wejscia laczacy wszystkie kroki algorytmu.

W kodzie zastosowano docstringi opisujace znaczenie funkcji, parametrow i zwracanych struktur. Przy bardziej zlozonych fragmentach dodano komentarze wyjasniajace, np. inicjalizacje table instances rozmiaru 1, wybor rodzicow do joinu kombinatorycznego oraz sposob przechowywania wartosci `PI` i `PR`.

## 6. Uruchamianie

Instalacja zaleznosci:

```bash
uv sync --dev
```

Uruchomienie testow:

```bash
uv run pytest
```

Odtworzenie wynikow i wykresow:

```bash
uv run main data/warsaw_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
uv run main data/prague_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
```

Pobranie danych OSM od nowa:

```bash
uv run dataset Warsaw Poland --stats
uv run dataset Prague "Czech Republic" --stats
```
