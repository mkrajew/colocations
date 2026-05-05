# Dokumentacja projektu

Implementacja dotyczy algorytmu odkrywania kolokacji przestrzennych opisanego w artykule:

Yan Huang, Shashi Shekhar, Hui Xiong, "Discovering Co-location Patterns from Spatial Data Sets: A General Approach", IEEE Transactions on Knowledge and Data Engineering, 16(12), 2004, s. 1472-1485.

Projekt implementuje wariant event-centric, w którym nie buduje się sztucznych transakcji. Zamiast tego wykorzystuje się relację sąsiedztwa przestrzennego oraz miary participation ratio, participation index i conditional probability.

## 1. Model danych

### Dane wejściowe

Dane wejściowe są reprezentowane jako tabela zdarzeń przestrzennych. Każdy wiersz opisuje jedną instancję obiektu przestrzennego:

| Kolumna | Znaczenie |
| --- | --- |
| `instance_id` | unikalny identyfikator instancji |
| `feature_type` | typ cechy przestrzennej, np. `amenity=restaurant` |
| `x` | współrzędna X w metrycznym układzie odniesienia |
| `y` | współrzędna Y w metrycznym układzie odniesienia |

Dane przykładowe w projekcie pochodzą z OpenStreetMap:

- `data/warsaw_osm_events.csv`
- `data/prague_osm_events.csv`

Skrypt `src/dataset.py` pobiera obiekty OSM, przypisuje im jedną etykietę `feature_type`, przekształca geometrię do projekcji metrycznej i zapisuje punkt reprezentatywny obiektu. Dzięki temu próg odległości `d` w algorytmie jest interpretowany w metrach.

### Zdarzenia i typy cech

Zgodnie z modelem event-centric z artykułu:

- typ cechy przestrzennej odpowiada elementowi zbioru `ET`, np. restauracja, bankomat, przystanek;
- instancja cechy odpowiada jednemu punktowi w przestrzeni;
- kolokacja jest uporządkowana kanonicznie jako krotka typów cech, np. `("amenity=atm", "amenity=bank")`;
- jedna instancja może mieć tylko jeden typ cechy w danych przetwarzanych przez algorytm.

### Relacja sąsiedztwa R

W artykule relacja sąsiedztwa `R` jest parametrem algorytmu i może być dowolna, o ile odpowiada semantyce dziedziny. W tej implementacji przyjęto relację euklidesową:

```text
(i, j) należy do R wtedy i tylko wtedy, gdy odległość_euklidesowa(i, j) <= d
```

Do szybkiego wyznaczania sąsiedztwa używany jest `scipy.spatial.cKDTree.query_pairs`. W strukturze sąsiedztwa przechowywane są tylko pary różnych instancji. Refleksyjność potrzebna dla kolokacji rozmiaru 1 jest obsłużona przez osobne zainicjalizowanie tablic jednoelementowych.

### Table instance i row instance

Dla kolokacji `C = {f1, f2, ..., fk}`:

- `row instance` to krotka instancji `(i1, i2, ..., ik)`, gdzie każda instancja ma odpowiedni typ cechy i wszystkie instancje w krotce są wzajemnie sąsiednie;
- `table instance` to zbiór wszystkich takich krotek dla danej kolokacji.

Przykład dla kolokacji `("A", "B")`:

```text
table_instance(("A", "B")) = [(A.1, B.1), (A.2, B.4), (A.3, B.4)]
```

### Participation ratio i participation index

Dla kolokacji `C` i cechy `fi`:

```text
PR(C, fi) = liczba różnych instancji fi występujących w table_instance(C)
            / liczba wszystkich instancji fi w danych
```

Participation index:

```text
PI(C) = min PR(C, fi) dla wszystkich fi należących do C
```

`PI` jest miarą prevalencji. Jest antymonotoniczna, więc jeżeli kolokacja nie spełnia progu `min_pi`, to jej nadzbiory nie muszą być rozważane.

### Reguły kolokacji

Dla prevalent colocation `C = C1 union C2` generowana jest reguła:

```text
C1 => C2
```

Warunkiem przyjęcia reguły jest:

```text
conditional_probability(C1 => C2) >= min_cp
```

W implementacji:

```text
cp(C1 => C2) =
    liczba różnych projekcji C1 z table_instance(C1 union C2)
    / liczba row instances w table_instance(C1)
```

Dla poprzednika jednoelementowego mianownikiem jest liczba wszystkich instancji tej cechy.

## 2. Algorytm

### Pseudokod

```text
Wejście:
    E - zbiór instancji zdarzeń: (instance_id, feature_type, x, y)
    d - próg sąsiedztwa przestrzennego
    min_pi - minimalny participation index
    min_cp - minimalne conditional probability

Wyjście:
    prevalent - zbiór prevalent colocations z wartościami PI
    rules - zbiór reguł kolokacji z PI i cp

Algorytm:
    1. Wczytaj współrzędne i typy cech.
    2. Policz liczbę instancji dla każdego feature_type.
    3. Utwórz P1:
        dla każdego feature_type f:
            P1 zawiera kolokację (f)
            table_instance((f)) = wszystkie instancje f
            PI((f)) = 1

    4. Zbuduj relację sąsiedztwa R:
        pairs = KDTree.query_pairs(distance=d)
        adjacency = lista sąsiadów dla każdej instancji

    5. Wygeneruj kandydatów rozmiaru 2:
        dla każdej pary sąsiadów (i, j) z pairs:
            jeżeli feature(i) != feature(j):
                dodaj uporządkowaną krotkę (i, j)
                do table_instance((feature_i, feature_j))

    6. Dla każdej kolokacji rozmiaru 2:
        oblicz PR i PI
        jeżeli PI >= min_pi:
            zapisz jako prevalent

    7. Ustaw k = 2.
       Dopóki istnieją prevalent colocations rozmiaru k:
            C_{k+1} = apriori_gen(P_k)

            dla każdego kandydata c w C_{k+1}:
                p = c bez ostatniej cechy
                q = c bez przedostatniej cechy
                połącz table_instance(p) i table_instance(q)
                po wspólnym prefiksie instancji
                sprawdź warunek sąsiedztwa dla dwóch nowych instancji

                jeżeli powstała niepusta table_instance(c):
                    oblicz PR i PI
                    jeżeli PI >= min_pi:
                        zapisz c jako prevalent

            k = k + 1

    8. Dla każdej prevalent colocation c:
        dla każdego niepustego właściwego podzbioru antecedent:
            consequent = c \ antecedent
            cp = liczba różnych projekcji antecedent z table_instance(c)
                 / liczba instancji antecedent
            jeżeli cp >= min_cp:
                dodaj regułę antecedent => consequent

    9. Zwróć prevalent colocations i rules.
```

### Generowanie kandydatów Apriori

Funkcja `apriori_gen` działa zgodnie z klasycznym krokiem Apriori:

1. Łączy kolokacje rozmiaru `k`, które mają wspólny prefiks.
2. Tworzy kandydata rozmiaru `k + 1`.
3. Usuwa kandydata, jeżeli dowolny jego podzbiór rozmiaru `k` nie był prevalent.

Dzięki antymonotoniczności `PI` takie przycinanie jest poprawne.

### Generowanie table instances

Implementacja używa podejścia hybrydowego opisanego w artykule:

- dla kolokacji rozmiaru 2 stosowany jest join geometryczny na parach punktów znalezionych przez KD-tree;
- dla kolokacji rozmiaru 3 i większych stosowany jest join kombinatoryczny po wspólnym prefiksie table instances, a następnie sprawdzany jest warunek sąsiedztwa dla nowych instancji.

## 3. Różnice względem algorytmu oryginalnego

Najważniejsze różnice i uproszczenia:

| Obszar | Artykuł | Implementacja |
| --- | --- | --- |
| Relacja sąsiedztwa | Dowolna relacja `R` przekazana przez użytkownika | Euklidesowa odległość `<= d` |
| Typ danych przestrzennych | Ogólnie obiekty przestrzenne, także rozszerzone | Punkty reprezentatywne OSM |
| Generowanie par | Opisane jako spatial join | KD-tree `query_pairs` |
| Join dla większych wzorców | Sort-merge join na table instances | Słownik prefiksów + sprawdzenie adjacency |
| Multi-resolution pruning | Opisane jako opcjonalna optymalizacja | Nie zaimplementowano |
| Dane eksperymentalne | Syntetyczne i NASA climate dataset | OpenStreetMap dla Warszawy i Pragi |
| CRS i jednostki | Zależne od danych | Automatyczna projekcja metryczna przez OSMnx |

Brak multi-resolution pruning nie zmienia definicji wykrywanych kolokacji. Wpływa głównie na wydajność dla dużych zbiorów danych. Implementacja nadal zachowuje główną logikę: generowanie kandydatów Apriori, table instances, participation index i reguły kolokacji.

## 4. Wyniki testów

### Testy jednostkowe

Uruchomienie:

```bash
uv run pytest
```

Wynik:

```text
21 passed
```

Zakres testów:

- `test_apriori_gen.py` - generowanie kandydatów i przycinanie Apriori;
- `test_build_neighbors.py` - budowa relacji sąsiedztwa, w tym przypadki brzegowe;
- `test_generate_size2_geometric.py` - generowanie table instances dla par;
- `test_join_combinatorial.py` - join kombinatoryczny dla kolokacji większych niż 2;
- `test_participation_index.py` - obliczanie participation ratio i participation index;
- `test_generate_rules.py` - generowanie reguł i filtrowanie po `min_cp`;
- `test_smoketest.py` - test regresyjny odtwarzający wartości PI z przykładu z artykułu.

Test smoke dla danych odpowiadających rysunkowi z artykułu sprawdza:

| Kolokacja | Oczekiwany PI |
| --- | ---: |
| `{A, B}` | 0.4 |
| `{A, C}` | 0.5 |
| `{B, C}` | 0.2 |
| `{A, B, C}` | 0.2 |

### Parametry wyników w katalogu `results`

Aktualne pliki wynikowe w `results/` odpowiadają parametrom:

```bash
uv run main data/warsaw_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
uv run main data/prague_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
```

### Zbiory danych

| Zbiór | Liczba zdarzeń | Najliczniejsze typy cech |
| --- | ---: | --- |
| Warszawa | 13235 | `highway=bus_stop`, `amenity=restaurant`, `shop=convenience`, `amenity=atm`, `amenity=cafe` |
| Praga | 11695 | `highway=bus_stop`, `amenity=restaurant`, `amenity=cafe`, `shop=convenience`, `leisure=park` |

### Wyniki liczbowe

| Zbiór | Prevalent colocations | Reguły |
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

Najsilniejsze reguły dla Warszawy:

| Reguła | Rozmiar | PI | cp |
| --- | ---: | ---: | ---: |
| `amenity=bank => amenity=atm` | 2 | 0.555 | 0.927 |
| `amenity=atm, amenity=cafe => amenity=restaurant` | 3 | 0.457 | 0.917 |
| `amenity=atm, amenity=restaurant => shop=convenience` | 3 | 0.415 | 0.866 |
| `amenity=cafe => amenity=restaurant` | 2 | 0.620 | 0.846 |
| `amenity=restaurant => highway=bus_stop` | 2 | 0.456 | 0.844 |

Najsilniejsze reguły dla Pragi:

| Reguła | Rozmiar | PI | cp |
| --- | ---: | ---: | ---: |
| `amenity=atm, amenity=cafe, tourism=hotel => amenity=restaurant` | 4 | 0.432 | 0.999 |
| `amenity=atm, amenity=cafe, shop=convenience => amenity=restaurant` | 4 | 0.404 | 0.997 |
| `amenity=cafe, tourism=hotel => amenity=restaurant` | 3 | 0.517 | 0.996 |
| `amenity=atm, tourism=hotel => amenity=restaurant` | 3 | 0.455 | 0.994 |
| `shop=convenience, tourism=hotel => amenity=restaurant` | 3 | 0.412 | 0.991 |

### Wykresy

Projekt generuje dwa typy wykresów dla każdego zbioru:

- `results/warsaw_summary.png`
- `results/warsaw_spatial_colocations.png`
- `results/prague_summary.png`
- `results/prague_spatial_colocations.png`

`*_summary.png` zawiera:

- liczbę prevalent colocations według rozmiaru;
- ranking największych wartości PI;
- rozrzut reguł względem `cp` i `PI`;
- ranking reguł według `cp`.

`*_spatial_colocations.png` pokazuje przestrzenne nałożenie instancji uczestniczących w wybranych kolokacjach. Szare punkty stanowią tło całego zbioru, a kolorowe markery wskazują instancje, które faktycznie uczestniczą w table instances danej kolokacji.

Komentarz:

- W obu miastach bardzo wysokie PI osiągają kolokacje łączące restauracje, sklepy convenience, kawiarnie, bankomaty i apteki. Jest to zgodne z intuicją dla centrów miejskich, gdzie usługi konsumenckie występują blisko siebie.
- Praga ma więcej wykrytych kolokacji i reguł niż Warszawa przy tych samych progach. W wynikach Pragi część bardzo silnych reguł zawiera `tourism=hotel`, co sugeruje koncentrację hoteli w obszarach z restauracjami i kawiarniami.
- Reguły o wysokim `cp` nie zawsze oznaczają najwyższe `PI`. `cp` mierzy warunkową przewidywalność konsekwentu z poprzednika, a `PI` dodatkowo wymaga udziału wszystkich cech kolokacji.
- Wykresy przestrzenne należy interpretować jako wizualną walidację: punkty uczestniczące w danej kolokacji powinny tworzyć lokalne skupienia zgodne z progiem sąsiedztwa `d=200 m`.

## 5. Skomentowany kod

Kod jest podzielony na moduły:

| Plik | Rola |
| --- | --- |
| `src/colocation.py` | rdzeń algorytmu: kandydaci, sąsiedztwo, table instances, PI, reguły |
| `src/dataset.py` | pobieranie i przygotowanie danych OSM |
| `src/main.py` | CLI do uruchamiania algorytmu i eksportu wyników |
| `src/plot.py` | wykresy podsumowujące i przestrzenne |
| `src/visualize.py` | pomocnicza wizualizacja surowych danych |
| `tests/` | testy jednostkowe i regresyjne |

Najważniejsze funkcje w `src/colocation.py`:

- `apriori_gen` - implementuje krok join/prune Apriori;
- `_build_neighbors` - buduje relację sąsiedztwa z KD-tree;
- `_generate_size2_geometric` - tworzy table instances dla kolokacji rozmiaru 2;
- `_join_combinatorial` - łączy table instances dla kolokacji rozmiaru 3 i większych;
- `_participation_index` - liczy `PR` i `PI`;
- `_generate_rules` - generuje reguły kolokacji;
- `discover_colocations` - publiczny punkt wejścia łączący wszystkie kroki algorytmu.

W kodzie zastosowano docstringi opisujące znaczenie funkcji, parametrów i zwracanych struktur. Przy bardziej złożonych fragmentach dodano komentarze wyjaśniające, np. inicjalizację table instances rozmiaru 1, wybór rodziców do joinu kombinatorycznego oraz sposób przechowywania wartości `PI` i `PR`.

## 6. Uruchamianie

Instalacja zależności:

```bash
uv sync --dev
```

Uruchomienie testów:

```bash
uv run pytest
```

Odtworzenie wyników i wykresów:

```bash
uv run main data/warsaw_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
uv run main data/prague_osm_events.csv --distance 200 --prevalence 0.4 --conditional 0.8 --plot
```

Pobranie danych OSM od nowa:

```bash
uv run dataset Warsaw Poland --stats
uv run dataset Prague "Czech Republic" --stats
```
