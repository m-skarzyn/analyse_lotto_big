# analyse_lotto_big

1. Pełna separacja danych: Sztuczny wiersz do predykcji przyszłości (`N+1`) jest tworzony wyłącznie w pamięci RAM na ułamek sekundy w momencie odpytywania modelu. Nigdy nie trafia do `raw_events.parquet` ani `features_cache.parquet`. Wyniki trafiają do oddzielnego pliku `logs/predictions.parquet` wraz z wyliczoną datą kolejnego losowania (wtorek, czwartek lub sobota) i wersją modelu.
2. Plik parametryzacyjny (`config.json`): Wszystkie reguły tworzenia grupy `L_GROUP` (np. dzielnik) oraz definicje statystyk kroczących trzymasz w czytelnym pliku JSON. Dodanie nowej cechy sprowadza się do dopisania jednego wpisu w pliku konfiguracyjnym bez ingerencji w kod Pythona.


## Project Layout
```
analyse_lotto_big/
├── config.json                  # Parametry grupy L_GROUP oraz lista statystyk (features)
├── data/
│   ├── raw_events.parquet       # Czyste dane historyczne (ID_LOS, DATE_LOS, L_1..L_6)
│   └── features_cache.parquet   # Przeliczone cechy historyczne (tylko zaobserwowane)
├── models/
│   └── model_YYYYMMDD_HHMMSS.joblib
├── logs/
│   └── predictions.parquet      # Historia predykcji, daty docelowe i weryfikacja trafności
├── ingest.py                    # Pobieranie CSV i dopisywanie nowych losowań
├── features.py                  # Silnik liczący L_GROUP i cechy na podstawie config.json
├── train.py                     # Selekcja cech i trening nowej wersji modelu
└── predict.py                   # Weryfikacja starych prognoz i predykcja na kolejny dzień

└── requirements.txt   
```


## Jak będzie działać pętla weryfikacji w praktyce?

1. Gdy pojawia się nowe zdarzenie (`predict.py`): Skrypt dopisuje nowy rekord do `raw_events.parquet`, wylicza jego rzeczywistą grupę i sprawdza w `predictions_log.parquet`, co na ten temat przewidywały zapisane wcześniej modele (v1, v2 itd.). Zapisuje wynik trafienia (1 lub 0) do logu, a następnie generuje prognozę na kolejny krok.
2. Gdy wymyślisz nową cechę (`features.py` -> `train.py`): Dopisujesz jedną funkcję w `features.py` i uruchamiasz `train.py`. Skrypt wykrywa nową kolumnę, przeprowadza selekcję cech, trenuje nowy model i zapisuje go w folderze models/ z nową sygnaturą czasową, nie kasując poprzednich wersji.


## `ingest.py` – Pobieranie CSV i przyrostowa aktualizacja danych

Pobiera plik z URL, sprawdza po parze (ID_LOS, DATE_LOS), których rekordów jeszcze nie ma w `data/raw_events.parquet`, i dopisuje tylko nowe.


## `features.py` – Dynamiczne grupowanie i przeliczanie cech
Ten moduł czyta `config.json`, wylicza `L_GROUP` (pamiętając, że w Pythonie np.`floor(x / 10 + 0.5)` daje klasyczne zaokrąglenie matematyczne znane z Excela, w przeciwieństwie do domyślnego `round()` w Pythonie, które zaokrągla do parzystych). Zawiera też funkcję wyznaczającą kolejną datę (wtorek -> czwartek -> sobota -> wtorek).
Skrypt wspiera tryb przyrostowy na pliku `data/features_cache.parquet`.


## `train.py` – Automatyczna selekcja cech i wersjonowanie modeli

Skrypt sprawdza, czy pojawiły się nowe cechy (`feat_*`), odrzuca szum za pomocą *Permutation Importance* i zapisuje nową wersję modelu w katalogu `models/` tylko wtedy, gdy zestaw cech uległ zmianie (lub gdy wymusisz trening).
Filtruje grupy występujące rzadziej niż `min_class_count` (domyślnie 3 razy w historii), przeprowadza selekcję cech i zapisuje model.

## `predict.py` – Codzienna pętla: aktualizacja, ocena modeli i predykcja

Ten plik uruchamiasz, gdy pojawia się nowe zdarzenie. Pobiera CSV, sprawdza trafność wszystkich zapisanych wcześniej modeli na nowym zdarzeniu (zapisując wynik w `logs/evaluation.parquet`) i zwraca predykcję na kolejny krok
Gdy pojawia się nowe losowanie:
1. Dociąga nowe wiersze do raw_events.parquet.
2. Przelicza cechy historyczne (zapisując je w features_cache.parquet).
3. Sprawdza w logs/predictions.parquet, czy dla świeżo zaobserwowanych dat istniały wcześniejsze predykcje – jeśli tak, uzupełnia kolumny actual_group oraz is_correct (1 lub 0).
4. Tworzy w pamięci RAM sztuczny wiersz na kolejną datę (wtorek/czwartek/sobota), wywołuje predykcję dla każdego zapisanego modelu i dopisuje te prognozy do logs/predictions.parquet (z pustym wynikiem rzeczywistym, czekającym na kolejne losowanie).

Zamiast polegać tylko na jednym strzale, skrypt zapisuje w logs/predictions.parquet **Top-1** oraz listę **Top-3** najbardziej prawdopodobnych grup na kolejną datę, a po nadejściu wyników sprawdza zarówno trafienie główne (is_correct_top1), jak i obecność w pierwszej trójce (`is_correct_top3`).


## `config.json` – Plik parametryzacyjny

Tutaj definiujesz sposób tworzenia `L_GROUP` oraz listę metryk. Gdy zechcesz dodać nową metrykę (np. okno 20 losowań dla konkretnej grupy), dopisujesz ją tutaj i uruchamiasz `train.py`.
- lag: Jaka grupa wypadła 1, 2 lub \(K\) losowań temu.
- rolling_freq: Ile razy grupa z poprzedniego losowania wystąpiła w ostatnich \(N\) losowaniach.
- draws_since_last: Ile losowań minęło od poprzedniego wystąpienia tej konkretnej grupy.
- day_of_week: Dzień tygodnia zdarzenia (wtorek=1, czwartek=3, sobota=5) – kluczowy, bo znasz go przed losowaniem!


## Jak wygląda Twoja codzienna praca z tym kodem?

• Gdy pojawia się nowe zdarzenie: Uruchamiasz `python predict.py`. Skrypt sam dociąga brakujące wiersze z URL, dopisuje wyniki trafności do `logs/evaluation.parquet` dla każdego historycznego modelu i drukuje prognozę na kolejny krok.
• **Gdy wymyślisz nową statystykę**: Otwierasz `features.py`, dopisujesz jedną linijkę zaczynającą się od `res["feat_..."]` i uruchamiasz `python train.py`. Skrypt sam wykrywa nową kolumnę, sprawdza czy realnie poprawia ona wyniki na danych historycznych i zapisuje nową wersję modelu w katalogu `models/`.

