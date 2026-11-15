# Lokalny System Predykcji Grup Zdarzeń

Minimalistyczny, lokalny pipeline uczenia maszynowego (Linux + Python) do przewidywania kolejnej grupy liczbowej `L_GROUP` na podstawie historii zdarzeń. System nie wymaga baz danych ani usług działających w tle — wszystkie dane przechowywane są w lekkich plikach Apache Parquet (`pyarrow`), a modele w plikach `.joblib` (`LightGBM`).


## 1. Instalacja środowiska

Upewnij się, że masz aktywne środowisko wirtualne (`venv`), a następnie zainstaluj wymagane biblioteki:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas pyarrow lightgbm scikit-learn requests joblib
```


## 2. Struktura plików projektu
```text
projekt_predykcja/
├── ingest_config.json           # Konfiguracja pobierania danych z sieci (URL, separator, format daty, flaga nagłówka `has_header` oraz mapowanie po nazwach lub indeksach kolumn).
├── config.json                  # Konfiguracja reguły wyliczania `L_GROUP` (dzielnik, separator) oraz definicje metryk (`features`), które mają być wyliczane i testowane przez model.
├── ingest.py                    # Skrypt pobierający CSV, mapujący kolumny, wyliczający `YEAR_LOS` oraz `L_GROUP`, a następnie dopisujący wyłącznie brakujące rekordy do `data/raw_events.parquet`.
├── features.py                  # Silnik przeliczający wskaźniki historyczne (z przesunięciem w dół bez wycieku danych z przyszłości) oraz generujący sztuczny wiersz kolejnego losowania (wtorek, czwartek, sobota) wyłącznie w pamięci RAM.
├── train.py                     # Skrypt trenujący nowy model `LightGBM` z automatyczną selekcją cech (*Permutation Importance*) i zapisujący wersję modelu w katalogu `models/`.
├── predict.py                   # Główny skrypt codziennego obiegu: pobiera nowe rekordy, sprawdza trafność wcześniejszych predykcji w `logs/predictions.parquet` i generuje prognozę na kolejną datę.
├── README.md                    # Dokumentacja projektu
├── PROJECT_CONTEXT.txt          # Opis projektu do wykorzystania w przyszłych promptach AI
├── requirements.txt             # lista bibliotek wymaganych do pracy
├── data/                        # (Tworzone automatycznie)
│   ├── raw_events.parquet       # Czyste obserwacje historyczne
│   └── features_cache.parquet   # Przeliczone cechy historyczne
├── models/                      # (Tworzone automatycznie)
│   └── model_YYYYMMDD_HHMMSS.joblib
└── logs/                        # (Tworzone automatycznie)
    └── predictions.parquet      # Historia predykcji i weryfikacja ich trafności
```

## 3. Konfiguracja przed pierwszym uruchomieniem

### A. Dostosowanie źródła CSV (`ingest_config.json`)
Plik obsługuje zarówno pliki CSV posiadające wiersz nagłówkowy, jak i pliki bez nagłówka (wskazanie po numerze kolumny licząc od `0`):

```json
{
  "csv_url": "https://twoj-serwer.pl/dane_losowan.csv",
  "separator": ",",
  "encoding": "utf-8",
  "date_format": "%Y-%m-%d",
  "has_header": false,
  "column_mapping_by_name": {
    "ID_LOS": "numer_losowania",
    "DATE_LOS": "data_losowania",
    "L_1": "liczba_1",
    "L_2": "liczba_2",
    "L_3": "liczba_3",
    "L_4": "liczba_4",
    "L_5": "liczba_5",
    "L_6": "liczba_6"
  },
  "column_mapping_by_index": {
    "ID_LOS": 0,
    "DATE_LOS": 1,
    "L_1": 2,
    "L_2": 3,
    "L_3": 4,
    "L_4": 5,
    "L_5": 6,
    "L_6": 7
  }
}
```
* Jeśli `has_header` ma wartość `false`, skrypt użyje słownika `column_mapping_by_index` (gdzie `0` to pierwsza kolumna w pliku CSV).
* Jeśli `has_header` ma wartość `true`, skrypt użyje słownika `column_mapping_by_name`.


## 4. Jak używać skryptów

### Krok 1: Inicjalne pobranie danych i pierwszy trening
Uruchom kolejno pobieranie danych historycznych oraz trening pierwszego modelu:

```bash
python ingest.py
python train.py
```

### Krok 2: Codzienny obieg (aktualizacja + ocena + predykcja)
Gdy pojawi się nowe zdarzenie, uruchom:

```bash
python predict.py
```
Możesz również podać inny adres URL pliku CSV bezpośrednio z linii poleceń:
```bash
python predict.py --url "https://inny-adres.pl/aktualne.csv"
```

### Krok 3: Dodawanie nowych statystyk (eksperymenty)
1. Otwórz plik `config.json` i dopisz nowy obiekt do listy `"features"` (np. z innym oknem `window` lub `years_back`).
2. Uruchom ponownie trening:
   ```bash
   python train.py
   ```
   Skrypt wykryje nową cechę, sprawdzi jej przydatność i utworzy nową wersję modelu w katalogu `models/` bez usuwania starszych wersji.
3. Aby wymusić ponowne przeliczenie modelu bez zmiany pliku `config.json` (np. po zebraniu dużej porcji nowych losowań), użyj flagi `--force`:
   ```bash
   python train.py --force
   ```
