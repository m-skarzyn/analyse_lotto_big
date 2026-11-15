# Lokalny Pipeline Predykcji Statystycznych

Minimalistyczny, lokalny system w języku Python (Linux) służący do przyrostowego pobierania danych ze źródła CSV, wyliczania autorskich statystyk kroczących (*feature engineering*) oraz trenowania i ewaluacji modeli klasyfikacji sekwencyjnej (`LightGBM`).

## 1. Wymagania i instalacja

System nie wymaga żadnych zewnętrznych serwerów baz danych ani aplikacji działających w tle. Wszystkie dane zapisywane są w lekkim formacie kolumnowym **Apache Parquet** (`pyarrow`).

W aktywnym środowisku `venv` zainstaluj wymagane pakiety:

```bash
pip install pandas pyarrow lightgbm scikit-learn requests joblib
```

## 2. Struktura plików projektu

```text
projekt_predykcja/
├── ingest_config.json           # Parametry źródła CSV (URL, separator, mapowanie kolumn)
├── config.json                  # Reguła wyliczania L_GROUP oraz definicje cech (features)
├── ingest.py                    # Pobieranie CSV i dopisywanie brakujących rekordów
├── features.py                  # Silnik cech historycznych oraz generator wiersza przyszłości w RAM
├── train.py                     # Automatyczna selekcja cech i wersjonowanie modeli
├── predict.py                   # Weryfikacja starych prognoz i predykcja na kolejną datę
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

### Krok A: Konfiguracja wejścia (`ingest_config.json`)
Uzupełnij adres URL oraz dopasuj mapowanie nazw kolumn z Twojego pliku CSV na wewnętrzne nazwy systemu:

```json
{
  "csv_url": "https://twoj-adres-do-pliku.csv",
  "csv_separator": ",",
  "date_format": "%Y-%m-%d",
  "column_mapping": {
    "ID_LOS": "nazwa_kolumny_id_w_csv",
    "DATE_LOS": "nazwa_kolumny_daty_w_csv",
    "L_1": "liczba_1",
    "L_2": "liczba_2",
    "L_3": "liczba_3",
    "L_4": "liczba_4",
    "L_5": "liczba_5",
    "L_6": "liczba_6"
  }
}
```
*Jeśli w Twoim pliku CSV nie ma kolumny z przyrastającym identyfikatorem `ID_LOS`, ustaw w mapowaniu `"ID_LOS": null` – skrypt `ingest.py` wygeneruje go automatycznie.*

### Krok B: Konfiguracja grupy i cech (`config.json`)
Plik `config.json` definiuje sposób tworzenia klasy docelowej `L_GROUP` (dzielnik i separator) oraz listę statystyk (`features`), które mają zostać policzone dla każdego zdarzenia na podstawie historii.

## 4. Jak używać poszczególnych skryptów

### 1. Pobieranie danych (`ingest.py`)
Pobiera plik CSV, sprawdza po parze `(ID_LOS, DATE_LOS)` lub dacie, które rekordy są nowe, wylicza `YEAR_LOS` oraz `L_GROUP` i dopisuje wyłącznie brakujące wiersze do `data/raw_events.parquet`.
* Uruchomienie z domyślnym URL z `ingest_config.json`:
  ```bash
  python ingest.py
  ```
* Uruchomienie z nadpisaniem adresu URL z linii poleceń:
  ```bash
  python ingest.py --url "https://inny-adres.csv"
  ```

### 2. Trenowanie modelu (`train.py`)
Przelicza cechy historyczne, sprawdza czy w `config.json` pojawiły się nowe cechy lub zmieniła się reguła grupy. Jeśli wykryje zmianę, przeprowadza selekcję cech (*Permutation Importance*) i zapisuje nową wersję modelu w katalogu `models/` (nie kasując starszych wersji).
* Standardowe uruchomienie (trenuje tylko przy wykryciu zmian w `config.json`):
  ```bash
  python train.py
  ```
* Wymuszenie ponownego treningu na aktualnych danych (np. po dopisaniu wielu nowych obserwacji):
  ```bash
  python train.py --force
  ```

### 3. Codzienna predykcja i ewaluacja (`predict.py`)
Główny skrypt operacyjny. Wykonuje kolejno:
1. Pobiera najnowsze obserwacje z CSV (wywołując `ingest.py`).
2. Sprawdza w `logs/predictions.parquet`, czy dla świeżo zaobserwowanych dat istniały wcześniejsze predykcje – jeśli tak, uzupełnia wynik rzeczywisty oraz flagi trafności (`is_correct_top1`, `is_correct_top3`).
3. Tworzy w pamięci RAM tymczasowy wiersz dla kolejnej daty zdarzenia (wtorek, czwartek lub sobota) – **nigdy nie zapisując go w danych bazowych**.
4. Odpytuje wszystkie zapisane w `models/` wersje modeli i dopisuje ich prognozy (Top-1 oraz Top-3) do `logs/predictions.parquet`.

* Uruchomienie z domyślnym URL z `ingest_config.json`:
  ```bash
  python predict.py
  ```
* Uruchomienie z przekazaniem własnego adresu URL:
  ```bash
  python predict.py --url "https://twoj-adres-do-pliku.csv"
  ```

### 4. Przeliczanie samych cech (`features.py`)
Zazwyczaj wywoływany automatycznie przez `train.py` i `predict.py`, ale można go uruchomić ręcznie, aby odświeżyć plik podglądowy `data/features_cache.parquet` po edycji `config.json`:
```bash
python features.py
```
