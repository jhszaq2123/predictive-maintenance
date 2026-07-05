# Predictive Maintenance AI

Projekt do pracy magisterskiej pt. "Analiza efektywnosci wybranych modeli AI dla potrzeb budowy systemu do predykcji awarii maszyn przemyslowych na podstawie danych czujnikowych."

## Status projektu

Publiczny stan repozytorium odpowiada fundamentowi danych z `AI4I 2020` oraz pierwszemu zaakceptowanemu baseline'owi:

`CSV -> preprocessing -> Random Forest -> ewaluacja -> zapis modelu -> predykcja`

Repozytorium zawiera rowniez kod przygotowany do kolejnych eksperymentow, ale po zakonczeniu Sprintu 2 jedynym formalnie domknietym baseline'em modelowym pozostaje `Random Forest`.

## Zakres po Sprincie 2

Aktualnie gotowe i zweryfikowane sa:

- preprocessing danych `AI4I 2020`,
- zapis przetworzonych zbiorow treningowego i testowego,
- trening modelu `RandomForestClassifier`,
- ewaluacja metrykami klasyfikacyjnymi i confusion matrix,
- predykcja z zapisanego modelu.

Repozytorium zawiera rowniez kod przygotowany do kolejnych etapow, w tym `XGBoost` oraz podstawowy szkic API, ale elementy te nie sa jeszcze czescia formalnie zakonczonego porownania modeli ani kolejnych zaakceptowanych sprintow.

## Struktura repozytorium

```text
predictive-maintenance-ai/
|-- api/
|   `-- main.py
|-- data/
|   |-- interim/
|   |-- processed/
|   `-- raw/
|-- docs/
|   |-- data-sources.md
|   `-- project-plan.md
|-- experiments/
|   |-- ai4i_random_forest/
|   `-- run_ai4i_random_forest.py
|-- models/
|-- notebooks/
|   `-- 01_ai4i_eda.ipynb
|-- reports/
|-- scripts/
|   `-- download_datasets.ps1
|-- src/
|   |-- data_preprocessing.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- train_random_forest.py
|   |-- train_xgboost.py
|   `-- predictive_maintenance/
|-- tests/
|   |-- test_ai4i_data.py
|   |-- test_config.py
|   |-- test_datasets.py
|   `-- test_random_forest_pipeline.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Dane

Do uruchomienia pipeline'u po Sprincie 2 wymagany jest zbior `AI4I 2020`.

Kod preprocessingu oczekuje pliku:

```text
data/raw/ai4i2020.csv
```

Mozna przygotowac dane recznie albo skorzystac ze skryptu:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_datasets.ps1
```

Skrypt pobiera zbiory do lokalnego katalogu `data/` i dla `AI4I 2020` zapisuje plik rowniez do sciezki wymaganej przez pipeline.

Surowe dane, dane przetworzone, modele i wygenerowane raporty nie sa przeznaczone do publikacji w repozytorium i powinny byc odtwarzane lokalnie.

## Architektura pipeline'u

`src/data_preprocessing.py`

- wczytuje `AI4I 2020`,
- usuwa kolumny identyfikacyjne i kolumny powodujace wyciek informacji,
- przygotowuje target `Machine failure`,
- koduje zmienna `Type`,
- skaluje cechy numeryczne,
- dzieli dane na `train/test`,
- zapisuje dane przetworzone i artefakty preprocessingu.

`src/train_random_forest.py`

- trenuje `RandomForestClassifier`,
- zapisuje model do lokalnego katalogu `models/`.

`src/evaluate.py`

- oblicza `Accuracy`, `Precision`, `Recall`, `F1` i `ROC-AUC`,
- zapisuje metryki modelu,
- generuje confusion matrix,
- zapisuje porownanie modeli tylko przy wspolnej ewaluacji wielu modeli.

`src/predict.py`

- wczytuje zapisany model `Random Forest`,
- przyjmuje pojedynczy rekord maszyny,
- zwraca prawdopodobienstwo awarii i klase predykcji.

`api/main.py`

- zawiera szkic endpointow `GET /health` oraz `POST /predict`,
- korzysta z modelu `Random Forest`,
- pozostaje elementem przygotowanym do dalszej formalnej walidacji w kolejnych etapach projektu.

## Instalacja

```powershell
pip install -r requirements.txt
```

## Uruchomienie po Sprincie 2

1. Przygotuj dane `AI4I 2020` w `data/raw/ai4i2020.csv`.

2. Wykonaj preprocessing:

```powershell
python .\src\data_preprocessing.py
```

3. Wytrenuj baseline `Random Forest`:

```powershell
python .\src\train_random_forest.py
```

4. Uruchom ewaluacje baseline'u:

```powershell
python .\src\evaluate.py --model random_forest
```

5. Uruchom przykladowa predykcje:

```powershell
python .\src\predict.py
```

6. Opcjonalnie uruchom lokalny szkic API:

```powershell
uvicorn api.main:app --reload
```

Oficjalny wykres confusion matrix dla baseline'u `Random Forest` jest zapisywany jako:

```text
reports/figures/random_forest_confusion_matrix.png
```

## Uruchomienie testow

```powershell
pytest
```

Testy wymagajace lokalnych danych lub przetworzonych artefaktow sa automatycznie pomijane, jesli te pliki nie zostaly jeszcze przygotowane.

## API

Przykladowy payload dla endpointu `POST /predict`:

```json
{
  "Type": "M",
  "air_temperature_k": 298.5,
  "process_temperature_k": 309.0,
  "rotational_speed_rpm": 1450,
  "torque_nm": 48.0,
  "tool_wear_min": 120
}
```

## Proponowane pierwsze wydanie

Sugerowana nazwa pierwszego publicznego wydania:

`v0.1.0 - Random Forest Baseline`
