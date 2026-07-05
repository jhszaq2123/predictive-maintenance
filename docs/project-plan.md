# Project Plan

## Temat pracy

Analiza efektywnosci wybranych modeli AI dla potrzeb budowy systemu do predykcji awarii maszyn przemyslowych na podstawie danych czujnikowych.

## Glowny cel

Porownac kilka klas modeli sztucznej inteligencji w trzech scenariuszach predictive maintenance:

- predykcja `RUL`,
- klasyfikacja awarii,
- detekcja anomalii.

## Pytania badawcze

1. Czy modele sekwencyjne `LSTM` i `GRU` sa skuteczniejsze od klasycznych modeli ML w analizie danych czasowych?
2. Czy modele zespolowe, takie jak `Random Forest` i `XGBoost`, sa mocnym punktem odniesienia dla klasyfikacji awarii?
3. Czy `Autoencoder` pozwala wykrywac anomalie mimo ograniczonej liczby przykladow awarii?

## Hipotezy robocze

- `LSTM` i `GRU` powinny miec przewage w zadaniach sekwencyjnych i predykcji `RUL`.
- `Random Forest` i `XGBoost` powinny dawac stabilne wyniki w klasyfikacji awarii.
- `Autoencoder` powinien byc sensowny w scenariuszu anomaly detection przy ograniczonych etykietach.

## Zbiory danych i mapowanie scenariuszy

| Dataset | Scenariusz glowny | Typ zadania |
| --- | --- | --- |
| NASA CMAPSS | Predykcja RUL | Regresja |
| AI4I 2020 | Klasyfikacja awarii | Klasyfikacja |
| SECOM lub AI4I 2020 | Detekcja anomalii | Anomaly detection |

## Modele do implementacji

### Regresja RUL

- `MLP`
- `Random Forest Regressor`
- `XGBoost Regressor`
- `LSTM`
- `GRU`

### Klasyfikacja awarii

- `MLP`
- `Random Forest Classifier`
- `XGBoost Classifier`

### Detekcja anomalii

- baseline statystyczny lub distance-based
- `Autoencoder`

## Metryki

### RUL

- `MAE`
- `RMSE`

### Klasyfikacja

- `Accuracy`
- `Precision`
- `Recall`
- `F1`
- `ROC-AUC`

### Anomalie

- `Precision`
- `Recall`
- `F1`
- `ROC-AUC`
- blad rekonstrukcji i analiza progu decyzyjnego

## Plan implementacyjny

### Etap 1. Fundament projektu

- [x] inicjalizacja repo
- [x] podstawowa struktura katalogow
- [x] dokument planu projektu
- [ ] przygotowanie srodowiska i zaleznosci treningowych

### Etap 2. Dane

- [ ] pobranie lub umieszczenie datasetow w `data/raw/`
- [ ] opis slownikow danych
- [ ] loader dla `NASA CMAPSS`
- [ ] loader dla `AI4I 2020`
- [ ] loader dla `SECOM`

### Etap 3. EDA i preprocessing

- [ ] analiza brakow danych
- [ ] analiza rozkladow i klas
- [ ] standaryzacja pipeline'u czyszczenia danych
- [ ] konstrukcja sekwencji czasowych
- [ ] engineering cech okien czasowych

### Etap 4. Modele i eksperymenty

- [ ] baseline'y
- [ ] modele klasyczne ML
- [ ] modele sekwencyjne
- [ ] autoencoder
- [ ] strojenie hiperparametrow

### Etap 5. Ewaluacja

- [ ] porownanie modeli na wspolnych metrykach
- [ ] analiza stabilnosci wynikow
- [ ] analiza odpornosci na zaklocenia i braki danych
- [ ] analiza czasu inferencji

### Etap 6. Prototyp systemu

- [ ] REST API
- [ ] integracja najlepszego modelu
- [ ] dashboard z wizualizacja predykcji

## Proponowana kolejnosc prac

1. `AI4I 2020` jako pierwszy dataset, bo jest prostszy do wejscia i dobrze nadaje sie do szybkiego pipeline'u.
2. `NASA CMAPSS` jako glowny tor dla `RUL`.
3. `SECOM` jako uzupelnienie do scenariusza anomalii, jesli jakosc danych bedzie akceptowalna.

## Co robimy teraz

Najblizszy krok techniczny:

- dodac loadery danych,
- przygotowac pierwszy eksperyment dla `AI4I 2020`,
- zrobic notebook EDA dla pierwszego datasetu.
