# Primi passi con PyTorch

![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)
![Models](https://img.shields.io/badge/Models-CNN%20%7C%20ResNet18-blueviolet)
![Technique](https://img.shields.io/badge/Technique-Transfer%20Learning-red)
![Analysis](https://img.shields.io/badge/Analysis-EDA%20%7C%20Shortcut%20Learning-9cf)
![Task](https://img.shields.io/badge/Task-Image%20Classification-informational)
![Dataset](https://img.shields.io/badge/Dataset-2207%20images-yellow)
![Recall](https://img.shields.io/badge/ResNet18%20Recall-76%25-green)
![Status](https://img.shields.io/badge/Status-Learning%20Project-lightgrey)

Tre notebook Jupyter sulle reti neurali con PyTorch: si parte da una semplice regressione, si arriva a un classificatore di immagini con reti convoluzionali e transfer learning, e infine si analizza in modo critico il dataset per capire cosa abbiano davvero imparato i modelli.

| Notebook | Argomento |
|---|---|
| [`neural_network.ipynb`](neural_network.ipynb) | Tensori e prima rete neurale: regressione su dati numerici |
| [`ImageClassifier.ipynb`](ImageClassifier.ipynb) | Classificazione di immagini: riconoscere il mio gatto tra altri gatti |
| [`ImageClassifierV2.ipynb`](ImageClassifierV2.ipynb) | Analisi esplorativa del dataset: scorciatoie, duplicati e un esperimento controllato sul modello |

## `neural_network.ipynb`: la prima rete neurale

Introduzione ai concetti di base, con un problema volutamente semplice: dati 14 valori di temperatura in ingresso (`X`) e i corrispondenti valori convertiti (`Y`), la rete deve imparare la relazione tra i due.

- **Tensori**: la struttura dati di PyTorch al posto delle liste.
- **Modello**: si parte dall'idea di un singolo neurone (`y = wx + b`) e si usa poi una piccola rete `Linear(1→10) → ReLU → Linear(10→10) → ReLU → Linear(10→1)`.
- **Addestramento**: loss `MSELoss`, ottimizzatore `Adam` (learning rate 0.01), 3000 epoche.
- **Risultato**: la loss scende da circa 590 a 0.44 e la rete, dato `70` in ingresso, prevede circa `21.16`.

## `ImageClassifier.ipynb`: riconoscere un gatto

L'obiettivo è una rete che, data la foto di un gatto, dica se è **il mio gatto** oppure **un altro gatto**. Il notebook è diviso in sezioni, ognuna con una spiegazione:

1. Import e percorso dei dati
2. La rete **CatClassifier**: una CNN con due blocchi convoluzione + ReLU + max pooling e due layer fully connected (architettura in [`CCN.png`](CCN.png))
3. Dataset e trasformazioni (`ImageFolder`, resize a 224×224)
4. Divisione train / validation (80% / 20%)
5. Addestramento di CatClassifier (15 epoche, Adam, `CrossEntropyLoss`)
6. Valutazione sul validation set: accuratezza, precisione, recall, matrice di confusione
7. **Transfer learning con ResNet18** pre-addestrata su ImageNet: cosa vuol dire, perché si sostituisce il layer `fc`, cosa succede ai pesi pre-addestrati durante il fine-tuning
8. Valutazione della ResNet e confronto con CatClassifier

<p align="center"><img src="CCN.png" alt="Architettura di CatClassifier" width="300"></p>

### Risultati

Il dataset è molto sbilanciato (2111 foto di altri gatti contro 96 del mio), quindi l'accuratezza da sola è ingannevole: le metriche che contano sono precisione e recall sulla classe del mio gatto. Sullo stesso validation set (442 immagini, di cui 21 del mio gatto):

| | CatClassifier | ResNet18 (fine-tuning) |
|---|---|---|
| Accuratezza | 94.57% | 98.87% |
| Precisione | 20.00% | 100.00% |
| Recall | 4.76% (1 su 21) | 76.19% (16 su 21) |
| Falsi positivi | 4 | 0 |

La rete costruita da zero, con così pochi esempi, non riesce a riconoscere il gatto; la ResNet pre-addestrata ottiene risultati molto migliori. L'analisi in `ImageClassifierV2.ipynb` mostra però che il dataset contiene delle scorciatoie, quindi questi numeri vanno letti con cautela.

### Dataset

Le immagini **non sono incluse** nel repository: si scaricano da Google Drive come [`dataset.zip`](https://drive.google.com/file/d/13qbs0Y6xzZCZsVG2nLBZGbBhveMfb65f/view).

Il notebook si aspetta che le immagini estratte si trovino in una cartella `drive` allo stesso livello della cartella del repository, con una sottocartella per classe:

```
ML/
├── code/            ← questo repository
└── drive/
    ├── 00_othercat/ ← foto di altri gatti   (etichetta 0)
    └── 01_marte/    ← foto del gatto da riconoscere (etichetta 1)
```

`ImageFolder` assegna le etichette in ordine alfabetico delle cartelle, per questo i prefissi `00_` e `01_`. Se usi un altro percorso, basta modificare `data_dir` nella prima sezione del notebook. [`marte_sofa.jpg`](marte_sofa.jpg) è una foto esterna al dataset, usata per provare le reti su una singola immagine.

## `ImageClassifierV2.ipynb`: cosa hanno imparato davvero i modelli?

Un'analisi esplorativa (EDA) del dataset, scritta da zero dopo il primo notebook. La domanda di partenza: *le metriche della ResNet misurano davvero la capacità di riconoscere il gatto, o qualcos'altro?*

Il notebook procede dal più economico al più costoso: prima i metadati dei file, poi i pixel, poi classificatori che **non vedono mai il gatto**, infine un esperimento controllato sul modello già addestrato. Ogni sezione spiega *perché* si fa un'analisi, *cosa* si trova e *cosa se ne deduce*.

**Risultati principali**

- **Le due classi vengono da fonti diverse.** Gli altri gatti vengono da un dataset pubblico (lato massimo ≤ 500 px, 4:3, qualità JPEG ~75); le foto del mio gatto vengono tutte dal telefono, passate da un'app (lato massimo 800 px, formati 9:16 / 3:4 / 16:9, qualità JPEG ~20, metadati EXIF rimossi).
- **Il dataset è separabile perfettamente senza guardare il gatto.** Una sola regola, *lato massimo > 585 px* oppure *qualità JPEG ≤ 30*, classifica correttamente tutte le 2207 immagini (cross-validation stratificata a 5 fold).
- **Una parte della scorciatoia sopravvive al resize a 224×224.** Il solo rapporto larghezza/altezza, che diventa una deformazione dell'immagine, trova il 45% delle foto del mio gatto con precisione ≥ 90% (PR-AUC 0.55, contro 0.044 del caso).
- **Esperimento controllato sulla compressione.** Ricomprimendo le stesse immagini, senza riaddestrare la ResNet, a qualità 20 la probabilità assegnata al mio gatto aumenta per l'82% degli altri gatti (×2.6 in mediana), mentre il controllo a qualità 75 non ha alcun effetto. La rete è quindi sensibile alla compressione forte, ma l'effetto sulle decisioni è piccolo (1 previsione cambiata su 421).
- **Altro:** un duplicato esatto finito a cavallo tra training e validation, due scatti a raffica, file `.jpg` che in realtà sono GIF, assenza di normalizzazione.

La conclusione è un piano per un confronto onesto tra modelli: dataset ripulito, split per gruppi, stesso protocollo per tutte le reti e test controllati sulle scorciatoie.

## Come eseguire i notebook

Servono Python 3 e queste librerie:

```bash
pip install torch torchvision matplotlib scikit-learn pillow pandas
```

Poi si aprono i notebook in VS Code o Jupyter e si eseguono le celle in ordine. L'addestramento gira su CPU: CatClassifier richiede pochi minuti, la ResNet18 parecchio di più. La prima volta i pesi pre-addestrati della ResNet (~45 MB) vengono scaricati automaticamente.

La sezione 9 di `ImageClassifierV2.ipynb` usa la ResNet addestrata nel primo notebook: per rieseguirla bisogna prima salvarne i pesi con l'ultima cella di `ImageClassifier.ipynb` (`resnet18_v1.pth`, escluso dal repository tramite `.gitignore`). I risultati restano comunque visibili negli output salvati.

## Riferimenti

`ImageClassifier.ipynb` è stato realizzato seguendo il video **[Crea una AI che RICONOSCE IL TUO GATTO! Corso Pytorch Reti Neurali Convoluzione - Python AI - PART 1](https://www.youtube.com/watch?v=K0hhmD4VeTU)** del canale [PitoneProgrammatore](https://www.youtube.com/@PitoneProgrammatoreOfficial).

`ImageClassifierV2.ipynb` è un lavoro originale che parte dai risultati del primo notebook.
