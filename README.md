# Traffic-sign-recognition

# Objectif du travail réalisé

Dans ce projet on va essayé de construire un modèle d'entrainnement qui est capable de classifier une image de panneaux routier. On doit fournir au modèle construit une image par exemple d'un panneau Stop et le programme doit nous dire de quels panneau il s'agit (Panneau Stop par exemple)

# Etapes de réalisation qui explique le choix de l’architecture du modèle et de l’algorithme d’apprentissage

- Exploration du Dataset fourni 
- Classification des images présentes dans le fichier **Test** sous format interpretable avec **ImageFolder**
- Transformation des images lors de leurs chargements :  

   - Les images ne sont pas toutes avec la même dimension, on s'assure alors que se soit le cas avec un **redimensionnement de 32pixels par 32pixels**
   - On transforme les images en tensor pour qu'en puisse faire des calculs matriciels en utilisant la librairie **torch**
   - On applique une normalization sur les images (j'ai essayé plusieurs normalisation déja définit, j'ai pris celle ou j'ai de meuilleurs résultat) 

- On charge en suite les données d'entraînements et de Test avec ces transformations
- On définit une taille de **batch = 128** (Pour une taille de batch = 256 le modèle se généralisait très mal)
- On veille à ce qu'ont mélange les données lors du chargement par lots afin d'éviter le cas qu'une classe soit présente en premiers avec un nombre de répétion propre à elle et ensuite une autre, etc on faisant cela, on veille à se que le modèle apprenne bien

- On définit un premier modèle avec 5 couches:
    - Dont 2 couches de convolution **(Détails des couches dans le code)**
    - 3 couches Linéaires **(Détails des couches dans le code)**
    ```py
            (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
            (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
            (fc1): Linear(in_features=1152, out_features=120, bias=True)
            (fc2): Linear(in_features=120, out_features=84, bias=True)
            (fc3): Linear(in_features=84, out_features=43, bias=True)
    ```  
    - On veille à utilisé à la fin **Softmax** car c'est l'idèale pour une classufication multiclass
    - Lors de l'entraînnement du modèle on utilise :  

        - La **CrossEntropyLoss** comme fonction de pert car on est dans une classification multiclass et on a utilisé Softmax
        - L'optimize **Adam** car c'est lui qui produit de meuilleur résultats
        - Un learning rate de **lr=0,001** (Avec un lr=0.0001) le programme prend trop de temps à s'exécuté
        - **30 epochs** : c'est suffisant car en remarque dans le graphe Accuracy et loss qu'à partir le l'epoch 10  environ le modèle à atteint une perte presque minimal et que la presion reste plus au moins stable à partir de 10 épochs. (cela ne sert à rien de mettre plus d'épochs)
        - Le modèle a une précision de 100% au niveau des trainning mais d'environ 90% à 94% lors de la phase de test (le modèle n'arrive pas à géneraliser bien les données)  

Les résultats de l'apprentissage sont les suivant :  

```py
        epoch : 1/30, loss = 488.633520, train_acc = 54.65, val_acc = 71.23
        epoch : 2/30, loss = 104.013614, train_acc = 90.24, val_acc = 86.77
        epoch : 3/30, loss = 49.707045, train_acc = 95.71, val_acc = 88.64
        epoch : 4/30, loss = 30.589001, train_acc = 97.49, val_acc = 89.68
        epoch : 5/30, loss = 21.830247, train_acc = 98.23, val_acc = 91.14
        epoch : 6/30, loss = 15.518154, train_acc = 98.71, val_acc = 91.12
        epoch : 7/30, loss = 12.550208, train_acc = 98.92, val_acc = 92.48
        epoch : 8/30, loss = 12.643125, train_acc = 98.87, val_acc = 92.82
        epoch : 9/30, loss = 9.260168, train_acc = 99.19, val_acc = 91.68
        epoch : 10/30, loss = 7.628214, train_acc = 99.32, val_acc = 92.36
        epoch : 11/30, loss = 7.354060, train_acc = 99.29, val_acc = 92.69
        epoch : 12/30, loss = 3.720191, train_acc = 99.67, val_acc = 92.91
        epoch : 13/30, loss = 6.073241, train_acc = 99.42, val_acc = 90.72
        epoch : 14/30, loss = 4.635571, train_acc = 99.58, val_acc = 93.06
        epoch : 15/30, loss = 5.749075, train_acc = 99.44, val_acc = 92.89
        epoch : 16/30, loss = 2.806275, train_acc = 99.74, val_acc = 92.07
        epoch : 17/30, loss = 4.691060, train_acc = 99.52, val_acc = 90.22
        epoch : 18/30, loss = 4.242615, train_acc = 99.60, val_acc = 93.69
        epoch : 19/30, loss = 3.876431, train_acc = 99.63, val_acc = 93.57
        epoch : 20/30, loss = 2.827184, train_acc = 99.72, val_acc = 91.14
        epoch : 21/30, loss = 4.380172, train_acc = 99.57, val_acc = 93.89
        epoch : 22/30, loss = 3.365056, train_acc = 99.67, val_acc = 92.62
        epoch : 23/30, loss = 3.330424, train_acc = 99.70, val_acc = 92.98
        epoch : 24/30, loss = 1.764263, train_acc = 99.82, val_acc = 93.16
        epoch : 25/30, loss = 0.444967, train_acc = 99.96, val_acc = 94.01
        epoch : 26/30, loss = 0.069774, train_acc = 100.00, val_acc = 94.30
        epoch : 27/30, loss = 0.037646, train_acc = 100.00, val_acc = 94.52
        epoch : 28/30, loss = 0.032114, train_acc = 100.00, val_acc = 94.54
        epoch : 29/30, loss = 0.024567, train_acc = 100.00, val_acc = 94.65
        epoch : 30/30, loss = 0.016576, train_acc = 100.00, val_acc = 94.62
```

- Afin d'avoir de meuilleurs performances j'ai fait plusieurs tests en utilisant différents type de Dropout(C'est une forme de régularisation) et changement le nombre de couches, le nombre de noeurones dans les couches, le nombre d'epochs et j'ai réussit à définir un modèle plus-tôt correct en utilisant :  

    - Des couches de convolution suivis de BatchNormalization et/ou de pooling avec différentes stride
    ```py
            (conv1): Conv2d(3, 120, kernel_size=(5, 5), stride=(1, 1))
            (conv1_bn): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (conv2): Conv2d(120, 160, kernel_size=(3, 3), stride=(1, 1))
            (conv2_bn): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(160, 250, kernel_size=(1, 1), stride=(1, 1))
            (conv3_bn): BatchNorm2d(250, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (fc1): Linear(in_features=2250, out_features=360, bias=True)
            (fc1_bn): BatchNorm1d(360, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.5, inplace=False)
            (fc2): Linear(in_features=360, out_features=120, bias=True)
            (fc3): Linear(in_features=120, out_features=84, bias=True)
            (fc4): Linear(in_features=84, out_features=43, bias=True)
    ```
    - Un Dropout de 0,5
    - En utilisant la **fonction d'activation elu** au lieu de relu (j'ai également testé la fonction d'activation **Leaky_ReLU** les résultats n'était pas aussi bon que celle de **elu**)
    - La fonction de perte reste la même (la mieux adapté) car on est dans une classification multiclass
    - L'optimizer reste Adam car c'est l'optimizer qui fournit de meuilleur résultat
    - Un learning rate de **lr=0,001** (Avec un lr=0.0001) le programme prend trop de temps à s'exécuté

    Les résultats de l'apprentissage sont les suivant :  

    ```py
        epoch : 1/50, loss = 383.544522, train_acc = 62.88, val_acc = 85.73
        epoch : 2/50, loss = 118.919049, train_acc = 87.69, val_acc = 91.84
        epoch : 3/50, loss = 76.792006, train_acc = 92.08, val_acc = 92.64
        epoch : 4/50, loss = 62.069460, train_acc = 93.57, val_acc = 94.52
        epoch : 5/50, loss = 48.558017, train_acc = 94.77, val_acc = 95.31
        epoch : 6/50, loss = 41.133014, train_acc = 95.57, val_acc = 95.39
        epoch : 7/50, loss = 37.423426, train_acc = 96.09, val_acc = 96.14
        epoch : 8/50, loss = 32.538206, train_acc = 96.65, val_acc = 95.95
        epoch : 9/50, loss = 32.594490, train_acc = 96.55, val_acc = 96.40
        epoch : 10/50, loss = 29.258125, train_acc = 96.94, val_acc = 96.44
        epoch : 11/50, loss = 25.228705, train_acc = 97.32, val_acc = 96.68
        epoch : 12/50, loss = 24.373867, train_acc = 97.42, val_acc = 96.88
        epoch : 13/50, loss = 24.924009, train_acc = 97.34, val_acc = 96.22
        epoch : 14/50, loss = 22.640959, train_acc = 97.68, val_acc = 96.69
        epoch : 15/50, loss = 21.470821, train_acc = 97.67, val_acc = 96.14
        epoch : 16/50, loss = 20.285575, train_acc = 97.82, val_acc = 96.82
        epoch : 17/50, loss = 18.668942, train_acc = 98.00, val_acc = 96.14
        epoch : 18/50, loss = 19.142459, train_acc = 97.98, val_acc = 96.12
        epoch : 19/50, loss = 18.389182, train_acc = 97.96, val_acc = 96.29
        epoch : 20/50, loss = 18.842782, train_acc = 98.02, val_acc = 96.62
        epoch : 21/50, loss = 16.355240, train_acc = 98.24, val_acc = 96.51
        epoch : 22/50, loss = 16.841317, train_acc = 98.21, val_acc = 96.92
        epoch : 23/50, loss = 18.494817, train_acc = 98.15, val_acc = 97.12
        epoch : 24/50, loss = 15.535410, train_acc = 98.35, val_acc = 96.69
        epoch : 25/50, loss = 14.198833, train_acc = 98.43, val_acc = 96.33
        epoch : 26/50, loss = 15.394787, train_acc = 98.41, val_acc = 96.98
        epoch : 27/50, loss = 16.730015, train_acc = 98.21, val_acc = 96.87
        epoch : 28/50, loss = 14.544597, train_acc = 98.48, val_acc = 96.75
        epoch : 29/50, loss = 14.586905, train_acc = 98.51, val_acc = 97.24
        epoch : 30/50, loss = 14.361549, train_acc = 98.49, val_acc = 96.89
        epoch : 31/50, loss = 14.637187, train_acc = 98.43, val_acc = 97.43
        epoch : 32/50, loss = 11.843690, train_acc = 98.69, val_acc = 97.44
        epoch : 33/50, loss = 13.840897, train_acc = 98.51, val_acc = 96.99
        epoch : 34/50, loss = 11.946140, train_acc = 98.76, val_acc = 97.13
        epoch : 35/50, loss = 12.456235, train_acc = 98.69, val_acc = 96.58
        epoch : 36/50, loss = 11.413202, train_acc = 98.80, val_acc = 96.87
        epoch : 37/50, loss = 12.002307, train_acc = 98.71, val_acc = 97.26
        epoch : 38/50, loss = 12.949800, train_acc = 98.68, val_acc = 96.79
        epoch : 39/50, loss = 11.844553, train_acc = 98.72, val_acc = 97.17
        epoch : 40/50, loss = 10.724169, train_acc = 98.80, val_acc = 96.63
        epoch : 41/50, loss = 10.953587, train_acc = 98.84, val_acc = 97.36
        epoch : 42/50, loss = 11.094376, train_acc = 98.85, val_acc = 96.92
        epoch : 43/50, loss = 10.414876, train_acc = 98.87, val_acc = 97.53
        epoch : 44/50, loss = 10.751014, train_acc = 98.87, val_acc = 97.10
        epoch : 45/50, loss = 10.893073, train_acc = 98.88, val_acc = 97.37
        epoch : 46/50, loss = 10.076255, train_acc = 98.93, val_acc = 96.88
        epoch : 47/50, loss = 10.372235, train_acc = 98.80, val_acc = 97.36
        epoch : 48/50, loss = 9.910371, train_acc = 98.97, val_acc = 97.01
        epoch : 49/50, loss = 9.766243, train_acc = 98.93, val_acc = 97.43
        epoch : 50/50, loss = 9.800699, train_acc = 98.97, val_acc = 97.26
    ```
    - 20 epochs était suffisantes pour avoir une accuracy maximal mais j'ai réalisé **50 epochs** afin d'être sûr des résultats obtenus
    - Dand ce modèle on remarque une fonction de perte qui diminue en fonction des epochs jusqu'à atteiendre un point minimal, on remarque égalament que le modèle arrive à avoir une précision assez proche entre les données de testes et les données d'entraînnement tout le long de l'apprentissage.

Pour résumer le modèle final a été réaliser suite a dinombrables tests et après avoir essayé plusieurs architectures différente, de même pour l'algorithme d'apprenstissage 

# Analyse de performances du modèle final

En regardant le graphique de la fonction de perte on remaque qu'effectivement le modèle au début ne pouvais pas bien classer les données car les poids et les biais ont était initialisé aléatoirement. Mais plus le temps passe et plus l'entrainnement continue plus le modèle apprend et ajuste les poids et les biais dans le but d'avoir une erreur minimale, On remarque qu'après un certain nombre d'epochs l'erreur à atteint un seuil ou elle ne peux plus dimininué (on a atteint l'erreur minimal).

On remarque également en évaluant le graphique de pression entre les données de test et d'entraînnements que au fure et à mesure que le temps passe les valeurs de test reste très proches aux valeurs d'entraînnement ce qui veux dire que notre modèle n'est pas en overfiting et qu'il a réuissi à se généraliser sur les données de test ou il n'a pas pu s'entraînner.

### Remarque
il n'est pas facile de savoir ce que nos convolutions filtres, mais je pense qu'elles essaye entre autre de définit les lignes horizontal et/ou verticale tout en essayant d'appliquer un filtre pour distinger les images flou et un filtre pour rendre les images non sombre (faire apparaître les panneaux) 

# Validation des résultats

Afin de mieux visualisé les classes ou le modèle a pu trop se généralisé ou non, j'ai utilisé la bibliothèque **seaborn** afin de réaliser un heatmaps qui nous permet selon la class voir le nombre de prédictions exact ou non et quels sont les prédictions erronées.

(**Le tableau peut être visualizer dans le fichier Amar_HENNI_final_CNN.ipynb**)

J'ai également utilisé la fonction **validate_top** afin de savoir le nombre d'erreur le plus élevès du modèle, on remarque qu'il y'à une classe ou le modèle a eu 155 erreurs de prédictions.

```py
    on validation set, accuracy top 2 = 0.9877276326207443 nb of errors =  155.0
    on validation set, accuracy top 3 = 0.9929532858273951 nb of errors =  89.0
    on validation set, accuracy top 4 = 0.9953285827395091 nb of errors =  59.0
    on validation set, accuracy top 5 = 0.9961203483768805 nb of errors =  49.0

```

Afin de d'avoir un aperçus des différentes classe et du pourcentage de prédictions correct de notre modèle sur ces classes j'ai utilisé la fonction **validate_category** qui a donnée les résultats suivant :

```py
    performens on validation dataset :
    classe :  0         Limitation de vitesse (20km/h), accuracy :  100  %
    classe :  1         Limitation de vitesse (30km/h), accuracy :  99  %
    classe :  2         Limitation de vitesse (50km/h), accuracy :  99  %
    classe :  3         Limitation de vitesse (60km/h), accuracy :  96  %
    classe :  4         Limitation de vitesse (70km/h), accuracy :  99  %
    classe :  5         Limitation de vitesse (80km/h), accuracy :  99  %
    classe :  6         Fin de limitation de vitesse (80km/h), accuracy :  88  %
    classe :  7         Limitation de vitesse (100km/h), accuracy :  98  %
    classe :  8         Limitation de vitesse (120km/h), accuracy :  97  %
    classe :  9         Interdiction de depasser      , accuracy :  100  %
    classe :  10        Interdiction de depasser pour vehicules > 3.5t, accuracy :  98  %
    classe :  11        Intersection ou vous etes prioritaire, accuracy :  99  %
    classe :  12        Route prioritaire             , accuracy :  96  %
    classe :  13        Ceder le passage              , accuracy :  99  %
    classe :  14        Arret a l intersection       , accuracy :  100  %
    classe :  15        Circulation interdite         , accuracy :  99  %
    classe :  16        Acces interdit aux vehicules > 3.5t, accuracy :  100  %
    classe :  17        Sens interdit                 , accuracy :  99  %
    classe :  18        Danger                        , accuracy :  87  %
    classe :  19        Virage à gauche              , accuracy :  100  %
    classe :  20        Virage à droite              , accuracy :  100  %
    classe :  21        Succession de virages         , accuracy :  76  %
    classe :  22        Cassis ou dos-d ane           , accuracy :  90  %
    classe :  23        Chaussee glissante            , accuracy :  89  %
    classe :  24        Chaussee retrecie par la droite, accuracy :  98  %
    classe :  25        Travaux en cours              , accuracy :  93  %
    classe :  26        Annonce feux                  , accuracy :  93  %
    classe :  27        Passage pietons               , accuracy :  53  %
    classe :  28        Endroit frequenté par les enfants, accuracy :  99  %
    classe :  29        Debouché de cyclistes        , accuracy :  98  %
    classe :  30        Neige ou glace                , accuracy :  74  %
    classe :  31        Passage d animaux sauvages    , accuracy :  99  %
    classe :  32        Fin des interdictions precedemment signalees, accuracy :  100  %
    classe :  33        Direction obligatoire à la prochaine intersection : à droite, accuracy :  99  %
    classe :  34        Direction obligatoire à la prochaine intersection : à gauche, accuracy :  99  %
    classe :  35        Direction obligatoire à la prochaine intersection : tout droit, accuracy :  99  %
    classe :  36        Direction obligatoire à la prochaine intersection : tout droit ou à droite, accuracy :  97  %
    classe :  37        Direction obligatoire à la prochaine intersection : tout droit ou à gauche, accuracy :  100  %
    classe :  38        Contournement obligatoire de l obstacle par la droite, accuracy :  99  %
    classe :  39        Contournement obligatoire de l obstacle par la gauche, accuracy :  97  %
    classe :  40        Carrefour giratoire           , accuracy :  97  %
    classe :  41        Fin d interdiction de depasser, accuracy :  80  %

```

On peut remarqué que la class pour la quelle notre modèle ne se généralise pas bien, c'est à dire la classe ou notre modèle a eu le plus d'erreurs de prédiction est la classe 27 **Passage pietons**  avec un pourcentage de réuissite de 53% seulement

Suite à cela j'ai fait un aperçu des images qui n'ont pas êtaient prédites correctement, il y'a eu au total 28 erreurs de classifications

# Script de test

Je n'ai pas réalisé de script de test séparer, mais un script integré dans mon code ou y'a une fonction **FinalResult** qui permet de chargé le modèle et une image, elle applique des transformations sur l'image (les mêmes que celle appliqué au modèle) 
```py
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)
```

Afin de correspondre les différentes class et index à nos images, il faut faire attention à **ImageFolder** qui réordonne les lables des classes. Pour savoir quelle ordre il a pris pour les classer, on regarde les données de test et la disposition des label avec **test_data.classes**, et ainsi on a les différentes valeurs des classes :

```py
        ['0','1','10','11','12','13','14','15','16','17','18','19','2','20','21','22','23','24','25','26','27','28','29','3','30','31','32','33','34','35','36','37','38','39','4','40','41','42','5','6','7','8','9']
```

Suite à ça je crée un dictionnaire qui fait correspondre un vrai label factice à un label factice, afin d'afficher les classes et indexes coresspondantes à l'image Et en suite elle utilise le modèle entrainné afin de dire à quelle classe apartient l'image en question ainsi que ça signification 





