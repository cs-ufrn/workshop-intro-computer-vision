
# k-Nearest Neighbors Classifier (k-NN)

O **k-NN** é o classificador mais simples na área de aprendizado de máquina. Diferentemente das redes neurais, não se realiza de fato um **aprendizado**, em vez disso, o algoritmo verifica a distância entre o objeto a ser classificado e os vetores de característica. Devido a sua simplicidade, é bastante utilizado em *benchmarks* de classificadores mais complexos como, Artificial Neural Network (**ANN**) e Suport Vector Machine (**SVM**).

<figure>
    <img src="https://amueller.github.io/applied_ml_spring_2017/images/classifier_comparison.png" alt="Classifier Comparison">
    <figcaption>Figura 1 - Comparação entre os classificadores (https://amueller.github.io/applied_ml_spring_2017/images/classifier_comparison.png).</figcaption>
</figure>,

![classifier_comparison](https://amueller.github.io/applied_ml_spring_2017/images/classifier_comparison.png)

## 1. Introdução

O **k-NN** não passa por um processo de aprendizagem como o **ANN** e **SVM**, contudo existe um mecanismo de *mapeamento* que servirá para criar o modelo utilizado na classificação dos dados. Este modelo consiste numa lista de **descritores**, que tratam-se de listas das características que descrevem os exemplos dos modelos.

<center>
    <img src="https://ars.els-cdn.com/content/image/1-s2.0-S1568494617305859-gr2.jpg" width="450" height="450" style="float:left">
    <img src="https://csdl-images.computer.org/trans/tp/1996/06/figures/i06483.gif"  width="400" height="400" style="float:right">
    <figcaption style="clear:both">Figura 2 - Exemplos de descritores (vetores de características) de imagens.</figcaption>
</center>

A Figura 2 mostra exemplos de vetores de características sobre imagens, onde cada pixel se torna um elemento do vetor. Portanto cada imagem será representado por seu vetor característico correspondente, o qual fará parte de um conjunto de vetores que serão utilizados no processo de classificação. Além desse tipo de vetor característica, os elementos desses vetores podem ser nominais ou reais, como: **cor, preço, ano, altura, peso, estado civil, etc**. Outra informação relevante é que cada vetor possui uma **classe** associada, o qual servirá de referência para classificar os dados. A Figura 3 mostra a forma geral de um conjunto de descritores e suas repectivas classes. 

<center>
    <img src="http://www.big-data.tips/wp-content/uploads/2016/08/textdata-features.jpg" width="500"/>
    <figcaption>Figura 3 - Conjunto de descritores genérico.</figcaption>
</center>

### 1.1 Obtenção dos dados

Neste problema será utilizado o *dataset* <a href="http://archive.ics.uci.edu/ml/datasets/Iris">Iris</a>, que consiste em um conjunto de dados que visa classificar os tipos de flores Ísis em **Setosa, Versicolour e Virginica**. Esse dataset é composto por 150 instâncias, sendo 50 para cada classe. Estes descritores são formados por 4 atributos: **tamanho da sépala, largura da sépala, tamanho da pétala** e **largura da pétala**.


```python
import pandas as pd

# Carregando o dataset
dataset = pd.read_csv("dataset/iris/dataset.csv", header=None)

class_column = len (dataset.columns) - 1

# Checando os dados
print (dataset)

# Descobrindo o número de instâncias por classes
for i in pd.unique(dataset[class_column]):
    print( i + ': ' + str(len (dataset.loc[dataset[class_column] == i])) )
```

           0    1    2    3               4
    0    5.1  3.5  1.4  0.2     Iris-setosa
    1    4.9  3.0  1.4  0.2     Iris-setosa
    2    4.7  3.2  1.3  0.2     Iris-setosa
    3    4.6  3.1  1.5  0.2     Iris-setosa
    4    5.0  3.6  1.4  0.2     Iris-setosa
    5    5.4  3.9  1.7  0.4     Iris-setosa
    6    4.6  3.4  1.4  0.3     Iris-setosa
    7    5.0  3.4  1.5  0.2     Iris-setosa
    8    4.4  2.9  1.4  0.2     Iris-setosa
    9    4.9  3.1  1.5  0.1     Iris-setosa
    10   5.4  3.7  1.5  0.2     Iris-setosa
    11   4.8  3.4  1.6  0.2     Iris-setosa
    12   4.8  3.0  1.4  0.1     Iris-setosa
    13   4.3  3.0  1.1  0.1     Iris-setosa
    14   5.8  4.0  1.2  0.2     Iris-setosa
    15   5.7  4.4  1.5  0.4     Iris-setosa
    16   5.4  3.9  1.3  0.4     Iris-setosa
    17   5.1  3.5  1.4  0.3     Iris-setosa
    18   5.7  3.8  1.7  0.3     Iris-setosa
    19   5.1  3.8  1.5  0.3     Iris-setosa
    20   5.4  3.4  1.7  0.2     Iris-setosa
    21   5.1  3.7  1.5  0.4     Iris-setosa
    22   4.6  3.6  1.0  0.2     Iris-setosa
    23   5.1  3.3  1.7  0.5     Iris-setosa
    24   4.8  3.4  1.9  0.2     Iris-setosa
    25   5.0  3.0  1.6  0.2     Iris-setosa
    26   5.0  3.4  1.6  0.4     Iris-setosa
    27   5.2  3.5  1.5  0.2     Iris-setosa
    28   5.2  3.4  1.4  0.2     Iris-setosa
    29   4.7  3.2  1.6  0.2     Iris-setosa
    ..   ...  ...  ...  ...             ...
    120  6.9  3.2  5.7  2.3  Iris-virginica
    121  5.6  2.8  4.9  2.0  Iris-virginica
    122  7.7  2.8  6.7  2.0  Iris-virginica
    123  6.3  2.7  4.9  1.8  Iris-virginica
    124  6.7  3.3  5.7  2.1  Iris-virginica
    125  7.2  3.2  6.0  1.8  Iris-virginica
    126  6.2  2.8  4.8  1.8  Iris-virginica
    127  6.1  3.0  4.9  1.8  Iris-virginica
    128  6.4  2.8  5.6  2.1  Iris-virginica
    129  7.2  3.0  5.8  1.6  Iris-virginica
    130  7.4  2.8  6.1  1.9  Iris-virginica
    131  7.9  3.8  6.4  2.0  Iris-virginica
    132  6.4  2.8  5.6  2.2  Iris-virginica
    133  6.3  2.8  5.1  1.5  Iris-virginica
    134  6.1  2.6  5.6  1.4  Iris-virginica
    135  7.7  3.0  6.1  2.3  Iris-virginica
    136  6.3  3.4  5.6  2.4  Iris-virginica
    137  6.4  3.1  5.5  1.8  Iris-virginica
    138  6.0  3.0  4.8  1.8  Iris-virginica
    139  6.9  3.1  5.4  2.1  Iris-virginica
    140  6.7  3.1  5.6  2.4  Iris-virginica
    141  6.9  3.1  5.1  2.3  Iris-virginica
    142  5.8  2.7  5.1  1.9  Iris-virginica
    143  6.8  3.2  5.9  2.3  Iris-virginica
    144  6.7  3.3  5.7  2.5  Iris-virginica
    145  6.7  3.0  5.2  2.3  Iris-virginica
    146  6.3  2.5  5.0  1.9  Iris-virginica
    147  6.5  3.0  5.2  2.0  Iris-virginica
    148  6.2  3.4  5.4  2.3  Iris-virginica
    149  5.9  3.0  5.1  1.8  Iris-virginica
    
    [150 rows x 5 columns]
    Iris-setosa: 50
    Iris-versicolor: 50
    Iris-virginica: 50


## 2. Princípio de funcionamento

A ideia básica do classificador **k-NN** está em medir a *distância* entre o indivíduo a ser classificado e os descritores, onde a classe atribuida a esse indivíduo será a mesma que a maioria dos **k** descritores mais próximos. Existem vários cálculos de distâncias que podem ser utilizados como métricas para encontrar os descritores mais próximos, sendo as mais conhecidas a *Distância Euclidiana* e a *Distância Manhattan*. 

<center>
    <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/795b967db2917cdde7c2da2d1ee327eb673276c0" width="450">
    <figcaption style="clear:both">Equação 1 - Fórmula da Distância Euclidiana.</figcaption>
    <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/02436c34fc9562eb170e2e2cfddbb3303075b28e"  width="400">
    <figcaption style="clear:both">Equação 2 - Fórmula da Distância Manhattan.</figcaption>
</center>

### 2.1 Criando as funções para cálculos de distâncias


```python
from math import sqrt

def euclidean(p, q):
    if len(p) != len (q):
        return -1
    
    local_sum = 0
    for i in range(0, len(p)):
        local_sum += pow(q[i] - p[i], 2)
    
    return sqrt (local_sum)

def manhattan(p, q):
    if len(p) != len (q):
        return -1
    
    local_sum = 0
    for i in range(0, len(p)):
        local_sum += abs(p[i] - q[i])
    
    return local_sum
```

## 3. Pré-processamento

O processo geral de implementação do **k-NN** segue as seguintes etapas:

<ul>
    <li>
        <strong>Etapa 1: </strong>
        Obter os dados, assim como verificar a precisão dos dados, realizar correções e remoção de dados desnecessários.
    </li>
    <li>
        <strong>Etapa 2: </strong>
        Separar o conjunto de dados em 2 conjuntos: <i>treino</i> e <i>testes</i>, sendo o primeiro composto por cerca de 60%-85% do total, e o segundo com o restante.
    </li>
    <li>
        <strong>Etapa 3: </strong>
        Realizar a classificação, seguindo o princípio de funcionamento descrito acima.
    </li>
</ul>


```python
import numpy as np

# Embaralhando os dados
dataset = dataset.iloc[np.random.permutation(len(dataset))]

# Separando o dataset por classes
setosa     = dataset.loc[dataset[class_column] == 'Iris-setosa']
versicolor = dataset.loc[dataset[class_column] == 'Iris-versicolor']
virginica  = dataset.loc[dataset[class_column] == 'Iris-virginica']
```


```python
train_percentage = 0.75

# Obtendo os conjuntos de treino e de testes
trainset = pd.concat([    setosa[0: int (len(setosa)     * train_percentage + 1)],
                      versicolor[0: int (len(versicolor) * train_percentage + 1)],
                       virginica[0: int (len(virginica)  * train_percentage + 1)]])

testset =  pd.concat([    setosa[int (len(setosa)     * train_percentage + 1) : ],
                      versicolor[int (len(versicolor) * train_percentage + 1) : ],
                       virginica[int (len(virginica)  * train_percentage + 1) : ]])

```


```python
from operator import itemgetter
from collections import Counter

# Criando a função de classificação
def knn(k, trainset, element):
    distance = []
    
    for _, row in trainset.iterrows():
        distance.append((manhattan(row[0:class_column], element[0:class_column]), row[class_column]))
    
    distance.sort(key=itemgetter(0))
    distance = [classes[1] for classes in distance[0:k]]
    
    most_common = Counter(distance)
    #print("Classification: " + max(most_common, key=most_common.get) + ", " + element[4])
    return max(most_common, key=most_common.get)

# Função de avaliação de acurácia
def evaluate(k, trainset, testset):
    acc = 0
    for _, row in testset.iterrows():
        if( knn(k, trainset, row) == row[class_column] ):
            acc += 1
    
    return acc / len(testset)

# Descobrindo a acurácia para todas as configurações possíveis
def evaluate_by_config(trainset, testset):
    for k in range(1,len(dataset) + 1):
        print("K = " + str(k) + ", acc = " + str(evaluate(k, trainset, testset)))
        
evaluate_by_config(trainset, testset)
```

    K = 1, acc = 0.9722222222222222
    K = 2, acc = 0.9722222222222222
    K = 3, acc = 1.0
    K = 4, acc = 0.9722222222222222
    K = 5, acc = 1.0
    K = 6, acc = 0.9722222222222222
    K = 7, acc = 0.9722222222222222
    K = 8, acc = 0.9722222222222222
    K = 9, acc = 1.0
    K = 10, acc = 0.9722222222222222
    K = 11, acc = 0.9722222222222222
    K = 12, acc = 0.9722222222222222
    K = 13, acc = 0.9722222222222222
    K = 14, acc = 0.9722222222222222
    K = 15, acc = 0.9722222222222222
    K = 16, acc = 0.9722222222222222
    K = 17, acc = 0.9722222222222222
    K = 18, acc = 0.9722222222222222
    K = 19, acc = 1.0
    K = 20, acc = 0.9722222222222222
    K = 21, acc = 0.9722222222222222
    K = 22, acc = 0.9722222222222222
    K = 23, acc = 0.9444444444444444
    K = 24, acc = 0.9722222222222222
    K = 25, acc = 0.9444444444444444
    K = 26, acc = 0.9444444444444444
    K = 27, acc = 0.9444444444444444
    K = 28, acc = 0.9444444444444444
    K = 29, acc = 0.9166666666666666
    K = 30, acc = 0.9166666666666666
    K = 31, acc = 0.9166666666666666
    K = 32, acc = 0.9444444444444444
    K = 33, acc = 0.9166666666666666
    K = 34, acc = 0.9722222222222222
    K = 35, acc = 0.9444444444444444
    K = 36, acc = 0.9722222222222222
    K = 37, acc = 0.9444444444444444
    K = 38, acc = 0.9444444444444444
    K = 39, acc = 0.9166666666666666
    K = 40, acc = 0.9444444444444444
    K = 41, acc = 0.9166666666666666
    K = 42, acc = 0.9166666666666666
    K = 43, acc = 0.9166666666666666
    K = 44, acc = 0.9166666666666666
    K = 45, acc = 0.9166666666666666
    K = 46, acc = 0.9166666666666666
    K = 47, acc = 0.9166666666666666
    K = 48, acc = 0.9166666666666666
    K = 49, acc = 0.8888888888888888
    K = 50, acc = 0.8888888888888888
    K = 51, acc = 0.8888888888888888
    K = 52, acc = 0.9166666666666666
    K = 53, acc = 0.8888888888888888
    K = 54, acc = 0.9166666666666666
    K = 55, acc = 0.9166666666666666
    K = 56, acc = 0.9166666666666666
    K = 57, acc = 0.8888888888888888
    K = 58, acc = 0.8888888888888888
    K = 59, acc = 0.8888888888888888
    K = 60, acc = 0.9166666666666666
    K = 61, acc = 0.8611111111111112
    K = 62, acc = 0.8611111111111112
    K = 63, acc = 0.8611111111111112
    K = 64, acc = 0.8611111111111112
    K = 65, acc = 0.8333333333333334
    K = 66, acc = 0.8333333333333334
    K = 67, acc = 0.8333333333333334
    K = 68, acc = 0.8611111111111112
    K = 69, acc = 0.8333333333333334
    K = 70, acc = 0.8333333333333334
    K = 71, acc = 0.8333333333333334
    K = 72, acc = 0.9166666666666666
    K = 73, acc = 0.8888888888888888
    K = 74, acc = 0.8888888888888888
    K = 75, acc = 0.8333333333333334
    K = 76, acc = 0.9444444444444444
    K = 77, acc = 0.9444444444444444
    K = 78, acc = 0.9444444444444444
    K = 79, acc = 0.9444444444444444
    K = 80, acc = 0.9444444444444444
    K = 81, acc = 0.9166666666666666
    K = 82, acc = 0.9444444444444444
    K = 83, acc = 0.9444444444444444
    K = 84, acc = 0.9444444444444444
    K = 85, acc = 0.9722222222222222
    K = 86, acc = 0.9444444444444444
    K = 87, acc = 0.9722222222222222
    K = 88, acc = 0.9722222222222222
    K = 89, acc = 0.9722222222222222
    K = 90, acc = 0.9722222222222222
    K = 91, acc = 0.9722222222222222
    K = 92, acc = 0.9722222222222222
    K = 93, acc = 0.9722222222222222
    K = 94, acc = 0.9722222222222222
    K = 95, acc = 0.9722222222222222
    K = 96, acc = 0.9722222222222222
    K = 97, acc = 0.9722222222222222
    K = 98, acc = 0.9722222222222222
    K = 99, acc = 0.9722222222222222
    K = 100, acc = 0.9722222222222222
    K = 101, acc = 0.9722222222222222
    K = 102, acc = 0.9722222222222222
    K = 103, acc = 0.9722222222222222
    K = 104, acc = 0.9722222222222222
    K = 105, acc = 0.9722222222222222
    K = 106, acc = 0.9722222222222222
    K = 107, acc = 0.9722222222222222
    K = 108, acc = 0.9722222222222222
    K = 109, acc = 0.9722222222222222
    K = 110, acc = 0.9722222222222222
    K = 111, acc = 0.9722222222222222
    K = 112, acc = 0.9722222222222222
    K = 113, acc = 0.9722222222222222
    K = 114, acc = 0.9722222222222222
    K = 115, acc = 0.9722222222222222
    K = 116, acc = 0.9722222222222222
    K = 117, acc = 0.9722222222222222
    K = 118, acc = 0.9722222222222222
    K = 119, acc = 0.9722222222222222
    K = 120, acc = 0.9722222222222222
    K = 121, acc = 0.9722222222222222
    K = 122, acc = 0.9722222222222222
    K = 123, acc = 0.9722222222222222
    K = 124, acc = 0.9722222222222222
    K = 125, acc = 0.9722222222222222
    K = 126, acc = 0.9722222222222222
    K = 127, acc = 0.9722222222222222
    K = 128, acc = 0.9722222222222222
    K = 129, acc = 0.9722222222222222
    K = 130, acc = 0.9722222222222222
    K = 131, acc = 0.9722222222222222
    K = 132, acc = 0.9722222222222222
    K = 133, acc = 0.9722222222222222
    K = 134, acc = 0.9722222222222222
    K = 135, acc = 0.9722222222222222
    K = 136, acc = 0.9722222222222222
    K = 137, acc = 0.9722222222222222
    K = 138, acc = 0.9722222222222222
    K = 139, acc = 0.9722222222222222
    K = 140, acc = 0.9722222222222222
    K = 141, acc = 0.9722222222222222
    K = 142, acc = 0.9722222222222222
    K = 143, acc = 0.9722222222222222
    K = 144, acc = 0.9722222222222222
    K = 145, acc = 0.9722222222222222
    K = 146, acc = 0.9722222222222222
    K = 147, acc = 0.9722222222222222
    K = 148, acc = 0.9722222222222222
    K = 149, acc = 0.9722222222222222
    K = 150, acc = 0.9722222222222222


## Referências

[1] Kevin Zakka. A Complete Guide to K-Nearest-Neighbors with Applications in Python and R, Available at: https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/ (Accessed: 28th March 2018).

[2] Maxwell. Aprendizado de máquina - conceitos básicos, Available at: https://www.maxwell.vrac.puc-rio.br/25796/25796_4.PDF (Accessed: 28th March 2018).

[3] Wikipedia (24th February 2018) k-nearest neighbors algorithm, Available at: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm (Accessed: 28th March 2018).
