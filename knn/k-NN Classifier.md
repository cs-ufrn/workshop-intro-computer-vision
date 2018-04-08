
# k-Nearest Neighbors Classifier (k-NN)

O **k-NN** é o classificador mais simples na área de aprendizado de máquina. Diferentemente das redes neurais, não se realiza de fato um **aprendizado**, em vez disso, o algoritmo verifica a distância entre o objeto a ser classificado e os vetores de característica. Devido a sua simplicidade, é bastante utilizado em *benchmarks* de classificadores mais complexos como, Artificial Neural Network (**ANN**) e Suport Vector Machine (**SVM**).

<center>
    <figure>
        <img src="https://amueller.github.io/applied_ml_spring_2017/images/classifier_comparison.png" alt="Classifier Comparison">
        <figcaption>Figura 1 - Comparação entre os classificadores (https://amueller.github.io/applied_ml_spring_2017/images/classifier_comparison.png).</figcaption>
    </figure>
</center>

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
# Importações das bibliotecas
import pandas as pd
from math import sqrt,floor
import numpy as np
from operator import itemgetter
from collections import Counter
```

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

## 3. Processamento

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

### 3.1 Obtenção dos dados


```python
# Carregando o dataset
dataset = pd.read_csv("dataset/iris/dataset.csv", header=None)

# Índice das classes
class_column = len (dataset.columns) - 1

# Checando os dados
print (dataset)

# Lista com os nomes das classes
class_names = pd.unique(dataset[class_column])

# Descobrindo o número de instâncias por classes; e
# Definindo número mínimo de classes
min_classes = len(dataset)
for i in class_names:
    print( str(i) + ': ' + str(len (dataset.loc[dataset[class_column] == i])) )
    min_classes = min(min_classes, len (dataset.loc[dataset[class_column] == i]))
```

### 3.2 Separação dos conjuntos


```python
train_percentage = 0.7

# Obtendo os conjuntos de treino e de testes

trainset = dataset.loc[dataset[class_column] == class_names[0]][0:floor(train_percentage * min_classes)]
testset  = dataset.loc[dataset[class_column] == class_names[0]][floor(train_percentage * min_classes):]

for i in range(1,len(class_names)):
    trainset = pd.concat([trainset, dataset.loc[dataset[class_column] == class_names[i]][0:floor(train_percentage * min_classes)]])
    testset  = pd.concat([testset,  dataset.loc[dataset[class_column] == class_names[i]][floor(train_percentage * min_classes):]])

print("Tamanho trainset: " + str(len(trainset)))
print("Tamanho testset: " + str(len(testset)))
```

### 3.3 Classificação


```python
# Criando a função de classificação
def knn(k, train, element,function):
    distance = []
    
    local_class_column = len (train.columns) - 1
    
    for _, row in train.iterrows():
        distance.append((function(row[0:local_class_column], element[0:local_class_column]), row[local_class_column]))
    
    distance = sorted(distance)
    distance = [classes[1] for classes in distance[0:k]]
    
    most_common = Counter(distance)
    return max(most_common, key=most_common.get)

# Função de avaliação de acurácia
def evaluate(k, train, test, function):
    acc = 0
    
    local_class_column = len (train.columns) - 1
    
    for _, row in test.iterrows():
        if( knn(k, train, row, function) == row[local_class_column] ):
            acc += 1
    
    return acc / len(test)

# Descobrindo a acurácia para todas as configurações possíveis
def evaluate_by_config(train, test, function):
    for k in range(1, min_classes + 1):
        print("K = " + str(k) + ", acc = " + str(evaluate(k, train, test, function)))
```


```python
# Checando a melhor configuração
evaluate_by_config(trainset, testset,euclidean)
```

## 4 Reduzindo os dados e mantendo acurácia


```python
# Criando uma cópia dos dados
cp_dataset = dataset.copy()

# Alterando a label da classe para um número
index = 1
for i in class_names:
    cp_dataset.loc[cp_dataset[class_column] == i, 4] = index
    index += 1
    
# Novo dataset
print(cp_dataset)

# Correlação entre os dados
print("\nCorrelação: ")
print(cp_dataset.corr())

# Covariância entre os dados
print("\nCovariância: ")
print(cp_dataset.cov())
```


```python
# Novos datasets de treino e teste com apenas os descritores 2 e 3, além da classe
new_trainset = trainset.iloc[:,2:]
new_testset  = testset.iloc[:,2:]

# Ajustando os índices das colunas
new_trainset.columns = range(new_trainset.shape[1])
new_testset.columns = range(new_testset.shape[1])

# Checando a acurácia
evaluate_by_config(new_trainset, new_testset, euclidean)
```

## Referências

[1] Kevin Zakka. A Complete Guide to K-Nearest-Neighbors with Applications in Python and R, Available at: https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/ (Accessed: 28th March 2018).

[2] Maxwell. Aprendizado de máquina - conceitos básicos, Available at: https://www.maxwell.vrac.puc-rio.br/25796/25796_4.PDF (Accessed: 28th March 2018).

[3] Wikipedia (24th February 2018) k-nearest neighbors algorithm, Available at: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm (Accessed: 28th March 2018).
