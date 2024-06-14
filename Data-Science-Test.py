import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, 0:5]
y = iris.target
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target_names = ['setosa', 'versicolor', 'virginica']
X = pd.DataFrame(X, columns=feature_names)

# utilizamos o head para exibir as primeiras linhas do dataframe (por padrão 5).
X.head(2)
# Retorna uma amostra (aleatória) de n elementos, no nosso exemplo 4 elementos.
X.sample(4)
# Retorna uma descrição do data set.
X.describe()

———————

media = X['sepal length (cm)'].mean() # Média da coluna
print("Média ", media)

———————

mediana = X['sepal length (cm)'].median() # Mediana da coluna
print("Mediana ", mediana)

———————

moda = X['sepal length (cm)'].mode() # Moda da coluna
print("Moda ", moda[0])

———————

variancia = X['sepal length (cm)'].var() # Variância da coluna
print("Variância ", variancia)

———————

desvio = X['sepal length (cm)'].std() # Desvio padrão da coluna
print("Desvio padrão ", desvio)

———————

Q1 = X['sepal length (cm)'].quantile(0.25)
Q2 = X['sepal length (cm)'].quantile(0.5)
Q3 = X['sepal length (cm)'].quantile(0.75)
print('Primeiro quartil ', Q1)
print('Segundo quartil (Mediana)', Q2)
print('Terceiro quartil ', Q3)

———————

Q1 = X['sepal length (cm)'].quantile(0.25)
Q2 = X['sepal length (cm)'].quantile(0.5)
Q3 = X['sepal length (cm)'].quantile(0.75)
IQR = Q3 - Q1
print('Intervalo interquartil ', IQR)

———————

sns.boxplot(X['sepal length (cm)'])

———————

sns.displot(X['sepal length (cm)'])


———————

# Import those libraries
import pandas as pd
from scipy.stats import pearsonr

# Import your data into Python
df = pd.read_csv("PABD-02a.csv")

# Convert dataframe into series
list1 = df['NF']
list2 = df['Freq']

print(df)

# Apply the pearsonr()
corr, _ = pearsonr(list1, list2)
print('Pearsons correlation: %.3f' % corr)

———————

# importar pacotes necessários
import numpy as np
import matplotlib.pyplot as plt

# exemplo de plots determinísticos
np.random.seed(42)
det_x = np.arange(0,10,0.1)
det_y = 2 * det_x + 3

# exemplo de plots não determinísticos
non_det_x = np.arange(0, 10, 0.1)
non_det_y = 2 * non_det_x + np.random.normal(size=100)

# plotar determinísticos vs. não determinísticos
fig, axs = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)

axs[0].scatter(det_x, det_y, s=2)
axs[0].set_title("Determinístico")
axs[1].scatter(non_det_x, non_det_y, s=2)
axs[1].set_title("Não Determinístico")

plt.show()

———————

# importar os pacotes necessários
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# criar modelo linear e otimizar
lm_model = LinearRegression()
lm_model.fit(non_det_x.reshape(-1,1), non_det_y)

# extrair coeficientes, beta0=intercept e beta1=coef
slope = lm_model.coef_
intercept = lm_model.intercept_

# imprimir os valores encontrados para os parâmetros
print("b0: \t{}".format(intercept))
print("b1: \t{}".format(slope[0]))

# plotar pontos e retas com parâmetros otimizados
plt.scatter(non_det_x, non_det_y, s=3)
plt.plot(non_det_x, (non_det_x * slope + intercept), color='r')

plt.show()

——————

# Importando Bibliotecas:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)
# Carregando a base de dados
df = pd.read_csv('PABD-05a.csv')
# Número de linhas e colunas:
df.shape
(200, 5)
# Estatística Descritiva:
df.describe()
# Tipos de Dados:
df.dtypes
# Definindo um estilo para os gráficos:
plt.style.use('fivethirtyeight')
# Verificando os dados
df.head()

———————

# Verificando as distribuição dos dados:
plt.figure(1 , figsize = (15 , 6))
n = 0
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 25)
    plt.title('{} '.format(x))
plt.show()
# Contagem de Amostras por Sexo:
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Genre' , data = df)
plt.show()

———————

# Idade vs Renda Anual:
plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Age',
        y = 'Annual Income (k$)',
        data = df[df['Genre'] == gender],
        s = 200 ,
        alpha = 0.5 , label = gender)
plt.xlabel('Age'),
plt.ylabel('Annual Income (k$)')
plt.title('Idade vs Renda Anual')
plt.legend()
plt.show()
# Renda Anual vs Pontuação de Gastos:
plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Annual Income (k$)',
        y = 'Spending Score (1-100)' ,
        data = df[df['Genre'] == gender],
        s = 200,
        alpha = 0.5 , label = gender)
plt.xlabel('Annual Income (k$)'),
plt.ylabel('Spending Score (1-100)')
plt.title('Renda Anual vs Pontuação de Gastos')
plt.legend()
plt.show()

———————

# Distribuição de Idade, Renda Anual e Pontuação de Gastos segmentado por Sexo:
plt.figure(1 , figsize = (15 , 7))
n = 0
for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'Genre' , data = df , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'Genre' , data = df)
    plt.ylabel('Genre' if n == 1 else '')
    plt.title('Idade, Renda Anual e Pontuação de Gastos por Sexo' if n == 2 else '')
plt.show()

———————

# Selecionando o número de clusters através do método Elbow (Soma das distâncias quadráticas intra clusters):
X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n))
    algorithm.fit(X2)
    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Número de Clusters') , plt.ylabel('Soma das Distâncias Q intra Clusters')
plt.show()

———————

# Inicializando e Computando o KMeans com o valor de 4 clusters:
algorithm = (KMeans(n_clusters = 4))
algorithm.fit(X2)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10,
    random_state=None, tol=0.0001, verbose=0)
# Visualizando os grupos criados e seus centroides:
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z.reshape(xx.shape)
plt.imshow(
    Z2 , interpolation='nearest', extent=(xx.min(),
    xx.max(), yy.min(), yy.max()), cmap = plt.cm.Pastel2,
    aspect = 'auto', origin='lower')
plt.scatter( x = 'Annual Income (k$)', y = 'Spending Score (1-100)',
    data = df, c = labels2, s = 200 )
plt.scatter(x = centroids2[: , 0],
    y = centroids2[: , 1], s = 300, c = 'red', alpha = 0.5)
plt.ylabel('Pontuação de Gastos (1-100)') , plt.xlabel('Renda Anual (k$)')
plt.show()
