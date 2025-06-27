import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name = input('Masukkan nama file (contoh: dataset.csv): ')
dataset = pd.read_csv('dataset/' + file_name)
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values
n = len(dataset)

x_sum = np.sum(X)
x_mean = np.mean(X)
x2_sum = np.sum(X ** 2)
x_sum2 = x_sum ** 2

y_sum = np.sum(Y)
y_mean = np.mean(Y)
xy_sum = np.sum(X * Y)

def y_predict(x_predict):
    b = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum2)
    a = y_mean - b * x_mean
    return a + b * x_predict

def plot_data():
    plt.scatter(X, Y, color='red')
    plt.plot(X, 
             [y_predict(x) for x in X], 
             color='blue')
    plt.title(dataset.keys()[0] + ' vs ' + dataset.keys()[1])
    plt.xlabel(dataset.keys()[0])
    plt.ylabel(dataset.keys()[1])
    plt.show()

confirm = input('Tampilkan grafik? (y/n): ')
if confirm.lower() == 'y':
    plot_data()

result = input('Masukkan nilai untuk diprediksi: ')
try:
    result = float(result)
    print(f'Nilai prediksi: {y_predict(result)}')
except ValueError:
    print('Masukkan nilai angka.')