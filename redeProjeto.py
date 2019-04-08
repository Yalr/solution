import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras
import time

inicio=time.time() # INICIA A CONTAGEM DO TEMPO

def n_to_str(a):
    letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']
    str_pdr = "Produto "
    b = []
    for i in range(a):
        b.append(str_pdr + letras[i])

    return b


def set_weigth_mes():
    weigth = []

    for i in range(12):
        if i == 0:
            weigth.append(np.random.uniform(4.2, 5.0))
        if i == 1:
            weigth.append(np.random.uniform(3.6, 4.4))
        if i == 2:
            weigth.append(np.random.uniform(2.6, 3.4))
        if i == 3:
            weigth.append(np.random.uniform(3.4, 4.2))
        if i == 4:
            weigth.append(np.random.uniform(2.2, 2.8))
        if i == 5:
            weigth.append(np.random.uniform(1.8, 2.6))
        if i == 6:
            weigth.append(np.random.uniform(2.6, 4.4))
        if i == 7:
            weigth.append(np.random.uniform(3.6, 3.4))
        if i == 8:
            weigth.append(np.random.uniform(2.0, 2.8))
        if i == 9:
            weigth.append(np.random.uniform(3.0, 3.8))
        if i == 10:
            weigth.append(np.random.uniform(3.6, 4.4))
        if i == 11:
            weigth.append(np.random.uniform(4.6, 5.2))

    return weigth


def set_qnt_UF_mes(n_anos):
    UF_mes = []
    messes = []
    weigth = []

    for i in range(26):
        weigth = set_weigth_mes()
        weight = np.divide(weigth, 2)
        for j in range(n_anos * 12):
            qnt = np.random.randint(15, 225)
            valor = float("{0:.2f}".format(qnt * weigth[j % 12]))
            messes.append(valor)
        UF_mes.insert(i, messes)
        messes = []

    # print(UF_mes, "\n\n\n")
    return UF_mes


def set_dataset(vetor_prod, n_anos):
    empresas = []
    produtos = []
    UF_mes = []

    n_empresas = len(vetor_prod)
    n_produtos = vetor_prod
    # n_empresas = 2
    # n_produtos = 3
    # n_anos = 1

    for i in range(n_empresas):
        for j in range(vetor_prod[i]):
            produtos.append(set_qnt_UF_mes(n_anos))
        empresas.append(produtos)
        produtos = []

    media = [0] * (n_anos * 12)
    legend = []

    arq1 = open('Mocks.txt', 'w')
    # print("Without Smooth Curve Grafics:")
    for i in range(len(empresas)):
        arq1.write('\nEmpresa {}\n'.format(i + 1))
        # print("\nEmpresa {}\n".format(i+1))
        for j in range(len(empresas[i])):
            arq1.write('\nProduto {}\n'.format(j + 1))
            # print("\nProduto {}\n".format(j+1))
            for k in range(len(empresas[i][j])):
                arq1.write('{}'.format(empresas[i][j][k]))
                # print(empresas[i][j][k])
                media = np.add(empresas[i][j][k], media)
            media = np.divide(media, 26)
            list_x = [c for c in range(len(media))]
            list_y = media
            #plt.plot(list_x, list_y)

        """
        plt.title("Amostra de Produtos ao Mês(Without Smooch Curve)")
        plt.xlabel("Messes")
        plt.ylabel("MPE(Média de Produtos de todos Estado)")
        plt.ylim(0, np.max(media) * 2)
        plt.legend(n_to_str(n_empresas))
        plt.show()

        """
    arq1.close()

    # arq2 = open('WithSmoochCurveGrafics.txt', 'w')
    # print("With Smooch Curve Grafics:")
    for i in range(len(empresas)):
        # arq2.write('\nEmpresa {}\n'.format(i + 1))
        # print("\nEmpresa {}\n".format(i + 1))
        for j in range(len(empresas[i])):
            # arq2.write('\nProduto {}\n'.format(j + 1))
            # print("\nProduto {}\n".format(j+1))
            for k in range(len(empresas[i][j])):
                # arq2.write('{}'.format(empresas[i][j][k]))
                # print(empresas[i][j][k])
                media = np.add(empresas[i][j][k], media)
            media = np.divide(media, 26)
            list_x = [c for c in range(len(media))]
            list_y = media
            tam = len(list_y)
            xnew = np.linspace(1, tam, num=150, endpoint=True)
            x = np.linspace(1, tam, tam)
            f2 = interp1d(x, list_y, kind='cubic')
            """
            plt.plot(xnew, f2(xnew))
            """

        """
        plt.legend(["Produto A", "Produto B", "Produto C"])
        plt.title("Amostra de Produtos ao Mês(With Smooch Curve)")
        plt.xlabel("Messes")
        plt.ylabel("MPE(Média de Produtos de todos Estado)")
        plt.ylim(0, np.max(media) * 2)
        plt.legend(n_to_str(n_empresas))
        plt.show()

        """

    # arq2.close()
    return np.array(empresas)


f=lambda A, target_empresa, target_produto, target_uf: np.array(A[target_empresa][target_produto][target_uf]) # Seleciona A empresa, o produto e o UF
g=lambda A, xmax, xmin:np.array([ (x-xmin)/(xmax-xmin)for x in A ]) # Normaliza os dados de f
h=lambda A, target:np.array([ A[i:i+target] for i in range(len(A)-target) ]) # Batch de entrada
i=lambda A, target:np.array([ (A[i+target]) for i in range(len(A)-target) ]) # Labels

def load_data(A, target, target_empresa, target_produto, target_uf):

    A=f(dataset, target_empresa, target_produto, target_uf) # SELECIONA EMPRESA, PRODUTO,UF
    A=g(A, A.max(), A.min()) # NORMALIZA
    return np.array(h(A,target)), np.array(i(A,target)) # RETORNA ENTRADA E LABEL

def train_net(data,label,dataset):
    keras.optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
    net=Sequential()
    net.add( LSTM( units=128,return_sequences=True, input_shape=(data.shape[1], 1)) )
    net.add (Dropout( 0.8 ) )
    net.add( LSTM( units=64,return_sequences=False) )
    net.add (Dropout( 0.8 ) )
    net.add(Dense(units=1, activation='relu'))
    net.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])
    net.fit(x=data,y=label,epochs=200)

    dataset_test = set_dataset(vetor_prod, n_anos)
    data_test, label_test=load_data(dataset_test,target,0,0,0)
    data_test=np.reshape(data_test,(data_test.shape[0], data_test.shape[1], 1))
    predict=net.predict(data_test, verbose=0)

    # PRINTANDO O GRÁFICO DA BASE
    x=f(dataset,0,0,1) # AQUI EU SELECIONO A BASE DE DADOS A SER PRINTADA, COR VERMELHA NO GRAFICO
    x=g(x, x.max(), x.min())
    x=np.array(x)

    plt.plot(x[11:20], c= "R") # PRINTA OS PRIMEIROS 12 MESES DE X
    # PRINTANDO O GRÁFICO DO PREDICT
    plt.plot(g(predict[1:11], predict.max(), predict.min())) # AQUI EU PRINT O PREDICT DOS PRIMEIROS 12 MESES, NORMALIZADO
    plt.show()

    """

    # AQUI PRINTO O GRÁFICO INVERTIDO
    plt.plot(x[0:12], c= "R")
    plt.plot(-g(predict[0:12], predict[0:12].max(), predict[0:12].min()))
    plt.show()
    #AQUI PRINTO O GRÁFICO INVERTIDO SOMADO COM -1
    plt.plot(x[0:12], c= "R")
    plt.plot(1-g(predict[0:12], predict[0:12].max(), predict[0:12].min()))
    plt.show()
    #AQUI PRINTO O GRÁFICO SOMADO COM 1
    plt.plot(x[0:12], c= "R")
    plt.plot(1+g(predict[0:12], predict[0:12].max(), predict[0:12].min()))
    plt.show()


    """

    # NOTE O GRÁFICO QUE MAIS SE ADEQUA À TENDÊNCIA.
        
    return predict

if __name__ == '__main__':
    vetor_prod = [1, 2, 3, 4]
    n_anos = 10
    target=10
    dataset = set_dataset(vetor_prod, n_anos)
    #print(dataset)
    """
    for i in range(len(dataset)):
        print("Empresa {}\n".format(i+1))
        for j in range(len(dataset[i])):
            print("Produto {}\n".format(j + 1))
            for k in range(len(dataset[i][j])):
                print(dataset[i][j][k])
    """
    data, label=load_data(dataset,target,0,0,0)
    pred=train_net(np.reshape(data,(data.shape[0], data.shape[1], 1)),label, dataset)
    fim = time.time()
    print(fim - inicio) # TERMINA DE CONTAR O TEMPO. NOTE QUE O TEMPO DE VISUALIZAÇÃO DO GRÁFICO É LEVADO EM CONTA.
