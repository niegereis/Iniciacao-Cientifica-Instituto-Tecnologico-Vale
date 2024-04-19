import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix


def treinar_modelo(nome_planilha,qtd_amostras):
    dados = pd.read_excel(nome_planilha)
    X = dados.drop('Movimento ', axis=1) 
    y = dados['Movimento ']

    tamanho_teste = qtd_amostras

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=tamanho_teste, shuffle=False)

    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)


    modelo = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_treino.shape[1],)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.005)
    modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    modelo.fit(X_treino, y_treino, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    precisao = modelo.evaluate(X_teste, y_teste)[1]
    print(f'A precisão do modelo é: {precisao}')

    probabilidades = modelo.predict(X_teste)
    y_pred = (probabilidades > 0.5).astype("int32")

    print(f'Acurácia: {accuracy_score(y_teste, y_pred)}')
    # print('Matriz de Confusão:')
    # print(confusion_matrix(y_teste, y_pred))

    previsoes = modelo.predict(X_teste)

    for i in range(len(previsoes)):
        if previsoes[i] > 0.5:
            if y_teste.iloc[i] == 1:
                print(f"Amostra {i + 1}: Correto! Em movimento!")
            else:
                print(f"Amostra {i + 1}: Incorreto! Parado!")
        else:
            if y_teste.iloc[i] == 0:
                print(f"Amostra {i + 1}: Correto! Parado!")
            else:
                print(f"Amostra {i + 1}: Incorreto! Em movimento!")


if __name__ == '__main__':
    nome_planilha = input("Digite a planilha onde estão os dados para treinamento: ")
    entrada_qtd_amostras = input("Digite a quantidade de amostras necessárias da planilha: ")
    qtd_amostras = int(entrada_qtd_amostras)
    treinar_modelo(nome_planilha,qtd_amostras)