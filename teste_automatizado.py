from rede_neural import treinar_modelo 
import os

def test_treinar_modelo():
    x = os.listdir(".")
    # print(x)

    for i in range(len(x)):
        if "xlsx" in x[i]:
            print(x[i], int(x[i][12:].split('.')[0]) - 1)
            treinar_modelo(x[i],int(x[i][12:].split('.')[0]) - 1)
            print("\n")

test_treinar_modelo()