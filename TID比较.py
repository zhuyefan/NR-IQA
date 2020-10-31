import yaml
from matplotlib import pyplot as plt
import xlrd
import pandas as pd
import numpy as np
np.random.rand
def main():

    data = pd.read_excel("data/TID2013_11score.xlsx")

    x = data["无失真"]
    l = len(x)
    y = data["I级score"]
    z = data["II级score"]
    zz = data["III级score"]
    zzz = data["IV级score"]
    plt.plot(range(0,l),x,label="score(0)",color="r")
    plt.plot(range(0, l), y, label="score(I)", color="orange")
    plt.plot(range(0, l), z, label="score(II)", color="y")
    # plt.plot(range(0,l),y,label="salt540p(0.2)",color="g")
    plt.plot(range(0, l), zz, label="score(III)", color="k")
    plt.plot(range(0, l), zzz, label="score(IV)", color="b")
    # plt.plot(range(0, l), y, label="salt540p(0.7)", color="b")
    plt.legend(loc="upper right")
    plt.title("style 11")
    # xx=np.arange(70)
    # yy=np.arange(70)
    # plt.plot(xx,yy)
    # a = np.random.rand(10)
    # print(a)
    plt.show()


if __name__ == '__main__':
    main()