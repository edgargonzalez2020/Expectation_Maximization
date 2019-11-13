import numpy as np
import sys
import matplotlib.pyplot as plt
def main():
    data = np.loadtxt(sys.argv[1])
    points = data[:,: -int(sys.argv[2])]
    probabilities = data[:,[2,3]]
    print(probabilities)
    colors=["#0000FF", "#00FF00", "#FF0066"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,x in enumerate(points):
        print(int(np.argmax(probabilities[i])))
        ax.scatter(x[0],x[1], color=colors[int(np.argmax(probabilities[i]))])
    plt.show()
if __name__ == '__main__':
    main()
