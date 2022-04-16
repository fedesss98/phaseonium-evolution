import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0, 1, 100)
b = a * np.random.rand(100)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    plt.plot(np.arange(100), a, color='orange')
    plt.plot(np.arange(100), b)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
