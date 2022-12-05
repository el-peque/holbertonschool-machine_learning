#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

columns = ('Farrah', 'Fred', 'Felicia')
rows = ('apples', 'bananas', 'oranges', 'peaches')
colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
y_offset = np.zeros(len(columns))
index = np.arange(len(columns)) + 0.3

for row in range(len(rows)):
    plt.bar(columns, fruit[row], width=0.5,
            bottom=y_offset, color=colors[row], label=rows[row])
    y_offset = y_offset + fruit[row]

plt.ylim(0, 80)
plt.yticks(np.arange(0, 90, 10))

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.legend()

plt.show()
