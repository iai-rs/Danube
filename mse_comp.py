import matplotlib.pyplot as plt
import numpy as np

species = ("Bromacil", "2.4D", "F", "2.4D&F", "Bentazone")
penguin_means = {
    '4LP': (0.0125, 0.0001, 0.0004, 1.45E-05, 0.0012),
    'LSTM': (0.0017592615,	1.36E-05,	0.0037221021,	3.52E-05,	1.60E-05),
    'Transformer': (0.000794662,	5.35E-05,	5.02E-05,	7.48E-05,	0.0011014686),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE')
ax.set_title('ANN MSE accordung to substance')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
#ax.set_ylim(0, 250)

plt.show()
