import numpy as np
import matplotlib.pyplot as plt

# Replace these lists with your actual MSE values for each model
mse_4lp = [0.0125, 0.0001, 0.0004, 1.45e-5 , 0.0012]
mse_lstm = [0.0017592615, 1.36419485e-05, 0.0037221021, 3.5156765e-05, 1.6001328e-05]
mse_transformer = [0.0007946620, 5.346842e-05, 5.024851e-05, 7.484535e-05, 0.0011014686]

# Create separate histograms for each model
plt.figure(figsize=(10, 6))

#plt.hist(mse_4lp, bins=15, alpha=0.9, label='4LP')
#plt.hist(mse_lstm, bins=15, alpha=0.9, label='LSTM')
#plt.hist(mse_transformer, bins=15, alpha=0.9, label='Transformer')

#plt.xlabel('Mean Squared Error (MSE)')
#plt.ylabel('Frequency')
#plt.title('Histogram of MSE Values for Different Models')
#plt.legend()
#plt.grid(True)

mean_mse_4lp = mse_4lp[0]
mean_mse_lstm = mse_lstm[0]
mean_mse_transformer = mse_transformer[0]

models = ['4LP', 'LSTM', 'Transformer']
mean_mse_values = [mse_4lp, mse_lstm, mse_transformer]

plt.bar(models, mse_4lp, density=True, color=['blue', 'black', 'yellow'])

plt.xlabel('Model Type')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Mean MSE Values for Different Models')
plt.grid(True)

plt.show()