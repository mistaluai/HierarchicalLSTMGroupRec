import matplotlib.pyplot as plt
import os

# Print where files will be saved
print("Files will be saved to:", os.getcwd())

# Data extracted from logs
epochs = list(range(1, 36))  # 1 to 35

train_loss = [1.889242, 1.468715, 1.181142, 0.999674, 0.874274, 0.848291, 0.769902,
              0.741221, 0.704062, 0.674053, 0.648516, 0.647605, 0.609759, 0.605793,
              0.601569, 0.582230, 0.599834, 0.571734, 0.552336, 0.558188, 0.568164,
              0.536477, 0.549512, 0.580998, 0.557979, 0.543040, 0.537784, 0.583701,
              0.546012, 0.549353, 0.528934, 0.528151, 0.552606, 0.559908, 0.552229]

val_loss = [3.126394, 4.405808, 1.280703, 1.362224, 0.974231, 1.779173, 0.825966,
            0.868298, 0.835031, 0.781170, 0.800147, 0.777892, 0.756460, 0.763473,
            0.770273, 0.780230, 0.769748, 0.786089, 0.784674, 0.794300, 0.796147,
            0.788295, 0.786109, 0.775828, 0.775446, 0.776597, 0.781497, 0.785076,
            0.769665, 0.780568, 0.783617, 0.779921, 0.777012, 0.770599, 0.781037]

val_accuracy = [0.208799, 0.317673, 0.524236, 0.506339, 0.616704, 0.428784, 0.671887,
                0.673378, 0.689038, 0.692767, 0.709172, 0.707681, 0.715884, 0.716629,
                0.718121, 0.716629, 0.721849, 0.720358, 0.721104, 0.721849, 0.724832,
                0.721849, 0.726324, 0.724832, 0.728561, 0.726324, 0.725578, 0.724832,
                0.729306, 0.722595, 0.726324, 0.727069, 0.727815, 0.724087, 0.722595]

# Create first plot: Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, pad=15)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Create second plot: Validation Accuracy
plt.figure(figsize=(12, 6))
plt.plot(epochs, val_accuracy, 'g-', label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Validation Accuracy', fontsize=14, pad=15)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots have been saved as 'loss_plot.png' and 'accuracy_plot.png'")