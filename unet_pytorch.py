import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

########################################################################################################################
# Contraction path: conv, dropout, max pool, and weight initialisation.
########################################################################################################################
conv1 = nn.Conv2d(in_channels=IMG_CHANNELS, out_channels=16, kernel_size=(3, 3), padding=1)
conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
conv9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv3.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv4.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv5.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv6.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv7.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv8.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv9.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv10.weight, mode='fan_out', nonlinearity='relu')
########################################################################################################################
# Expansion path: skipped connections, conv, dropout, max pool, and weight initialisation.
########################################################################################################################
conv_t1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=0)
conv11 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1)
conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
conv_t2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=0)
conv13 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1)
conv14 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
conv_t3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2), padding=0)
conv15 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1)
conv16 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
conv_t4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(2, 2), padding=0)
conv17 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1)
conv18 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
conv_final = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1))
nn.init.kaiming_normal_(conv_t1.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv11.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv12.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv_t2.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv13.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv14.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv_t3.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv15.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv16.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv_t4.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv17.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv18.weight, mode='fan_out', nonlinearity='relu')
nn.init.kaiming_normal_(conv_final.weight, mode='fan_out', nonlinearity='sigmoid')

dropout1 = nn.Dropout(0.1)
dropout2 = nn.Dropout(0.2)
dropout3 = nn.Dropout(0.3)
max_pool = nn.MaxPool2d(kernel_size=(2, 2))


########################################################################################################################
# Forward pass.
########################################################################################################################
def forward(x):
    # First layer (contraction).
    c1 = F.relu(conv1(x))
    c1 = dropout1(c1)
    c1 = F.relu(conv2(c1))
    p1 = max_pool(c1)
    # Second layer (contraction).
    c2 = F.relu(conv3(p1))
    c2 = dropout1(c2)
    c2 = F.relu(conv4(c2))
    p2 = max_pool(c2)
    # Third layer (contraction).
    c3 = F.relu(conv5(p2))
    c3 = dropout2(c3)
    c3 = F.relu(conv6(c3))
    p3 = max_pool(c3)
    # Fourth layer (contraction).
    c4 = F.relu(conv7(p3))
    c4 = dropout2(c4)
    c4 = F.relu(conv8(c4))
    p4 = max_pool(c4)
    # Fifth layer (contraction).
    c5 = F.relu(conv9(p4))
    c5 = dropout3(c5)
    c5 = F.relu(conv10(c5))
    # Sixth layer (expansion).
    u6 = conv_t1(c5)
    u6 = t.cat([u6, c4], dim=1)
    c6 = F.relu(conv11(u6))
    c6 = dropout2(c6)
    c6 = F.relu(conv12(c6))
    # Seventh layer (expansion).
    u7 = conv_t2(c6)
    u7 = t.cat([u7, c3], dim=1)
    c7 = F.relu(conv13(u7))
    c7 = dropout2(c7)
    c7 = F.relu(conv14(c7))
    # Eighth layer (expansion).
    u8 = conv_t3(c7)
    u8 = t.cat([u8, c2], dim=1)
    c8 = F.relu(conv15(u8))
    c8 = dropout1(c8)
    c8 = F.relu(conv16(c8))
    # Nine layer (expansion).
    u9 = conv_t4(c8)
    u9 = t.cat([u9, c1], dim=1)
    c9 = F.relu(conv17(u9))
    c9 = dropout1(c9)
    c9 = F.relu(conv18(c9))
    # Output layer.
    outputs = t.sigmoid(conv_final(c9))
    return outputs


########################################################################################################################
# Build the model
########################################################################################################################
inputs = t.randn(1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

outputs = forward(inputs)

params = [param for layer in [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10,
                              conv_t1, conv_t2, conv_t3, conv_t4, conv11, conv12, conv13, conv14,
                              conv15, conv16, conv17, conv18, conv_final] for param in layer.parameters()]
optimizer = optim.Adam(params, lr=0.001)
criterion = nn.BCELoss()
# Print the model summary
print("Model Summary:")
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {outputs.shape}")
