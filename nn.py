import math
import sys, os
import torch
import torch.nn as nn
from table import lookup_table, lookup

X = torch.tensor([[x] for x in lookup_table], dtype=torch.float32)
Y = torch.tensor([[lookup(x)] for x in lookup_table], dtype=torch.float32)
EPOCHS = 15

class LogNet(nn.Module):
    def __init__(self):
        super(LogNet, self).__init__()
        self.hidden = nn.Linear(1, 8)
        self.out = nn.Linear(8, 1)
        self.act = nn.Tanh()
    
    def forward(self, x):
        h = self.act(self.hidden(x))
        y = self.out(h)
        return y

model = LogNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

def train():
    for epoch in range(EPOCHS):
        mse_sum = 0.0
        for i in range(len(X)):
            optimizer.zero_grad()
            loss = criterion(model(X[i:i+1]), Y[i:i+1])
            loss.backward()
            optimizer.step()
            mse_sum += loss.item()

        mse = mse_sum / len(X)

        print(f"MSE: {mse:.20f}, Epoch: {epoch}")

def predict_log(x):
    inc = 0
    while x >= 10:
        x /= 10
        inc += 1

    with torch.no_grad():
        y = model(torch.tensor([[x]], dtype=torch.float32)).item()

    return y + inc

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python nn.py [number_to_predict]")
        sys.exit(1)

    if len(sys.argv) == 1:
        train()
        torch.save(model.state_dict(), "log_model.pth")
    elif len(sys.argv) == 2:
        if not os.path.exists("log_model.pth"):
            print("Model not trained yet. Run without arguments first.")
            sys.exit(1)

        model.load_state_dict(torch.load("log_model.pth"))
        model.eval()

        x = float(sys.argv[1])
        yp = predict_log(x)
        print(f"Predicted: {yp}")
        print(f"True:      {math.log10(x)}")
        print(f"Error:     {abs(yp - math.log10(x))}")

