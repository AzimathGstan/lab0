# PyTorch 101: MNIST Digit Recognizer / 手寫數字辨識入門

Welcome to the introductory PyTorch lab! In this session, we will build, train, and test a Convolutional Neural Network (CNN) to recognize handwritten digits. This repository is specifically optimized for CPU training, meaning it will run smoothly in your WSL (Ubuntu) environment without needing dedicated GPU drivers.

歡迎來到 PyTorch 入門實驗課！在本節課中，我們將建立、訓練並測試一個卷積神經網路（CNN）來辨識手寫數字。此儲存庫專為 CPU 訓練進行了最佳化，這意味著它可以在您的 WSL (Ubuntu) 環境中順暢執行，無需安裝複雜的 GPU 驅動程式。

---

## 🛠️ Step 1: Environment Setup / 步驟一：環境設定

First, open your WSL Ubuntu terminal and clone this repository.  
首先，打開您的 WSL Ubuntu 終端機並複製此儲存庫：

```bash
git clone https://github.com/AzimathGstan/lab0.git
cd lab0
```

To keep our dependencies from clashing with your system Python, create and activate a virtual environment.  
為了避免套件與系統的 Python 環境發生衝突，請建立並啟動虛擬環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required CPU-optimized PyTorch libraries. We use a custom index to ensure we don't download massive, unnecessary GPU binaries.  
安裝所需的 CPU 最佳化 PyTorch 套件。我們使用自訂的下載來源，以確保不會下載到龐大且不必要的 GPU 檔案：

```bash
pip install -r requirements.txt
```

---

## 🧠 Step 2: Train the Model / 步驟二：訓練模型

Time to train the network! The dataset is already bundled in the `data/` folder, so this will run completely offline.  
開始訓練網路！資料集已經內建在 `data/` 資料夾中，因此這段程式碼可以完全離線執行。

```bash
python3 train.py
```

Watch the progress bar! It will train for 3 epochs. Once finished, it will save your newly trained network weights to `weights/mnist_model.pth`.  
請觀察進度條！模型將訓練 3 個 Epoch。完成後，最新訓練的網路權重會自動儲存到 `weights/mnist_model.pth`。

> **Failsafe / 備用方案:** > If your computer crashes or the script fails, don't worry. A pre-trained backup model is already included in the `weights/` folder, so you can still complete the rest of the lab!  
> 如果您的電腦當機或程式執行失敗，請別擔心。`weights/` 資料夾中已經為您準備好了一個預先訓練好的備用模型，您仍然可以繼續完成後續的實驗！

---

## 🔍 Step 3: Terminal Inference / 步驟三：終端機推論 (Inference)

Let's test the model on a single, random image from the test dataset. We will print the image directly into your terminal using ASCII art and see what the model predicts.  
讓我們從測試集中隨機抽取一張圖片來測試模型。我們將使用 ASCII 藝術（字元畫）將圖片直接印在終端機上，並查看模型的預測結果。

```bash
python3 inference.py
```

Run this a few times to see how the model handles different handwritten digits.  
您可以多執行幾次，看看模型如何處理不同的手寫數字。

---

## 🏆 Step 4: The Benchmark Challenge / 步驟四：基準測試與挑戰

Testing one image is fun, but how accurate is your model overall? Run the benchmark script to test your network against all 10,000 images in the testing set:  
測試單張圖片很有趣，但您的模型整體準確率有多高呢？執行基準測試程式，用測試集中的 10,000 張圖片來評估您的網路：

```bash
python3 benchmark.py
```

**Your Challenge / 你的挑戰：** Open `model.py` and modify the neural network architecture. Try adding another convolutional layer (`nn.Conv2d`), increasing the number of channels, or changing the size of the linear layers.   
打開 `model.py` 並修改神經網路架構。試著加入更多的卷積層（`nn.Conv2d`）、增加通道數，或是改變線性層（Linear layers）的大小。

1. Edit `model.py` / 編輯 `model.py`。
2. Run `python3 train.py` to retrain your new architecture / 執行 `python3 train.py` 重新訓練新架構。
3. Run `python3 benchmark.py` to see your new score / 執行 `python3 benchmark.py` 查看新分數。

**Can you beat a 98.5% accuracy? / 你能突破 98.5% 的準確率嗎？**

---

## 🎨 Bonus: Interactive Drawing Canvas / 額外挑戰：互動式畫布

Want to test the model against your own handwriting in real-time?   
想試試看讓模型即時辨識您親手寫的數字嗎？

First, you need to install the full version of OpenCV to handle the graphical window:  
首先，您需要安裝完整版的 OpenCV 來處理圖形視窗：

```bash
pip install opencv-python
```

*(Note for WSL Users: If the window fails to open, your WSL environment might be missing rendering libraries. Run `sudo apt-get update && sudo apt-get install libgl1-mesa-glx libglib2.0-0` in your terminal to fix this).* *（WSL 使用者注意：如果視窗無法開啟，您的 WSL 環境可能缺少渲染圖形的函式庫。請在終端機執行 `sudo apt-get update && sudo apt-get install libgl1-mesa-glx libglib2.0-0` 來修復此問題）。*

Once installed, launch the interactive canvas:  
安裝完成後，啟動互動式畫布：

```bash
python3 interactive.py
```

* **Draw** a number (0-9) with your mouse. / 用滑鼠**畫出**一個數字 (0-9)。
* **Spacebar:** Force the model to predict your drawing. / 按下 **空白鍵** 讓模型進行預測。
* **C:** Clear the canvas. / 按下 **C** 清除畫布。
* **Q:** Quit. / 按下 **Q** 退出程式。
