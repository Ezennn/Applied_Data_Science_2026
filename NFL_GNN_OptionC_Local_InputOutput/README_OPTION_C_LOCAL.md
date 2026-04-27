# NFL GNN Option C Local Runner

## 杩欑増鐨勭洰鐨?/ Purpose

杩欐槸涓€鐗堝彧淇濈暀 **鏂规 C / Option C** 鐨勬湰鍦?PyCharm 杩愯鍖呫€?
It keeps only the **Option C** workflow for local PyCharm execution.

瀹冩敮鎸佷綘鐨勫師濮嬫枃浠跺悕锛屼笉闇€瑕佹敼鍚嶏細

It supports your current raw filenames without renaming:

```text
input_2023_w01.csv
output_2023_w01.csv
```

涔熷吋瀹瑰師鏉ョ殑鏍囧噯鍛藉悕锛?
It also supports the standard names:

```text
train_input_2023_w01.csv
train_output_2023_w01.csv
```

---

## 1. 鏂囦欢澶圭粨鏋?/ Folder structure

鎶婇」鐩В鍘嬪悗锛屽缓璁繚鎸佽繖涓粨鏋勶細

After unzipping, keep this structure:

```text
NFL_GNN_OptionC_Local_InputOutput/
鈹?鈹溾攢 data/
鈹?  鈹溾攢 input_2023_w01.csv
鈹?  鈹溾攢 output_2023_w01.csv
鈹?  鈹溾攢 input_2023_w02.csv
鈹?  鈹溾攢 output_2023_w02.csv
鈹?  鈹斺攢 ...
鈹?鈹溾攢 nfl_trajectory_gnn_pipeline.py
鈹溾攢 export_preprocessed_weekly_csvs.py
鈹溾攢 train_from_preprocessed_weeks_local.py
鈹溾攢 train_recommended_holdout_split_local.py
鈹?鈹溾攢 check_torch_cuda_local.py
鈹溾攢 preprocess_weekly_tracking_csvs_local.py
鈹溾攢 train_recommended_holdout_split_entry_local.py
鈹溾攢 run_preprocess_and_train_local.py
鈹溾攢 requirements_local.txt
鈹斺攢 README_OPTION_C_LOCAL.md
```

閲嶈锛氬鏋滆瀹屾暣璺戞柟妗?C锛岄渶瑕?1鈥?8 鍛ㄧ殑 input 鍜?output锛屽洜涓烘ā鍨嬩細鐢?1鈥?3 鍛ㄨ缁冿紝14鈥?8 鍛ㄨ瘎浼般€?
Important: to run full Option C, you need input and output files for weeks 1鈥?8, because the model trains on weeks 1鈥?3 and evaluates on weeks 14鈥?8.

---

## 2. 瀹夎渚濊禆 / Install dependencies

鍦?PyCharm Terminal 閲岃繍琛岋細

Run in the PyCharm Terminal:

```bash
pip install -r requirements_local.txt
```

寤鸿鍦?PyCharm 閲屾妸瑙ｉ噴鍣ㄨ缃负椤圭洰鐨勮櫄鎷熺幆澧冿紙渚嬪 `./.venv`锛夛紝鍚﹀垯鍙兘浼氱敤鍒扮郴缁?Python/Anaconda锛屽鑷村寘瑁呴敊鐜銆?
In PyCharm, set the interpreter to the project virtualenv (e.g. `./.venv`), otherwise you may install packages into a different Python.

濡傛灉浣犵殑 PyTorch 涓嶆槸 GPU 鐗堬紝鍙互瀹夎 CUDA 鐗?PyTorch锛屼緥濡傦細

If your PyTorch is not GPU-enabled, install a CUDA build, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. 鎺ㄨ崘杩愯椤哄簭 / Recommended run order

### Step 1: 娴嬭瘯 GPU / Test GPU

杩愯锛?
Run:

```text
check_torch_cuda_local.py
```

濡傛灉鐪嬪埌锛?
If you see:

```text
cuda available: True
```

璇存槑鏈湴 GPU 鍙互琚?PyTorch 浣跨敤銆?
It means PyTorch can use your local GPU.

---

### Step 2: 棰勫鐞?/ Preprocess

杩愯锛?
Run:

```text
preprocess_weekly_tracking_csvs_local.py
```

瀹冧細浠?`data/` 璇诲彇锛?
It reads from `data/`:

```text
input_2023_wXX.csv + output_2023_wXX.csv
```

骞剁敓鎴愶細

and generates:

```text
preprocessed_csv_weekly/
鈹溾攢 preprocessed_train_input_2023_w01.csv
鈹溾攢 preprocessed_train_output_2023_w01.csv
鈹斺攢 ...
```

---

### Step 3: 杩愯鏂规 C / Run Option C

杩愯锛?
Run:

```text
train_recommended_holdout_split_entry_local.py
```

瀹冧細鎵ц锛?
It performs:

```text
train weeks: 1鈥?3
evaluate weeks: 14鈥?8
```

杈撳嚭鐩綍锛?
Output folders:

```text
artifacts_recommended_1_13_14_18/
holdout_predictions_recommended_14_18/
```

---

## 4. 涓€閿繍琛?/ One-click run

濡傛灉浣犳兂浠庨澶勭悊鍒拌缁冧竴娆℃€ц窇瀹岋紝鍙互鐩存帴杩愯锛?
If you want to run preprocessing and training in one go, run:

```text
run_preprocess_and_train_local.py
```

---

## 5. 鏈€閲嶈鐨勭粨鏋?/ Main result files

璁粌瀹屾垚鍚庨噸鐐圭湅锛?
After training, check:

```text
artifacts_recommended_1_13_14_18/training_history.csv
holdout_predictions_recommended_14_18/holdout_metrics.csv
```

鍏朵腑 `holdout_metrics.csv` 鏄?14鈥?8 鍛ㄨ瘎浼扮粨鏋溿€?
`holdout_metrics.csv` contains the week 14鈥?8 evaluation metrics.

---

## 6. 甯歌闂 / Common issues

### 鎵句笉鍒版暟鎹?/ Data not found

纭 `data/` 閲屾湁鎴愬鏂囦欢锛?
Make sure `data/` contains paired files:

```text
input_2023_w01.csv
output_2023_w01.csv
```

鎴栬€咃細

or:

```text
train_input_2023_w01.csv
train_output_2023_w01.csv
```

### 14鈥?8 鍛ㄨ瘎浼板け璐?/ Week 14鈥?8 evaluation fails

纭浣犳湁锛?
Make sure you have:

```text
input_2023_w14.csv ~ input_2023_w18.csv
output_2023_w14.csv ~ output_2023_w18.csv
```

鏂规 C 闇€瑕?14鈥?8 鍛ㄧ殑 truth output 鏉ョ畻 RMSE銆?
Option C needs week 14鈥?8 output truth files to compute RMSE.

### CUDA out of memory

鎶?`train_recommended_holdout_split_entry_local.py` 閲岀殑锛?
In `train_recommended_holdout_split_entry_local.py`, reduce:

```python
batch_size=4
```

鏀规垚锛?
Change to:

```python
batch_size=2
```

### Windows 澶氳繘绋?(DataLoader) 鎶ラ敊 / multiprocessing spawn error

鍦?Windows 涓婏紝濡傛灉 `num_workers > 0`锛孭ython 浼氱敤 `spawn` 鍚姩瀛愯繘绋嬶紱璁粌鑴氭湰闇€瑕佸姞锛?
```python
if __name__ == "__main__":
    main()
```

鏈粨搴撶殑 `train_recommended_holdout_split_entry_local.py` 鍜?`run_preprocess_and_train_local.py` 宸茬粡鍔犲ソ璇ヤ繚鎶ゃ€?
### `cuda available: False` / GPU not available

濡傛灉浣犵湅鍒帮細

```text
torch version: 2.x.x+cpu
cuda available: False
GPU not available
```

杩欓€氬父琛ㄧず浣犲畨瑁呯殑鏄?**CPU-only 鐨?PyTorch**锛堢増鏈彿甯?`+cpu`锛夛紝鎵€浠ュ嵆浣挎満鍣ㄤ笂鏈?NVIDIA 鏄惧崱锛孭yTorch 涔熺敤涓嶄簡 CUDA銆?
瑙ｅ喅鏂瑰紡锛?
1) 纭鏈哄櫒鏈?NVIDIA GPU锛屽苟涓斿凡瀹夎鏈€鏂版樉鍗￠┍鍔紙鍙湪缁堢杩愯 `nvidia-smi` 妫€鏌ワ級
2) 鍗歌浇 CPU 鐗堝苟瀹夎 CUDA 鐗?PyTorch锛屼緥濡傦細

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

鐒跺悗閲嶆柊杩愯 `check_torch_cuda_local.py` 楠岃瘉銆?
### 鏂版樉鍗?(sm_120) 鎶ラ敊 `no kernel image is available`

濡傛灉 `check_torch_cuda_local.py` 閲屾樉绀轰綘鐨?GPU 绫讳技锛?
```text
gpu capability: sm_120
```

骞跺嚭鐜帮細

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

璇存槑浣犺繖寮犳柊涓€浠?NVIDIA GPU 鐨勮绠楄兘鍔?(SM) **姣斿綋鍓嶅畨瑁呯殑 PyTorch wheel 鏀寔鑼冨洿鏇撮珮**锛坵heel 鍙甫浜嗗埌 `sm_90` 鐨?CUDA kernel锛夈€?
瑙ｅ喅鏂瑰紡锛氬畨瑁呮敮鎸佷綘 GPU 鐨勬洿鏂扮増 PyTorch锛堥€氬父鏄洿楂樼増鏈殑绋冲畾鐗堬紝鎴?nightly锛夈€傜ず渚嬶紙璇峰缁堢敤浣犻」鐩殑 `.venv`锛夛細

```bash
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

瀹夎鍚庨噸鏂拌繍琛?`check_torch_cuda_local.py`锛岀‘璁や笉鍐嶆姤 `no kernel image`銆?
