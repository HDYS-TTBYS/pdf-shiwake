# pdf-shiwake

## 使い方

1.exeをダウンロード

<https://github.com/HDYS-TTBYS/pdf-shiwake/releases/download/1.0.0/pdf-shiwake.exe>

2.config.example.yamlを参考にconfig.yamlを作成する

3.pdf-shiwake.exeをダブルクリック

## ビルド(単一exeファイルにする)

1.tesseractを"C:\Program Files\"にインストールする

- 日本語トレーニングデータも取得する

2.popperを"C:\Program Files\"にインストールする

3.パッケージインストール

```bash
pip install -r requirements.txt
```

4.exe化

```bash
pyinstaller pdf-shiwake.spec
```

## python実行方法

1.tesseractを"C:\Program Files\"にインストールする

- 日本語トレーニングデータも取得する
- PATHを通す

2.C:\Program Files\popper\bin配下のファイルを".\popper"へ置く

3.実行

```bash
python main.py
```
