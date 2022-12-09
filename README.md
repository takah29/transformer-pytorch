# Transformer

## Features

- Implemented with only basic PyTorch module
- Beam Search Decoding

## Requirements

- Python 3.10
- PyTorch 1.13
- Torchtext 0.14
- spaCy 3.4

## Environment setting

```bash
pip install pipenv  # If pipenv is not installed
cd <repository root path>
pipenv sync
pipenv shell
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Usage

### Translation from German to English

```bash
# Activate virtual environment
pipenv shell

# Download Multi30k dataset. then create dataset, word frequency files and a parameter setting file.
python build_multi30k_dadaset.py

# Training model on Multi30k dataset and save the model in the models directory
python train_translation_model.py multi30k_dataset

# Input sample source texts with a redirection
python translate.py multi30k_dataset < multi30k_dataset/src_val_texts.txt

# output
# text: Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen
#    -> A group of men are cleaning up a truck .
# text: Ein Mann schläft in einem grünen Raum auf einem Sofa .
#    -> A man sleeping on a couch in a green room on a couch .
# text: Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau .
#    -> A boy with headphones sitting on his shoulders on a woman 's shoulders .
# text: Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf
#    -> Two men are building a blue inflatable raft up a lake on a lake .
# text: Ein Mann mit beginnender Glatze , der eine rote Rettungsweste trägt , sitzt in einem kleinen Boot .
#    -> A balding man wearing a red life jacket sitting on a small boat .
# text: Eine Frau in einem rotem Mantel , die eine vermutlich aus Asien stammende Handtasche in einem blauen Farbton hält , springt für einen Schnappschuss in die Luft .
#    -> A woman in a red coat holding an Indian purse in an auditorium .
# text: Ein brauner Hund rennt dem schwarzen Hund hinterher .
#    -> A brown dog runs toward the black dog after a black dog .
# text: Ein kleiner Junge mit einem Giants-Trikot schwingt einen Baseballschläger in Richtung eines ankommenden Balls .
#    -> A little boy wearing a tracksuit is swinging a baseball bat toward a ball .
# text: Ein Mann telefoniert in einem unaufgeräumten Büro
#    -> A man in a messy office is talking on the phone .
# text: Eine lächelnde Frau mit einem pfirsichfarbenen Trägershirt hält ein Mountainbike
#    -> A smiling woman with a briefcase holding a mountain bag .
# ...
```

### Translation from English to Japanese

```bash
# Activate virtual environment
pipenv shell

# Download kftt dataset. then create dataset, word frequency files and a parameter setting file.
python build_kftt_dadaset.py

# Training model on kftt dataset and save the model in the models directory
python train_translation_model.py kftt_dataset

# Input sample source texts with a redirection
python translate.py kftt_dataset < kftt_dataset/src_test_texts.txt

# output
# text: dogen was a zen monk in the early kamakura period .
#    -> 道元 ( <unk> ) は 鎌倉 時代 初期 の 禅僧 。
# text: the founder of soto zen
#    -> 曹洞 禅宗 の 祖 。
# text: later in his life he also went by the name kigen .
#    -> 晩年 は 名 の 由来 と し て も 知 ら れ た 。
# text: within the sect he is referred to by the honorary title koso .
#    -> 宗派 内 で は 名誉 教授 と 称 さ れ る 。
# text: posthumously named bussho dento kokushi , or joyo-daishi .
#    -> 死後 、 正一 位 、 城陽 菩薩 、 城陽 、 城陽 - - - 弘法 大師 。
# text: he is generally called dogen zenji .
#    -> 一般 に 道元 と 呼 ば れ る 。
# text: he is reputed to have been the one that spread the practices of tooth brushing , face washing , table manners and cleaning in japan .
#    -> 日本 に お い て
# text: another story has it that he was the first one to bring moso-chiku ( moso bamboo ) to japan .
#    -> また 、 一 つ の <unk> を <unk> し た と い う 。
# text: though some points are unclear about dogen 's birth , all accounts agree that he was born in the line of udaijin ( minister of the right ) michichika tsuchimikado ( minamoto no michichika or michichika koga ) .
#    -> 道元 の 生年 に つ い て は 不明 で あ る 。
# text: although it is generally accepted that he was born in shoden sanso in kohata , kyoto , to michichika and fujiwara no ishi , the daughter of daijo-daijin ( grand minister of state ) motofusa matsudono ( fujiwara no motofusa ) , recent research suggests that he may have been the son of michitomo horikawa , who was presumed to be his adoptive father .
#    -> 一般 的 に は 京都 所司 代 で あ っ た が 、 松殿 基房 の 娘 で あ る 。
# ...
```