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
