# Automatic Contouring of Primary Gross Tumor Volume for Nasopharynx Cancer by Deep Learning
Here are the implementation resources for auto-contouring GTVp for NPC patients. <br>

fcn.py is the training file, containing the network architecture.
dicomSubject.py is for data loadining.
three_channel_dicomSubject.py and three_channel_fcn.py are for three-modality data training.

A local training job can be run with the command:
```bash
python fcn.py
```
