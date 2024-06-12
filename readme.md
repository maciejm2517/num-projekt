1. Dodaj odpowiednie pliki do folderu data oraz zapisz je w dvc
```
dvc add data/PlantVillage
dvc add data/PotatoPlants
```
2. W pierwszym terminalu uruchom mlops 
```
mlflow ui
```
3. W drugim terminalu wytrenuj model np.
```
python train.py --learning_rate 0.01 --epochs 10 --batch_size 64
```
4. Dodaj go do dvc
```
dvc add models/my_model.h5
```
5. W trzecim terminalu uruchom streamlit
```
streamlit run streamlit.py
```
