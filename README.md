# TFM deliverable
Este código implementa una Red Neuronal Cuántica (QNN) diseñada para simular sistemas cuánticos y realizar tareas de aprendizaje automático cuántico. La QNN está compuesta por múltiples capas de puertas cuánticas personalizadas, que aplican operaciones específicas al estado cuántico de un oscilador armónico. La red se entrena utilizando el optimizador L-BFGS.

## Estructura del proyecto
```
├── network.py                # Implementación principal de la QNN y la puerta personalizada
├── evaluation.ipynb          # Cuaderno de Jupyter para la generación de datos, entrenamiento y visualización
├── requirements.txt          # Dependencias de Python
├── results.pkl               # Resultados de entrenamiento
├── images/                   # Gráficas de resultados de entrenamiento
    ├── eval-curve.jpg        # Curva de referencia a aproximar
    ├── losses.jpg            # Curvas de entrenamiento
    └── results.jpg           # Curva de referencia y curvas aproximadas
└── README.md                 # Este archivo README
```

## Uso
1. Instala las dependencias del proyecto (preferiblemente en un entorno virtual): `pip install -r requirements.txt`

2. Ejecuta `evaluation.ipynb` para entrenar redes con distinto número de puertas y visualizar los resultados.