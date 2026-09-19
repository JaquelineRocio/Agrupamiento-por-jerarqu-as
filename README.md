# Actividad 08 — Clustering jerárquico

**Unidad II · Machine Learning I**  
**Autora:** Jaqueline Rocio Ramos Vargas · Trabajo individual

## Resumen ejecutivo

Se aplica agrupamiento aglomerativo a documentos, clientes simulados y fotografías. Se comparan linkages manteniendo las mismas variables, distancias y muestras dentro de cada ejercicio.

| Ejercicio | Estado de esta ejecución | Evidencia |
|---|---|---|
| Texto — 6 puntos | Ejecutado | Notebook, métricas y términos por grupo cuando se descarga el corpus |
| Clientes — 6 puntos | Ejecutado | 28 combinaciones de cuatro linkages y k=2…8; perfiles |
| Imágenes — 6 puntos | Ejecutado | 18 combinaciones: dos fotografías, tres linkages, tres k |
| Resumen — 2 puntos | Este README | Código en notebook, tablas, hallazgos y referencias |


### Resultados principales de esta ejecución

- **Texto:** average, ARI=0.1327; correspondencia limitada con las 20 categorías. Ward obtiene mayor V-measure en la ejecución de referencia.
- **Clientes:** average, k=5, silhouette=0.6736; ventaja muy pequeña sobre la segunda combinación.
- **Imágenes:** china.jpg: a k=12, ward tiene el menor MSE RGB (0.00762) entre las tres alternativas. flower.jpg: a k=12, ward tiene el menor MSE RGB (0.00474) entre las tres alternativas. El criterio es fidelidad del color, no exactitud semántica.

## Cómo ejecutar y entregar

1. Abrir `Actividad_08_Clustering_Jerarquico.ipynb` en Google Colab mediante **Archivo → Subir notebook**.
2. Usar CPU, conexión a internet y **Entorno de ejecución → Ejecutar todas**. No requiere GPU ni cuenta Kaggle.
3. Comprobar que la última verificación indique los tres ejercicios ejecutados.
4. Descargar `Actividad_08_resultados.zip` desde el panel de archivos de Colab y guardar el notebook con sus salidas mediante **Archivo → Descargar → Descargar .ipynb**.
5. Para GitHub, subir el notebook, este README y la carpeta `resultados/` juntos; así se muestran las figuras. Esta entrega no crea un repositorio automáticamente.

También se incluye `solucion_actividad08.py`, con las mismas celdas de código, para ejecución local. Instalar `requirements.txt` y ejecutar `python solucion_actividad08.py` desde la carpeta del proyecto. Las versiones efectivas se guardan en `resultados/entorno.json`. El ZIP que genera el notebook contiene resultados y README; guardar el notebook por separado.

## Diseño y criterios de éxito

El clustering es aglomerativo: fusiona grupos hasta construir una jerarquía. Single usa la menor distancia entre puntos; complete la mayor; average el promedio; Ward minimiza el incremento de dispersión interna y requiere distancia euclídea. Las alturas de distintos linkages no son directamente intercambiables.

- **Texto:** mayor ARI a k=20; V-measure como desempate. Evaluación exploratoria sobre la misma muestra, sin afirmaciones sobre generalización.
- **Clientes:** mayor silhouette; desempate por menor Davies–Bouldin y menor k. ARI sintético solo como evaluación posterior.
- **Imágenes:** menor MSE a igual k para fidelidad RGB, más inspección de fronteras; no equivale a mejor segmentación semántica.

ARI=1 significa coincidencia perfecta de particiones; cerca de 0 es el nivel esperado por azar. V-measure y silhouette se maximizan; Davies–Bouldin se minimiza. Los números de cluster son identificadores arbitrarios.

## 1. Texto — 20 Newsgroups

Se solicitan 100 documentos de cada una de las 20 categorías del subconjunto train. Se eliminan cabeceras, pies, citas, textos muy cortos y duplicados exactos normalizados. Se aplica TF–IDF (hasta 15 000 términos), SVD de 100 componentes y normalización L2. Los cuatro linkages reciben la misma matriz y distancia euclídea.

Las etiquetas se usan para estratificar y evaluar; no se incluyen como características. Fijar k=20 y seleccionar por ARI usa conocimiento de las categorías, por lo que no constituye una selección totalmente no supervisada. La muestra reduce el costo cuadrático; no representa una ejecución sobre los 18 846 documentos completos.

| linkage | k | ARI | V_measure | silhouette | davies_bouldin | min_n | max_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| average | 20 | 0.1327 | 0.3196 | 0.0131 | 3.9803 | 8 | 687 |
| ward | 20 | 0.0974 | 0.3911 | 0.0198 | 3.8967 | 16 | 600 |
| complete | 20 | 0.0773 | 0.1894 | 0.0063 | 5.7594 | 36 | 208 |
| single | 20 | 0.0000 | 0.0204 | -0.0988 | 1.0861 | 1 | 1979 |

### Hallazgos y mejor linkage

- Con 2000 documentos y k=20, average obtuvo el mayor ARI (0.1327) entre los cuatro linkages.
- La representación SVD retuvo 17.0% de la varianza de TF–IDF; sus 100 componentes no conservan toda la información.
- El grupo más grande del ganador concentra 34.4% de los documentos; el más pequeño contiene 8.
- ward: ARI=0.0974, silhouette=0.0198, grupo mayor=30.0%.
- complete: ARI=0.0773, silhouette=0.0063, grupo mayor=10.4%.
- average: ARI=0.1327, silhouette=0.0131, grupo mayor=34.4%.
- single: ARI=0.0000, silhouette=-0.0988, grupo mayor=99.0%.
- Grupo 12: 32 documentos; sci.med (18), misc.forsale (2); términos: edu, gordon, chastity, dsl, geb, shameful, intellect, cadre, skepticism, surrender.
- Grupo 15: 8 documentos; comp.os.ms-windows.misc (4), sci.crypt (1); términos: ax, max, pl, en, ml, bhj, wm, bh, mq, gk.
- Grupo 3: 175 documentos; rec.sport.hockey (81), rec.sport.baseball (77); términos: team, year, game, players, league, think, season, hockey, play, games.
- En deportes, el grupo 3 contiene 158 documentos de las dos categorías relacionadas, dentro de 175 documentos totales. Esto muestra temas amplios, no una separación perfecta de cada foro.
- En hardware, el grupo 6 contiene 140 documentos de las dos categorías relacionadas, dentro de 231 documentos totales. Esto muestra temas amplios, no una separación perfecta de cada foro.
- En sistemas y ventanas, el grupo 8 contiene 103 documentos de las dos categorías relacionadas, dentro de 171 documentos totales. Esto muestra temas amplios, no una separación perfecta de cada foro.
- Las métricas no coinciden: ward lidera V-measure (0.3911); se mantiene average como ganador según el ARI definido de antemano. El ARI bajo del ganador indica correspondencia limitada, no recuperación satisfactoria de las 20 categorías.
- Single concentra 99.0% en un grupo; su Davies–Bouldin=1.0861 no debe interpretarse aisladamente como superioridad temática.
- En esta ejecución aparecen grupos con vocabulario de firmas o residuos, como gordon/chastity/geb y ax/max. La limpieza de sklearn es heurística: una categoría dominante no garantiza un tema interpretable. Revisar estos casos sería un siguiente paso, sin cambiar retrospectivamente los resultados.
- Las categorías de los foros no equivalen necesariamente a temas separables: vocabulario compartido, mensajes con varios asuntos y reducción dimensional pueden mezclar categorías. Estas son explicaciones plausibles, no causas demostradas por las métricas.

### Temas encontrados

| cluster | n | categorias_principales | proporcion_dominante | terminos |
| --- | --- | --- | --- | --- |
| 0 | 89 | comp.graphics (16), comp.sys.ibm.pc.hardware (9) | 0.1798 | thanks, mail, know, advance, hi, appreciated, info, read, don, greatly |
| 1 | 687 | talk.politics.mideast (85), soc.religion.christian (78) | 0.1237 | people, don, just, think, god, does, know, like, government, say |
| 2 | 192 | rec.motorcycles (50), rec.autos (50) | 0.2604 | car, bike, space, cars, like, engine, long, just, don, time |
| 3 | 175 | rec.sport.hockey (81), rec.sport.baseball (77) | 0.4629 | team, year, game, players, league, think, season, hockey, play, games |
| 4 | 126 | misc.forsale (54), sci.electronics (19) | 0.4286 | new, price, sale, condition, shipping, offer, book, phone, box, asking |
| 5 | 44 | comp.graphics (13), sci.med (6) | 0.2955 | use, graphics, package, processing, deskjet, post, image, looking, standard, radiologist |
| 6 | 231 | comp.sys.ibm.pc.hardware (71), comp.sys.mac.hardware (69) | 0.3074 | card, drive, mac, video, thanks, scsi, use, monitor, speed, software |
| 7 | 23 | alt.atheism (5), sci.space (2) | 0.2174 | really, organization, laws, hear, state, groups, idea, say, gain, election |
| 8 | 171 | comp.windows.x (64), comp.os.ms-windows.misc (39) | 0.3743 | windows, file, program, using, files, window, dos, problem, version, thanks |
| 9 | 39 | alt.atheism (7), talk.politics.guns (5) | 0.1795 | com, list, mail, mailing, internet, dave, article, dtmedin, catbyte, ingr |
| 10 | 30 | alt.atheism (5), sci.crypt (4) | 0.1667 | post, questions, tell, sorry, don, ll, posts, topic, flame, computer |
| 11 | 11 | sci.space (3), misc.forsale (2) | 0.2727 | need, chicago, russians, vesa, resources, mode, navigation, possible, send, demonstrating |
| 12 | 32 | sci.med (18), misc.forsale (2) | 0.5625 | edu, gordon, chastity, dsl, geb, shameful, intellect, cadre, skepticism, surrender |
| 13 | 13 | sci.crypt (3), misc.forsale (3) | 0.2308 | chip, intel, ticket, amd, duke, clipper, timer, chance, intergraph, thier |
| 14 | 37 | talk.politics.misc (5), rec.autos (4) | 0.1351 | ago, years, didn, said, did, article, memory, know, weeks, chain |
| 15 | 8 | comp.os.ms-windows.misc (4), sci.crypt (1) | 0.5000 | ax, max, pl, en, ml, bhj, wm, bh, mq, gk |
| 16 | 23 | soc.religion.christian (4), sci.electronics (4) | 0.1739 | good, point, deleted, pc, better, doing, let, speculation, south, protection |
| 17 | 33 | talk.religion.misc (9), rec.sport.hockey (4) | 0.2727 | heard, amorc, kent, zionism, oto, cheers, order, know, interested, claim |
| 18 | 11 | alt.atheism (3), sci.space (2) | 0.2727 | guess, greek, body, maybe, like, ripped, deleted, false, damage, stuff |
| 19 | 25 | rec.motorcycles (4), rec.sport.hockey (3) | 0.1600 | ve, seen, got, stacks, machines, happen, think, laser, guy, talent |

![Jerarquías de texto](resultados/figuras/texto_dendrogramas.png)

![Categorías y clusters](resultados/figuras/texto_contingencia.png)

## 2. Segmentación de clientes

Se generan 500 clientes con cinco centros mediante make_blobs (semilla 42), usando ingreso anual e índice de gasto. Se estandarizan las variables. Los datos son sintéticos y su estructura compacta favorece ciertos linkages; no se presentan como datos de Mall Customers ni evidencia de eficacia comercial.

### Mejor k por linkage según silhouette

| linkage | k | silhouette | davies_bouldin | ARI | min_n | max_n |
| --- | --- | --- | --- | --- | --- | --- |
| average | 5 | 0.6736 | 0.4557 | 0.9657 | 95 | 103 |
| ward | 5 | 0.6735 | 0.4541 | 0.9705 | 94 | 104 |
| complete | 5 | 0.6731 | 0.4632 | 0.9753 | 97 | 104 |
| single | 3 | 0.4868 | 0.6269 | 0.4807 | 100 | 300 |

### Comparación a igual k seleccionado

| linkage | k | silhouette | davies_bouldin | ARI |
| --- | --- | --- | --- | --- |
| ward | 5 | 0.6735 | 0.4541 | 0.9705 |
| complete | 5 | 0.6731 | 0.4632 | 0.9753 |
| average | 5 | 0.6736 | 0.4557 | 0.9657 |
| single | 5 | 0.1975 | 0.6973 | 0.4786 |

### Hallazgos

- average con k=5 obtiene silhouette=0.6736 y Davies–Bouldin=0.4557.
- El ARI frente a los grupos generadores es 0.9657; se usa para comprobar recuperación, no para seleccionar el modelo.
- Los perfiles corresponden a las medias observadas. Sus nombres y acciones son interpretaciones posteriores al clustering.
- La simulación favorece grupos compactos por construcción. Un buen resultado no demuestra eficacia comercial en clientes reales.
- Antes de aplicar estas acciones se necesitan datos reales, validación del negocio y seguimiento de resultados.
- La diferencia de silhouette entre la mejor combinación y ward con k=5 es apenas 0.000108; es una ventaja numérica pequeña, sin evidencia de superioridad estable.

### Perfiles y acciones hipotéticas

| cluster | n | ingreso_medio | gasto_medio | perfil | accion_hipotetica |
| --- | --- | --- | --- | --- | --- |
| 0 | 100 | 25.77 | 80.23 | Ingreso bajo, gasto alto | Evaluar recompensas accesibles; no inferir capacidad crediticia. |
| 1 | 103 | 93.38 | 26.02 | Ingreso alto, gasto bajo | Investigar barreras de compra antes de ofrecer promociones. |
| 2 | 102 | 95.59 | 80.53 | Ingreso alto, gasto alto | Evaluar beneficios de fidelización y atención preferente. |
| 3 | 100 | 24.19 | 25.24 | Ingreso bajo, gasto bajo | Probar ofertas de entrada y medir su respuesta. |
| 4 | 95 | 59.12 | 49.27 | Ingreso medio, gasto medio | Probar recomendaciones y promociones moderadas. |

Los umbrales para nombrar los perfiles son didácticos: ingreso bajo <40, alto >80; gasto bajo <40, alto >65. No intervienen en el ajuste del clustering.

![Selección de k](resultados/figuras/clientes_seleccion_k.png)

![Comparación de clientes](resultados/figuras/clientes_comparacion.png)

![Jerarquías de clientes](resultados/figuras/clientes_dendrogramas.png)

## 3. Segmentación de imágenes

Se segmentan china.jpg y flower.jpg a 96×64 píxeles. Características: RGB en 0–1 y coordenadas normalizadas con peso 0.25. Conectividad de cuatro vecinos. Se comparan Ward, complete y average a 6, 12 y 24 regiones.

### Comparación a 12 regiones

| imagen | linkage | k | mse_rgb | silhouette_muestra | region_min_px | region_max_px |
| --- | --- | --- | --- | --- | --- | --- |
| china.jpg | ward | 12 | 0.00762 | 0.23912 | 129 | 1650 |
| china.jpg | average | 12 | 0.02473 | 0.08433 | 1 | 3152 |
| china.jpg | complete | 12 | 0.03307 | -0.09266 | 2 | 3129 |
| flower.jpg | ward | 12 | 0.00474 | 0.23566 | 16 | 1098 |
| flower.jpg | complete | 12 | 0.01010 | 0.39842 | 1 | 4654 |
| flower.jpg | average | 12 | 0.01024 | 0.41518 | 1 | 4657 |

### Hallazgos

- china.jpg: a k=12, ward tiene el menor MSE RGB (0.00762) entre las tres alternativas.
- flower.jpg: a k=12, ward tiene el menor MSE RGB (0.00474) entre las tres alternativas.
- china.jpg: la mayor silhouette muestreada corresponde a ward (0.2391); su región mayor abarca 26.9%. Debe interpretarse junto con tamaños y fronteras, especialmente cuando existen regiones de uno o dos píxeles.
- flower.jpg: la mayor silhouette muestreada corresponde a average (0.4152); su región mayor abarca 75.8%. Debe interpretarse junto con tamaños y fronteras, especialmente cuando existen regiones de uno o dos píxeles.
- Inspección de china.jpg a k=12: Ward conserva más diferencias de color entre cielo, agua, edificio y vegetación; average fusiona amplias zonas oscuras del edificio y el follaje.
- Inspección de flower.jpg a k=12: los tres métodos delimitan la flor principal. Ward subdivide pétalos y fondo por sus tonalidades; average conserva una silueta más simple. Más detalle cromático no implica una mejor máscara del objeto completo.
- Todas las regiones de todas las configuraciones resultaron conectadas; la conectividad de cuatro vecinos evitó grupos de píxeles aislados entre sí.
- Un menor MSE indica mayor fidelidad de color, pero no demuestra que una región corresponda a una flor, un edificio o un objeto completo.
- La reducción de resolución elimina detalles finos; el peso espacial 0.25 y los valores de k son decisiones didácticas fijas, no parámetros optimizados con máscaras humanas.
- Sin anotaciones de referencia no se reportan ARI ni IoU de segmentación semántica.

La silhouette usa la misma muestra de 1 200 píxeles para todas las configuraciones; puede omitir regiones muy pequeñas. `regiones_en_muestra` permite detectarlo. Las fronteras y la conectividad se verifican sobre los 6 144 píxeles completos.

![Segmentación china](resultados/figuras/imagen_china_comparacion.png)

![Detalle por k china](resultados/figuras/imagen_china_detalle_k.png)

![Segmentación flower](resultados/figuras/imagen_flower_comparacion.png)

![Detalle por k flower](resultados/figuras/imagen_flower_detalle_k.png)

## Conclusiones y limitaciones

El linkage cambia las fusiones y, por tanto, los tamaños y la composición de los grupos. No hay un criterio universalmente superior: la comparación debe vincularse a la representación, la distancia y el objetivo de cada ejercicio. La jerarquía permite inspeccionar varios niveles de detalle.

El éxito de los clientes simulados no implica éxito en documentos o fotografías. En texto se necesita coherencia temática; en clientes, utilidad y estabilidad de los perfiles; en imágenes, continuidad y correspondencia visual. Una métrica geométrica favorable no garantiza estas propiedades por sí sola.

Se utiliza una sola semilla y no se estima estabilidad entre muestras. No hay evaluación predictiva en un conjunto independiente; AgglomerativeClustering no proporciona directamente `predict` para observaciones nuevas. La palabra «mejor» se limita a las alternativas evaluadas y al criterio declarado.

## Archivos de resultados

- `texto_metricas.csv`, `texto_perfiles.csv`, `texto_contingencia.csv`, `texto_asignaciones.csv` y `texto_auditoria.json`: se generan cuando se ejecuta el corpus real.
- `clientes_metricas.csv`, `clientes_sinteticos.csv`, `clientes_perfiles.csv`.
- `imagenes_metricas.csv` y mapas `*_etiquetas.npy` (cada entero identifica una región).
- `figuras/`: dendrogramas, perfiles visuales y segmentaciones.
- `entorno.json`: versiones y semilla.

## Referencias

- Scikit-learn. [Clustering jerárquico y criterios de enlace](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering).
- Scikit-learn. [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).
- Scikit-learn. [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html).
- Scikit-learn. [TF–IDF y LSA para clustering de documentos](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html).
- Scikit-learn. [make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).
- Scikit-learn. [load_sample_image](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html).
- Scikit-learn. [Segmentación de imágenes con Ward](https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html).
- Scikit-learn. [grid_to_graph](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.grid_to_graph.html).
- [Créditos originales de las fotografías](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/datasets/images). `china.jpg`: danielbuechele; `flower.jpg`: vultilion. CC BY 2.0. Se muestran versiones reducidas y segmentadas.
