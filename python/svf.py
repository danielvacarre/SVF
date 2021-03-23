from itertools import zip_longest, product
from numpy import arange
from docplex.mp.model import Model
from sklearn.model_selection import KFold
from pandas import DataFrame, concat

##SUSTITUYE A TRANSFORMACIÓN
def transformation(x_i, t_k):
    """Función que evalúa si el valor de una celda es mayor o menor al de un nodo del grid.
       Si es mayor devuelve 1, si es igual devuelve 0 y si es menor devuelve -1.

    Parameters
    ----------
    x_i : float
        Valor de la celda a evaluar
    t_k : float
        Valor del nodo con el que se quiere comparar

    Returns
    -------
    res : int
        Resultado de la transformación
    """

    z = x_i - t_k
    if z < 0:
        return -1
    elif z == 0:
        return 0
    else:
        return 1

##SUSTITUYE CALCULATE_POS_PHI
def locate_position_observation(x, t):
    """Función que localiza la posición de una observación en el grid.
       Evalúa los valores de x en cada dimensión con los valores de las particiones t en cada dimension y devuelve la posición en el grid.

    Parameters
    ----------
    x : list
        Lista de coordenadas x de la observación a evaluar
    t : list
        Lista de particiones en cada dimensión

    Returns
    -------
    p : list
        Posición de la observación en el grid
    """

    p = []
    # transpuesta de t para calcular el vector de posiciones de las observaciones
    r = list(zip_longest(*t))
    for l in range(0, len(t)):
        for m in range(0, len(t[l])):
            trans = transformation(x[l], r[m][l])
            if trans < 0:
                p.append(m - 1)
                break
            if trans == 0:
                p.append(m)
                break
            if trans > 0 and m == len(t[l]) - 1:
                p.append(m)
                break
    return p

##SUSTITUYE CALCULATE_VALUE_PHI
def calculate_transformation_observation(vector_subind, p):
    """Función que calcula el valor de la transformación de una observación en función de su posición en el grid.
       El resultado de la transformación será una lista de 1 y 0.
       Si la celda en la que se encuentra la observación "domina" a la celda a evaluar el valor es 1, mientras que en el caso contrario es 0.

    Parameters
    ----------
    vector_subind : list
        Listado de posiciones de las celdas del grid
    p : List
        Posición de la observación en el grid

    Returns
    -------
    phi : list
        Resultado de la transformación
    """

    phi = []
    n_dim = len(p)
    for i in range(0, len(vector_subind)):
        for j in range(0, n_dim):
            if p[j] >= vector_subind[i][j]:
                r = 1
            else:
                r = 0
                break
        phi.append(r)
    return phi

#SUSTITUYE A CALCULATE_MATRIX_PHI
def calculate_matriz_transformations(X, t, vector_subind):
    """Función que crea la matriz de transformación de cada observación de la muestra de aprendizaje en base al grid escogido.
       Primero se ubica en qué celda del grid está la observación--> locate_position_observation(x, t, r).
       Después se calcula el valor de la transformación (phi) en base a su posición--> calculate_transformation_observation(vector_subind, p)

    Parameters
    ----------
    X : dataFrame
        Valores de los inputs
    t : list
        Lista de particiones en cada dimensión
    vector_subind: list
        Lista con la combinación de todos los índices de todos las particiones en el grid.
    Returns
    -------
    M : list
        Matriz de transformación de todas las observaciones
    """

    x_list = X.values.tolist()
    M = []
    for x in x_list:
        p = locate_position_observation(x, t)
        phi = calculate_transformation_observation(vector_subind, p)
        M.append(phi)
    return M

##SUSTITUYE A CREATE_MATRIX_T_EQUI
def create_matrix_partitions(X, d):
    """Función que crea las particiones de cada dimensión. Coge cada dimensión de inputs y la trocea en d particiones equidistantes

    Parameters
    ----------
    X : DataFrame
        Valores de los inputs
    d : int
        Número de particiones que se quiere hacer en el espacio de los inputs

    Returns
    -------
    t : list
        Lista de particiones en cada dimensión
    t_ind : list
        Índice (posición) de la partición t en cada dimensión
    """

    # Número de columnas x
    n_dim = len(X.columns)
    # Lista de listas de ts
    t = list()
    # Lista de indices (posiciones) para crear el vector de subind
    t_ind = list()
    for col in range(0, n_dim):
        # Ts de la dimension col
        ts = list()
        t_max = X.iloc[:, col].max()
        t_min = X.iloc[:, col].min()
        amplitud = (t_max - t_min) / (d)
        for i in range(0, d + 1):
            t_i = t_min + i * amplitud
            ts.append(t_i)
        t.append(ts)
        t_ind.append(arange(0, len(ts)))
    return t, t_ind

#SUSTITUYE A GENERATE_MODEL
def svf(X_cols,Y_cols, data, c, eps, d):
    """Función que crea las particiones de cada dimensión. Coge cada dimensión de inputs y la trocea en d particiones equidistantes

    Parameters
    ----------
    X_cols : list
        Lista con los nombres de los inputs
    Y_cols : list
        Lista con los nombres de los outputs
    data: DataFrame
        Dataframe con el conjunto de datos a entrenar
    c:
        Valor del hiperparámetro C
    eps:
        Valor de hiperparámetro épsilon
    d:
        Valor de hiperaparámetro d
    Returns
    -------
    mdl : class
        Objeto con el modelo generado
    """

    X = data.filter(X_cols)

    Y = data.filter(Y_cols)
    y = Y.values.tolist()

    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)

    #######################################################################
    # Matriz de t y de indices de ts
    t, t_ind = create_matrix_partitions(X, d)
    # vector de subindices de w
    vector_subind = list()
    for combination in product(*t_ind):
        vector_subind.append(combination)
    M = calculate_matriz_transformations(X, t, vector_subind)

    # Número de variables w
    n_var = len(M[0])
    #######################################################################

    # Variable w
    ##name_w: (i,j)-> i:es el indice de la columna de la matriz phi;j: es el índice de la dimensión de y
    name_w = [(i, j) for i in range(0, n_dim_y) for j in range(0, n_var)]
    w={}
    w=w.fromkeys(name_w, 1)

    # Variable Xi
    name_xi = [(i, j) for i in range(0, n_dim_y) for j in range(0, n_obs)]
    xi = {}
    xi=xi.fromkeys(name_xi, c)

    mdl = Model("SVF Multioutput")
    mdl.context.cplex_parameters.threads = 1

    ##Variable w
    w_var=mdl.continuous_var_dict(name_w,ub=1e+33,lb=0,name='w')
    ##Variable xi
    xi_var=mdl.continuous_var_dict(name_xi,ub=1e+33,lb=0,name='xi')

    # Función objetivo
    mdl.minimize(mdl.sum(w_var[i]*w[i] for i in name_w)+mdl.sum(xi_var[i]*xi[i] for i in name_xi))

    #Restricciones
    for i in range(0,n_obs):
        for r in range(0,n_dim_y):
            left_side=y[i][r] - mdl.sum(w_var[r, j] * M[i][j] for j in range(0, n_var))
            ##(1)
            mdl.add_constraint(
                left_side <= 0,
                ctname='c1_'+str(i)+"_"+str(r)
            )
            ##(2)
            mdl.add_constraint(
                -left_side <= eps + xi_var[r, i],
                ctname='c2_' + str(i) + "_" + str(r)
            )
    return mdl

#SUSTITUYE A MODIFY_MODEL()
def modify_svf(model, c, eps,prev_eps,n_obs,n_dim_y):
    """Modifica un modelo SVF generado

    Parameters
    ----------
    model: class
        Modelo a modificar
    c:
        Valor del hiperparámetro C
    eps:
        Valor del hiperparámetro eps
    prev_eps:
        Valor del hiperparámetro eps con el que se ha entrenado anteriormente el modelo
    n_obs:
        Número de observaciones del problema
    n_dim_y:
        Número de dimensiones de outputs del problema
    Returns
    -------
    mdl : class
        Objeto con el modelo generado
    """

    name_var=model.iter_variables()
    name_w=list()
    name_xi=list()
    for var in name_var:
        name=var.get_name()
        if name.find("w") == -1:
            name_xi.append(name)
        else:
            name_w.append(name)
    # Variable w
    w={}
    w = w.fromkeys(name_w, 1)

    # Variable Xi
    xi = {}
    xi = xi.fromkeys(name_xi, c)

    a=[model.get_var_by_name(i) * w[i] for i in name_w]
    b=[model.get_var_by_name(i) * xi[i] for i in name_xi]
    # Función objetivo
    model.minimize(model.sum(a)+model.sum(b))
    #Modificar restricciones
    for i in range(0,n_obs):
        for r in range(0,n_dim_y):
            const_name='c2_' + str(i) + "_" + str(r)
            rest = model.get_constraint_by_name(const_name)
            rest.rhs += (eps - prev_eps)
    return model

def get_solutions(model, n_dim_y):
    """Proporciona las soluciones del modelo entrenado.

    Parameters
    ----------
    model: class
        Modelo entrenado
    n_dim_y:
        Número de dimensiones de outputs del problema
    Returns
    -------
    mat_w : list
        Lista con los valores de w asociados a cada solución
    """
    model.solve()
    name_var = model.iter_variables()
    sol_w = list()
    sol_xi = list()
    for var in name_var:
        name = var.get_name()
        sol = model.solution[name]
        if name.find("w") == -1:
            sol_xi.append(sol)
        else:
            sol_w.append(sol)
    # Numero de ws por dimensión
    n_w_dim = int(len(sol_w) / n_dim_y)
    mat_w = [[] for i in range(0, n_dim_y)]
    cont = 0
    for i in range(0, n_dim_y):
        for j in range(0, n_w_dim):
            mat_w[i].append(round(sol_w[cont],6))
            cont += 1
    # print("MAT_W:",mat_w)
    return mat_w

#ESTA FUNCIÓN NO SERÍA NECESARIA PARA UN USUARIO FINAL. SOLO PARA NUESTRAS SIMULACIONES.
def calculate_d(data):
    """Función que calcula los valores del hiperaparámetro d (número de particiones en cada dimensión de inputs)

    Parameters
    ----------
    data:
        Conjunto de datos a entrenar
    Returns
    -------
    d : list
        Lista con los valores de d
    """
    n_obs = len(data.index)
    d = list()
    for i in range(1, 11):
        n = int(round(0.1 * i * n_obs, 0))
        if n > 0:
            d.append(n)
    return d

def svf_cv(X_cols,Y_cols,data, C, eps, D, n_folds, seed):
    """Función que realiza la validación cruzada de un conjunto de datos

    Parameters
    ----------
    X_cols : list
        Lista con los nombres de los inputs
    Y_cols : list
        Lista con los nombres de los outputs
    data: DataFrame
        Dataframe con el conjunto de datos a entrenar
    C: list
        Lista del hiperparámetro C con los valores a probar
    eps: list
        Lista del hiperparámetro épsilon con los valores a probar
    D: list
        Lista del hiperparámetro D con los valores a probar
    n_folds: int
        Número de folds a ejecutar en la validación cruzada
    seed:
        Semilla de la partición de folds
    Returns
    -------
    error_folds : Dataframe
        Dataframe con el valor de MSE de cada combinación de hiperparámetros
    """

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = 0
    error_folds = DataFrame(columns=["C","eps","d","error"])
    n_obs_data = len(data.values)
    for train_index, test_index in kf.split(data):
        fold += 1
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        n_obs_train = len(data_train.values)
        Y = data.filter(Y_cols)
        n_dim_y = len(Y.columns)
        for d in D:
            model_d = svf(X_cols,Y_cols,data_train, 0, 0, d)
            last_C=0
            for c in C:
                for i in range(len(eps)):
                    if i>0:
                        prev_eps=eps[i-1]
                    else:
                        if last_C==0:
                            prev_eps=0
                        elif last_C != c:
                            prev_eps=eps[len(eps)-1]
                    model = modify_svf(model_d, c, eps[i],prev_eps, n_obs_train, n_dim_y)
                    model.solve()
                    w=get_solutions(model,n_dim_y)
                    error_bruto = calculate_cv_mse(X_cols, Y_cols, data_train, data_test, w, d)
                    error_folds = error_folds.append(
                        {
                            "C": c,
                            "eps": eps[i],
                            "d": d,
                            "error": error_bruto,
                        },
                        ignore_index=True,
                    )
                    last_C=c
    error_folds = error_folds.groupby(['C','eps','d']).sum() / n_obs_data
    error_folds = error_folds.sort_index(ascending=False)
    return error_folds

def calculate_cv_mse(X_cols, Y_cols, data_train, data_test, w, d):
    """Función que calcula el MSE de un modelo entrenado. Compara la y del conjunto de datos de test (observada) con la y_est de SVF.

    Parameters
    ----------
    X_cols : list
        Lista con los nombres de los inputs
    Y_cols : list
        Lista con los nombres de los outputs
    data_train: DataFrame
        Conjunto de datos de entrenamiento
    data_test: Dataframe
        Conjunto de datos de test
    w : list
        Lista con los valores de w óptimos para un problema resulto
    d: int
        Valor del hiperparámetro d
    Returns
    -------
    error_folds : Dataframe
        Dataframe con el valor de MSE de cada combinación de hiperparámetros
    """

    data_test_X = data_test.filter(X_cols)
    data_test_Y = data_test.filter(Y_cols)
    n_dim_y = len(data_test_Y.columns)
    data_train_X = data_train.filter(X_cols)
    t, t_ind = create_matrix_partitions(data_train_X, d)
    error = 0
    for i in range(len(data_test_X)):
        x = data_test_X.iloc[i]
        for j in range(n_dim_y):
            y_est = svf_prediction(w[j], x, t, t_ind)
            y = data_test_Y.iloc[i, j]
            error = error + (y - y_est) ** 2
    return error

#SUSTITUYE A ESTIMACION
def svf_prediction(w, x, t, t_ind):
    """Función que calcula la predicción de una observación según una observación y los pesos óptimos del modelo entrenado
       prediction = w*·phi(x)

    Parameters
    ----------
    w : list
        Lista con los valores de w óptimos para un problema resulto
    x : list
        Lista con los valores de inputs de la observación a predecir
    t : list
        Lista de particiones en cada dimensión
    t_ind : list
        Índice (posición) de la partición t en cada dimensión
    Returns
    -------
    error_folds : Dataframe
        Dataframe con el valor de MSE de cada combinación de hiperparámetros
    """

    p = locate_position_observation(x, t)
    vector_subind = list()
    for combination in product(*t_ind):
        vector_subind.append(combination)
    phi = calculate_transformation_observation(vector_subind, p)
    prediction = 0
    for i in range(0, len(w)):
        prediction = prediction + w[i] * phi[i]
    return prediction

def dataframe_d(X_cols, Y_cols, data, d, w):
    """Función que crea un dataframe con todas las celdas del grid y su valor y predicho
    Parameters
    ----------
    X_cols : list
        Lista con los valores de w óptimos para un problema resulto
    Y_cols : list
        Lista con los valores de inputs de la observación a predecir
    data: DataFrame

    d: int
        Lista de particiones en cada dimensión
    w: list
        Índice (posición) de la partición t en cada dimensión
    Returns
    -------
    df_d : Dataframe
        Dataframe que tiene como inputs los valores de las celdas del grid y como outputs su valor SVF predicho.
    """

    X = data.filter(X_cols)
    Y = data.filter(Y_cols)
    n_dim_y = len(Y.columns)
    t, t_ind = create_matrix_partitions(X, d)
    vector_ts = list()
    for combination in product(*t):
        vector_ts.append(combination)
    df_d = DataFrame(vector_ts)
    num_columns = len(df_d.columns)
    name_columns = ["x" + str(i + 1) for i in range(num_columns)]
    df_d.columns = name_columns
    df_y = [[] for i in range(0, n_dim_y)]
    for i in range(len(df_d)):
        x = df_d.iloc[i]
        for j in range(n_dim_y):
            y_est = svf_prediction(w[j], x, t, t_ind)
            df_y[j].append(round(y_est, 6))
    name_columns = ["y" + str(i + 1) for i in range(n_dim_y)]
    df_y_empty = DataFrame(columns=name_columns)
    df_d = concat((df_d, df_y_empty), axis=1)
    for j in range(n_dim_y):
        df_d["y" + str(j + 1)] = df_y[j]
    return df_d
