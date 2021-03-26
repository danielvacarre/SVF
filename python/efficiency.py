from svf import dataframe_d
from numpy import round
from docplex.mp.model import Model
from numpy import nan,max,min

#SUSTITUYE A generate_vector_w_neg
def generate_ram_weight_input(n_dim_x,n_dim_y,X):
    """Función utilizada para calcular los pesos de los slacks de los inputs del modelo RAM

    Parameters
    ----------
    n_dim_x : int
        Numero de dimensiones de inputs
    n_dim_y : int
        Numero de dimensiones de outputs
    X: dataframe
        Dataframe con los datos de inputs
    Returns
    -------
    w_inp : list
        Lista con los pesos
    """

    X = X.to_numpy()
    w_inp=list()
    for i in range(n_dim_x):
        ran = (max(X[:,i]) - min(X[:,i]))
        sum = (n_dim_x + n_dim_y)
        w = 1 / ( sum * ran)
        w_inp.append(w)
    return w_inp

#SUSTITUYE A generate_vector_w_pos
def generate_ram_weight_output(n_dim_x,n_dim_y,Y):
    """Función utilizada para calcular los pesos de los slacks de los outputs del modelo RAM

    Parameters
    ----------
    n_dim_x : int
        Numero de dimensiones de inputs
    n_dim_y : int
        Numero de dimensiones de outputs
    Y: dataframe
        Dataframe con los datos de inputs
    Returns
    -------
    w_out : list
        Lista con los pesos
    """

    Y = Y.to_numpy()
    w_out=list()
    for i in range(n_dim_y):
        ran = (max(Y[:,i]) - min(Y[:,i]))
        sum = (n_dim_x + n_dim_y)
        w = 1 / ( sum * ran)
        w_out.append(w)
    return w_out

def efficiencySVF(X_cols, Y_cols, data, sol_w, best_d, best_eps, model, orientation):
    """Función que calcula diferentes medidas de eficiencia de un conjunto de datos.
       Para ello es necesario indicarle el modelo de eficiencia que se desea calcular y la orientación en caso de que sea necesario.

    Parameters
    ----------
    X_cols : list
        Lista con los nombres de los inputs
    Y_cols : list
        Lista con los nombres de los outputs
    data: DataFrame
        Dataframe con el conjunto de datos a entrenar
    sol_w: list
        Vector w de soluciones óptimas
    best_d: int
        Hiperporámetro d
    best_eps: float
        Hiperporámetro epsilon
    model: string
        Modelo a seleccionar:
            bcc: BCC model (Banker et. al. 1984)
            dd: DDF program based on ideas of Chambers et al. (1998)
            wa: Weighted Additive program based on ideas of Lovell and Pastor (1994). We used the weights of the RAM (Range Adjusted Measure) of inefficiency program of Cooper et al. (1999)
    orientation: string:
        in: input
        out: output
    Returns
    -------
    df_eff : Dataframe
        Dataframe que incluye el conjunto de datos original más:
            fdh: cálculo de la eficiencia FDH del modelo seleccionado
            dea: cálculo de la eficiencia DEA del modelo seleccionado
            svf: cálculo de la eficiencia SVF del modelo seleccionado
            csvf: cálculo de la eficiencia CSVF del modelo seleccionado
            svf_eps_insen: indica si la observación es epsilon insensible para el modelo SVF
            csvf_eps_insen: indica si la observación es epsilon insensible para el modelo CSVF
    """
    n_obs = len(data)

    df_d = dataframe_d(X_cols, Y_cols,data, best_d, sol_w)

    X = data.filter(X_cols)
    Y = data.filter(Y_cols)
    n_dim_x = len(X.columns)
    n_dim_y = len(Y.columns)

    # Creamos una copia del conjunto de datos inicial
    df_eff = data.copy()
    # Le añadimos 4 columnas
    df_eff['fdh'] = nan
    df_eff['dea'] = nan
    df_eff['svf'] = nan
    df_eff['csvf'] = nan
    df_eff['svf_eps_insen'] = ''
    df_eff['csvf_eps_insen'] = ''
    # Para cada observación del conjunto de datos inicial calculamos su eficiencia segun el modelo.
    for i in range(n_obs):
        if (model == "bcc"):
            if (orientation == "in"):
                df_eff['fdh'].values[i] = efficiency_fdh_bcc_input_oriented(data, i)
                df_eff['dea'].values[i] = efficiency_dea_bcc_input_oriented(data, i)
                df_eff['svf'].values[i] = efficiency_svf_bcc_input_oriented(data, df_d, i)
                df_eff['csvf'].values[i] = efficiency_csvf_bcc_input_oriented(data, df_d, i)
                df_eff['svf_eps_insen'].values[i] = efficiency_svf_bcc_input_oriented_epsilon_insensible(data, df_d, i,best_eps)
                df_eff['csvf_eps_insen'].values[i] = efficiency_csvf_bcc_input_oriented_epsilon_insensible(data, df_d,i, best_eps)
            elif (orientation == "out"):
                df_eff['fdh'].values[i] = efficiency_fdh_bcc_output_oriented(data, i)
                df_eff['dea'].values[i] = efficiency_dea_bcc_output_oriented(data, i)
                df_eff['svf'].values[i] = efficiency_svf_bcc_output_oriented(data, df_d, i)
                df_eff['csvf'].values[i] = efficiency_csvf_bcc_output_oriented(data, df_d, i)
                df_eff['svf_eps_insen'].values[i] = efficiency_svf_bcc_output_oriented_epsilon_insensible(data, df_d, i,best_eps)
                df_eff['csvf_eps_insen'].values[i] = efficiency_csvf_bcc_output_oriented_epsilon_insensible(data, df_d,i, best_eps)
        elif (model == "wa"):
            w_inp = generate_ram_weight_input(n_dim_x, n_dim_y, X)
            w_out = generate_ram_weight_output(n_dim_x, n_dim_y, Y)
            df_eff['fdh'].values[i] = efficiency_fdh_weighted_additive(data, i, w_inp, w_out)
            df_eff['dea'].values[i] = efficiency_dea_weighted_additive(data, i, w_inp, w_out)
            df_eff['svf'].values[i] = efficiency_svf_weighted_additive(data, df_d, i, w_inp, w_out)
            df_eff['csvf'].values[i] = efficiency_csvf_weighted_additive(data, df_d, i, w_inp, w_out)
            df_eff['svf_eps_insen'].values[i] = get_svf_weighted_additive_epsilon_insensible(data, df_d, i, w_inp, w_out, best_eps)
            df_eff['csvf_eps_insen'].values[i] = get_csvf_weighted_additive_epsilon_insensible(data, df_d, i, w_inp, w_out, best_eps)
        elif (model == "dd"):
            df_eff['fdh'].values[i] = efficiency_fdh_direction_distance(data, i)
            df_eff['dea'].values[i] = efficiency_dea_direction_distance(data, i)
            df_eff['svf'].values[i] = efficiency_svf_direction_distance(data, df_d, i)
            df_eff['csvf'].values[i] = efficiency_csvf_direction_distance(data, df_d, i)
            df_eff['svf_eps_insen'].values[i] = get_svf_direction_distance_epsilon_insensible(data, df_d, i, best_eps)
            df_eff['csvf_eps_insen'].values[i] = get_csvf_direction_distance_epsilon_insensible(data, df_d, i, best_eps)
    print(df_eff)
    return df_eff


## BCC OUTPUT ORIENTED ##

#The BCC output-oriented model offers a measure that allows a DMU to be approximated to the frontier by increasing the value of the outputs and maintaining the value of the inputs.

### FDH ###

def efficiency_fdh_bcc_output_oriented(data, i):
    """Función que calcula el valor de eficiencia FDH según el modelo BCC orientación output

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("FDH BCC OUTPUT ORIENTED")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    mdl.solve()
    eff = mdl.solution["phi"]
    return eff

### DEA ###

def efficiency_dea_bcc_output_oriented(data, i):
    """Función que calcula el valor de eficiencia DEA según el modelo BCC orientación output

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("DEA BCC OUTPUT ORIENTED")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    mdl.solve()
    eff = mdl.solution["phi"]
    return eff

### SVF ###

def efficiency_svf_bcc_output_oriented(data, df_est, i):
    """Función que calcula el valor de eficiencia SVF según el modelo BCC orientación output

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF BCC OUTPUT ORIENTED")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    #     print(landa_var)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    # print(mdl.export_to_string())
    mdl.solve()
    eff = mdl.solution["phi"]
    return eff

### CSVF ###

def efficiency_csvf_bcc_output_oriented(data, df_est, i):
    """Función que calcula el valor de eficiencia CSVF según el modelo BCC orientación output

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF BCC OUTPUT ORIENTED")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    #     print(landa_var)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["phi"]
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

### SVF EPSILON INSENSIBLE###

def efficiency_svf_bcc_output_oriented_epsilon_insensible(data, df_est, i,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida SVF del modelo BCC orientación output

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    eps_insen: string
        Indica si la DMU es o no es epsilon insensible
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            #Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF BCC OUTPUT ORIENTED epsilon insensible")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    #     print(landa_var)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * (y_est[k][r]) for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    # print(mdl.export_to_string())
    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution["phi"],3)
        if eff>1:
            eps_insen= 'No'
        else:
            eps_insen ='Yes'
    else:
        eff=0
        eps_insen = 'Yes'
    return eff,eps_insen

### CSVF EPSILON INSENSIBLE ###

def efficiency_csvf_bcc_output_oriented_epsilon_insensible(data, df_est, i,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida CSVF del modelo BCC orientación output

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    eps_insen: string
        Indica si la DMU es o no es epsilon insensible
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            #Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF BCC OUTPUT ORIENTED epsilon insensible")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    #     print(landa_var)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution["phi"],3)
        if eff>1:
            eps_insen= 'No'
        else:
            eps_insen ='Yes'
    else:
        eff=0
        eps_insen = 'Yes'
    return eff,eps_insen

## BCC INPUT ORIENTED ##

#The BCC input-oriented model offers a measure that allows a DMU to be approximated to the frontier by reducing the value of the inputs and maintaining the value of the outputs.

### FDH ###

def efficiency_fdh_bcc_input_oriented(data, i):
    """Función que calcula el valor de eficiencia FDH según el modelo BCC orientación input

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # y = [round(num,3) for num in y]
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("FDH BCC INPUT ORIENTED")
    # Variables
    ##Variable theta
    theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.minimize(theta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= theta * x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["theta"]
    else:
        eff=0
    # print(mdl.export_to_string())
    return eff

### DEA ###

def efficiency_dea_bcc_input_oriented(data, i):
    """Función que calcula el valor de eficiencia DEA según el modelo BCC orientación input

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # y = [round(num,3) for num in y]
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("DEA BCC INPUT ORIENTED")
    # Variables
    ##Variable theta
    theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.minimize(theta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= theta * x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["theta"]
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

### SVF ###

def efficiency_svf_bcc_input_oriented(data, df_est, i):
    """Función que calcula el valor de eficiencia SVF según el modelo BCC orientación input

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    y = [round(num,3) for num in y]
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    y_est = [round(num,3) for num in y_est]
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF BCC INPUT ORIENTED")
    # Variables
    ##Variable theta
    theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.minimize(theta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= theta * x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["theta"]
    else:
        eff=0
    # print(mdl.export_to_string())
    return eff

### CSVF ###

def efficiency_csvf_bcc_input_oriented(data, df_est, i):
    """Función que calcula el valor de eficiencia CSVF según el modelo BCC orientación input

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    y = [round(num,3) for num in y]
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    y_est = [round(num,3) for num in y_est]
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF BCC INPUT ORIENTED")
    # Variables
    ##Variable theta
    theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.minimize(theta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= theta * x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["theta"]
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

### SVF EPSILON INSENSIBLE ###

def efficiency_svf_bcc_input_oriented_epsilon_insensible(data, df_est, i,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida SVF del modelo BCC orientación input

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    eps_insen: string
        Indica si la DMU es o no es epsilon insensible
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    y = [round(num,3) for num in y]
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    y_est = [round(num,3) for num in y_est]
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            #Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF BCC INPUT ORIENTED epsilon insensible")
    # Variables
    ##Variable theta
    theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.minimize(theta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= theta * x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution["theta"],3)
        if eff<1:
            eps_insen= 'No'
        else:
            eps_insen ='Yes'
    else:
        eff=0
        eps_insen = 'Yes'
    return eff,eps_insen

### CSVF EPSILON INSENSIBLE  ###

def efficiency_csvf_bcc_input_oriented_epsilon_insensible(data, df_est, i,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida CSVF del modelo BCC orientación input

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    eps_insen: string
        Indica si la DMU es o no es epsilon insensible
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    y = [round(num,3) for num in y]
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    y_est = [round(num,3) for num in y_est]
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            #Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF BCC INPUT ORIENTED epsilon insensible")
    # Variables
    ##Variable theta
    theta = mdl.continuous_var(name="theta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.minimize(theta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= theta * x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution["theta"],3)
        if eff<1:
            eps_insen= 'No'
        else:
            eps_insen ='Yes'
    else:
        eff=0
        eps_insen = 'Yes'
    return eff,eps_insen

## DIRECTIONAL ##

#The directional distance function (DDF) program uses a vector g = (g^-,g^+)  to approximate a DMU to the frontier. In this case, the vector g is multiplied by an efficiency measure beta.
#The values of g^-=x and g^+=y

### FDH ###

def efficiency_fdh_direction_distance(data, i):
    """Función que calcula el valor de eficiencia FDH según el modelo direction distance function

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("FDH DIRECTIONAL DISTANCE")
    # Variables
    ##Variable beta
    beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(beta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <=x[i][j]-beta*x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[i][r]+beta*y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["beta"]
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

### DEA ###

def efficiency_dea_direction_distance(data, i):
    """Función que calcula el valor de eficiencia DEA según el modelo direction distance function

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("DEA DIRECTIONAL DISTANCE")
    # Variables
    ##Variable beta
    beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(beta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <=x[i][j]-beta*x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[i][r]+beta*y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["beta"]
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

def efficiency_svf_direction_distance(data, df_est, i):
    """Función que calcula el valor de eficiencia SVF según el modelo direction distance function

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF DIRECTIONAL DISTANCE")
    # Variables
    ##Variable beta
    beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(beta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-beta*x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+beta*y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["beta"]
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

def efficiency_csvf_direction_distance(data, df_est, i):
    """Función que calcula el valor de eficiencia CSVF según el modelo direction distance function

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF DIRECTIONAL DISTANCE")
    # Variables
    ##Variable beta
    beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(beta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-beta*x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+beta*y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["beta"]
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

def get_svf_direction_distance_epsilon_insensible(data, df_est, i,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida SVF del modelo DDF

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    eps_insen: string
        Indica si la DMU es o no es epsilon insensible
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            #Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF DIRECTIONAL DISTANCE epsilon insensible")
    # Variables
    ##Variable beta
    beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(beta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-beta*x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+beta*y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution["beta"],3)
        if eff>0:
            eps_insen= 'No'
        else:
            eps_insen ='Yes'
    else:
        eff=0
        eps_insen = 'Yes'
    return eff,eps_insen

def get_csvf_direction_distance_epsilon_insensible(data, df_est, i,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida CSVF del modelo DDF

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    eps_insen: string
        Indica si la DMU es o no es epsilon insensible
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            # Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF DIRECTIONAL DISTANCE epsilon insensible")
    # Variables
    ##Variable beta
    beta = mdl.continuous_var(name="beta", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(beta)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-beta*x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+beta*y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution["beta"],3)
        if eff>0:
            eps_insen= 'No'
        else:
            eps_insen ='Yes'
    else:
        eff=0
        eps_insen = 'Yes'
    return eff,eps_insen

## ADDITIVE ##

# Weighted additive model of Lovell and Pastor (1994). This program has the same constraints as the additive program of Charnes et al. (1985) but its objective function is modified through the assignment of weights to the input and the output slacks. With the combinations of weights and slacks a DMU can be projected to the efficient frontier.
#We used the weights of the RAM (Range Adjusted Measure) of inefficiency program of Cooper et al. (1999),

def efficiency_fdh_weighted_additive(data, i, w_inp, w_out):
    """Función que calcula el valor de eficiencia FDH según el modelo weighted additive

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    w_inp : list
        Lista con los pesos inputs
    w_out : list
        Lista con los pesos outputs
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    mdl = Model("FDH WEIGHTED ADDITIVE")
    # Variables
    ##Variable s
    name_s_neg = range(0, n_dim_x)
    s_neg_var=mdl.continuous_var_dict(name_s_neg,ub=1e+33,lb=0,name='s_neg')
    name_s_pos = range(0, n_dim_y)
    s_pos_var=mdl.continuous_var_dict(name_s_pos,ub=1e+33,lb=0,name='s_pos')
    # Variable landa
    name_landa = range(0, n_obs)
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(mdl.sum(s_neg_var[j] * w_inp[j] for j in range(n_dim_x)) +
                 mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <=x[i][j]-s_neg_var[j]
        )

    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[i][r]+s_pos_var[r]
        )

    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)

    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution.get_objective_value()
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

def efficiency_dea_weighted_additive(data, i, w_inp, w_out):
    """Función que calcula el valor de eficiencia DEA según el modelo weighted additive

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    w_inp : list
        Lista con los pesos inputs
    w_out : list
        Lista con los pesos outputs
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)

    mdl = Model("DEA WEIGHTED ADDITIVE")
    # Variables
    ##Variable s
    name_s_neg = range(0, n_dim_x)
    s_neg_var=mdl.continuous_var_dict(name_s_neg,ub=1e+33,lb=0,name='s_neg')
    name_s_pos = range(0, n_dim_y)
    s_pos_var=mdl.continuous_var_dict(name_s_pos,ub=1e+33,lb=0,name='s_pos')
    # Variable landa
    name_landa = range(0, n_obs)
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(mdl.sum(s_neg_var[j] * w_inp[j] for j in range(n_dim_x)) +
                 mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <=x[i][j]-s_neg_var[j]
        )

    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= y[i][r]+s_pos_var[r]
        )

    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)

    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution.get_objective_value()
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

def efficiency_svf_weighted_additive(data, df_est, i, w_inp, w_out):
    """Función que calcula el valor de eficiencia SVF según el modelo weighted additive

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    w_inp : list
        Lista con los pesos inputs
    w_out : list
        Lista con los pesos outputs
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)

    mdl = Model("SVF WEIGHTED ADDITIVE")
    # Variables
    ##Variable s
    name_s_neg = range(0, n_dim_x)
    s_neg_var=mdl.continuous_var_dict(name_s_neg,ub=1e+33,lb=0,name='s_neg')
    name_s_pos = range(0, n_dim_y)
    s_pos_var=mdl.continuous_var_dict(name_s_pos,ub=1e+33,lb=0,name='s_pos')
    # Variable landa
    name_landa = range(0, n_obs)
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(mdl.sum(s_neg_var[j] * w_inp[j] for j in range(n_dim_x)) +
                 mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-s_neg_var[j]
        )

    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+s_pos_var[r]
        )

    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)

    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution.get_objective_value()
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

def efficiency_csvf_weighted_additive(data, df_est, i, w_inp, w_out):
    """Función que calcula el valor de eficiencia CSVF según el modelo weighted additive

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    w_inp : list
        Lista con los pesos inputs
    w_out : list
        Lista con los pesos outputs
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)

    mdl = Model("CSVF WEIGHTED ADDITIVE")
    # Variables
    ##Variable s
    name_s_neg = range(0, n_dim_x)
    s_neg_var=mdl.continuous_var_dict(name_s_neg,ub=1e+33,lb=0,name='s_neg')
    name_s_pos = range(0, n_dim_y)
    s_pos_var=mdl.continuous_var_dict(name_s_pos,ub=1e+33,lb=0,name='s_pos')
    # Variable landa
    name_landa = range(0, n_obs)
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(mdl.sum(s_neg_var[j] * w_inp[j] for j in range(n_dim_x)) +
                 mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-s_neg_var[j]
        )

    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+s_pos_var[r]
        )

    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)

    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution.get_objective_value()
    else:
        eff=0
    #print(mdl.export_to_string())
    return eff

def get_svf_weighted_additive_epsilon_insensible(data, df_est, i, w_inp, w_out,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida SVF del modelo weighted additive

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    w_inp : list
        Lista con los pesos inputs
    w_out : list
        Lista con los pesos outputs
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """

    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            #Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value

    mdl = Model("SVF WEIGHTED ADDITIVE epsilon insensible")
    # Variables
    ##Variable s
    name_s_neg = range(0, n_dim_x)
    s_neg_var=mdl.continuous_var_dict(name_s_neg,ub=1e+33,lb=0,name='s_neg')
    name_s_pos = range(0, n_dim_y)
    s_pos_var=mdl.continuous_var_dict(name_s_pos,ub=1e+33,lb=0,name='s_pos')
    # Variable landa
    name_landa = range(0, n_obs)
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(mdl.sum(s_neg_var[j] * w_inp[j] for j in range(n_dim_x)) +
                 mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-s_neg_var[j]
        )

    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+s_pos_var[r]
        )

    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)

    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution.get_objective_value(), 3)
        if eff > 0:
            eps_insen = 'No'
        else:
            eps_insen = 'Yes'
    else:
        eff = 0
        eps_insen = 'Yes'
    return eff, eps_insen

def get_csvf_weighted_additive_epsilon_insensible(data, df_est, i, w_inp, w_out,eps):
    """Función que calcula si una DMU es epsilon insensible segun la medida CSVF del modelo weighted additive

    Parameters
    ----------
    data: dataframe
        Dataframe con el conjunto de datos
    df_est: dataframe
        Dataframe con las celdas del grid óptimo
    i: int
        Fila de la DMU del conjunto de datos de la que se quiere calcular su eficiencia
    w_inp : list
        Lista con los pesos inputs
    w_out : list
        Lista con los pesos outputs
    eps: flaot
        Valor de epsilon del modelo óptimo
    Returns
    -------
    eff: float
        Valor de la eficiencia
    """


    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    for obs in range(n_obs):
        for dim_y in range(n_dim_y):
            #Calculo el valor de y_sfv-eps. Si el valor es negativo le pongo un 0
            value = y_est[obs][dim_y] - eps
            if value < 0:
                y_est[obs][dim_y] = 0
            else:
                y_est[obs][dim_y] = value

    mdl = Model("CSVF WEIGHTED ADDITIVE epsilon insensible")
    # Variables
    ##Variable s
    name_s_neg = range(0, n_dim_x)
    s_neg_var=mdl.continuous_var_dict(name_s_neg,ub=1e+33,lb=0,name='s_neg')
    name_s_pos = range(0, n_dim_y)
    s_pos_var=mdl.continuous_var_dict(name_s_pos,ub=1e+33,lb=0,name='s_pos')
    # Variable landa
    name_landa = range(0, n_obs)
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(mdl.sum(s_neg_var[j] * w_inp[j] for j in range(n_dim_x)) +
                 mdl.sum(s_pos_var[r] * w_out[r] for r in range(n_dim_y)))
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <=x[i][j]-s_neg_var[j]
        )

    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= y[i][r]+s_pos_var[r]
        )

    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)

    msol=mdl.solve()
    if msol is not None:
        eff = round(mdl.solution.get_objective_value(), 3)
        if eff > 0:
            eps_insen = 'No'
        else:
            eps_insen = 'Yes'
    else:
        eff = 0
        eps_insen = 'Yes'
    return eff, eps_insen

