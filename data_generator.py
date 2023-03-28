import numpy as np

from scipy.integrate import solve_ivp
from doepy import build


def RWGS_reaction(x, p, T):
    pCO2, pH2, pCO, pH2O, pCH4, pN2 = x / x.sum() * p
    T += 273.15
    T_ref = 300 + 273.15
    k_ref = 8.13e-2 *1000
    Ea = 115*1000

    R = 8.31
    k = k_ref * np.exp(-Ea/R * (1/T - 1/T_ref))
    K_eq = np.exp(3.933 - 4076/(T - 39.64))

    a = 16.3

    return k * (pCO2 * np.power(pH2, 0.5) - pCO*pH2O/K_eq/np.power(pH2, 0.5)) / np.power(1 + a*pH2O/pH2, 2)

def FT_reaction(x, p, T):
    pCO2, pH2, pCO, pH2O, pCH4, pN2 = x / x.sum() * p
    T += 273.15
    T_ref = 300 + 273.15
    k_ref = 6.39e-2*1000
    Ea = 67.8*1000

    R = 8.31
    k = k_ref * np.exp(-Ea/R * (1/T - 1/T_ref))

    a = 9.07
    b = 2.44

    return k * pCO * pH2 / np.power(1 + a*pH2O/pH2 + b*pCO, 2)

def PFR_model(t, x, p, T):
    matrix_coef = np.array([[-1, 0],
                            [-1,-3],
                            [ 1,-1],
                            [ 1, 1],
                            [ 0, 1],
                            [ 0, 0]])

    reactions = np.array([RWGS_reaction(x, p, T), FT_reaction(x, p, T)])

    return np.dot(matrix_coef, reactions)

def get_sol(x0, V, p, T):
    sol = solve_ivp(PFR_model, [0, V.max()], x0, args=[p, T], dense_output=True, method='LSODA')
    return sol.sol(V)


def get_train_data():
    tau = np.append([0], np.exp(np.linspace(np.log(1e-4), np.log(0.1), 7)))

    inputs = []
    outputs = []

    CO2_0, H2_0, CO_0, N2_0 = 1, 3, 0, 1
    p, T = 10, 300
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))

    CO2_0, H2_0, CO_0, N2_0 = 1, 2, 0, 1
    p, T = 10, 300
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))

    CO2_0, H2_0, CO_0, N2_0 = 1, 6, 0, 1
    p, T = 10, 300
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))

    CO2_0, H2_0, CO_0, N2_0 = 1, 3, 0, 1
    p, T = 10, 250
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))

    CO2_0, H2_0, CO_0, N2_0 = 1, 3, 0, 1
    p, T = 10, 350
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))


    CO2_0, H2_0, CO_0, N2_0 = 1, 3, 0, 1
    p, T = 15, 300
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))

    CO2_0, H2_0, CO_0, N2_0 = 1, 3, 0, 1
    p, T = 20, 300
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))

    CO2_0, H2_0, CO_0, N2_0 = 0.5, 3, 0.5, 1
    p, T = 10, 300
    x0 = np.array([CO2_0, H2_0, CO_0, 0, 0, N2_0])
    inputs.append(np.append(x0, [T, p]))
    outputs.append(get_sol(x0, tau, p, T))

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    outputs = np.array([outputs[:, :, i] for i in range(outputs.shape[-1])])

    return tau, inputs, outputs


def get_test_data():
    tau = np.append([0], np.exp(np.linspace(np.log(1e-4), np.log(1), 20)))
    inputs = []
    outputs = []

    df = build.full_fact(
        {'Pressure':[12, 14, 16, 18],
         'Temperature':[255, 285, 315, 345],
         'CO2': [0, 0.5, 1.2],
         'H2':[2.5, 4.0, 5.5],
         'CO':[0, 0.3],
         'CH4':[0, 0.3],
         'H2O':[0, 0.3]
        })
    df['N2'] = 1

    nn_input = ['CO2', 'H2', 'CO', 'H2O', 'CH4', 'N2', 'Temperature', 'Pressure']
    for index in range(df.shape[0]):
        outputs.append(get_sol(df[nn_input[:-2]].iloc[index].values,
                             tau,
                             df['Pressure'].iloc[index],
                             df['Temperature'].iloc[index]
                         ))
        inputs.append(df[nn_input].iloc[index].values)

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    outputs = np.array([outputs[:, :, i] for i in range(outputs.shape[-1])])

    return tau, inputs, outputs

