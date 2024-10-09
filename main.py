# Bibliotecas
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

# Identificação - Smith
def identificacao_smith(time, potencia):
    last_value = potencia[-1] # Último valor do vetor potência

    # Definindo os percentuais de 28.3% e 63.2%
    t1_percent = 0.283 * last_value
    t2_percent = 0.632 * last_value

    # índices de onde ocorre 28.3% e 63.2%
    index_t1 = np.argmax(potencia >= t1_percent)
    index_t2 = np.argmax(potencia >= t2_percent)
    t1 = time[index_t1]
    t2 = time[index_t2]
    
    tau = 1.5*(t2-t1)
    theta = t2 - tau
    k = last_value
    
    return tau, theta, k
    
# Identificação - Sundaresan 
def identificacao_sundaresan(time, potencia):
    last_value = potencia[-1] # Último valor do vetor potência

    # Definindo os percentuais de 35.3% e 85.3%
    t1_percent = 0.353 * last_value
    t2_percent = 0.853 * last_value

    # índices de onde ocorre 35.3% e 85.2%
    index_t1 = np.argmax(potencia >= t1_percent)
    index_t2 = np.argmax(potencia >= t2_percent)
    t1 = time[index_t1]
    t2 = time[index_t2]
    
    tau = 2.0*(t2-t1)/3.0
    theta = 1.3*t1 - 0.29*t2
    k = last_value
    
    return tau, theta, k

# Carrega o arquivo .mat
data = loadmat('Dataset_Grupo8.mat')

# Exibe as variáveis contidas no arquivo
# print(data.keys())

# Dados
data_degrau = np.array(data['TARGET_DATA____ProjetoC213_Degrau'])
data_potencia = np.array(data['TARGET_DATA____ProjetoC213_PotenciaMotor'])

# Separacao das variaveis
time = data_degrau[:,0]                         # Tempo [x]
degrau = data_degrau[:,1]                       # Degrau [y]
potencia = data_potencia[:,1]                   # Potencia [y]
potencia = [y - potencia[0] for y in potencia]  # Normalizacao potencia

# Calculo das variaveis tau, theta e k
# tau, theta, k = identificacao_smith(time, potencia)
tau, theta, k = identificacao_sundaresan(time, potencia)
print(f"tau: {tau}")
print(f"theta = {theta}")
print(f"k = {k}")

# Criar a função de transferência do sistema de primeira ordem (sem tempo morto)
num = [k]  # Numerador (ganho K)
den = [tau, 1]  # Denominador (τs + 1)

# Sistema sem tempo morto
sistema_sem_atraso = ctrl.TransferFunction(num, den)

# Aproximação de Padé de 1ª ordem para modelar o atraso
# Note que a função pade() retorna numerador e denominador para o atraso
num_pade, den_pade = ctrl.pade(theta, 1)

# Criar a função de transferência do atraso
sistema_com_atraso = ctrl.TransferFunction(num_pade, den_pade)

# Sistema completo com atraso
sistema_total = ctrl.series(sistema_com_atraso, sistema_sem_atraso)

# Simular a resposta ao degrau
tempo, resposta = ctrl.step_response(sistema_total, T=time)

# Graficos
plt.subplot(1, 2, 1)
plt.plot(tempo, resposta, color='blue')
plt.plot(tempo, degrau, label="Degrau", color="blue", linestyle="--")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(tempo, potencia, color='red')
plt.plot(tempo, degrau, label="Degrau", color="blue", linestyle="--")
plt.grid(True)
plt.tight_layout()
plt.show()