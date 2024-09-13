import warnings
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics")

"""
==================================================================================================================================
Q2: CONSIDERING THE APPROVED PROMOTIONS FOR THE LAST QUARTER OF 2018, HOW MANY TONS OF PRODUCTS ARE WE GOING TO SELL IN Q4 2018?
==================================================================================================================================
"""

# Passo 1: Filtrar as 5.148 promoções aprovadas para o Q4 de 2018 e mapear as combinações de basecode e CUSTOMER_HIERARCHY_LVL2_CD.
# Carregar os dados de promoções e vendas
promo_data = pandas.read_excel('Promo.xlsx')
invoice_data = pandas.read_excel('Invoice.xlsx')

# Converter as colunas de data para datetime
promo_data['event_start_dt'] = pandas.to_datetime(promo_data['event_start_dt'])
promo_data['event_end_dt'] = pandas.to_datetime(promo_data['event_end_dt'])
invoice_data['INVOICE_DT'] = pandas.to_datetime(invoice_data['INVOICE_DT'])

# Definir o período de Q4 de 2018
q4_2018_start = pandas.to_datetime('2018-10-01')
q4_2018_end = pandas.to_datetime('2018-12-31')

# Filtrar as promoções que ocorrem no Q4 de 2018
promo_q4_2018 = promo_data[(promo_data['event_start_dt'] >= q4_2018_start) & (promo_data['event_end_dt'] <= q4_2018_end)]

# Exibir a quantidade de promoções e as combinações de basecode e CUSTOMER_HIERARCHY_LVL2_CD
print(f"\nTotal Promoções Q4 2018: {len(promo_q4_2018)}")
promo_combinations_q4_2018 = promo_q4_2018[['basecode', 'customer_hierarchy_lvl2_cd']].drop_duplicates()
print(f"Total Combinações Q4 2018: {len(promo_combinations_q4_2018)}")

# Passo 2: Filtrar essas combinações dentro do Q4 de 2017 e somar o volume (invoice_qty).
# Definir o período de Q4 de 2017
q4_2017_start = pandas.to_datetime('2017-10-01')
q4_2017_end = pandas.to_datetime('2017-12-31')

# Filtrar as vendas que ocorreram no Q4 de 2017
q4_sales_2017 = invoice_data[(invoice_data['INVOICE_DT'] >= q4_2017_start) & (invoice_data['INVOICE_DT'] <= q4_2017_end)]

# Filtrar as vendas no Q4 de 2017 com as combinações de basecode e CUSTOMER_HIERARCHY_LVL2_CD de Q4 de 2018
filtered_sales_q4_2017 = pandas.merge(q4_sales_2017, promo_combinations_q4_2018, how='inner',
                                      left_on=['basecode', 'CUSTOMER_HIERARCHY_LVL2_CD'],
                                      right_on=['basecode', 'customer_hierarchy_lvl2_cd'])

# Somar o volume total para cada combinação de basecode e CUSTOMER_HIERARCHY_LVL2_CD
grouped_sales_q4_2017 = filtered_sales_q4_2017.groupby(['basecode', 'CUSTOMER_HIERARCHY_LVL2_CD'])['invoice_qty'].sum().reset_index()

# Exibir o somatório do volume para cada combinação
print(f"Total Combinações Q4 2017: {len(grouped_sales_q4_2017)}")
print(f"Total Volume Q4 2017: {sum(grouped_sales_q4_2017['invoice_qty'])}")

# Passo 3: Treinar o modelo de Gradient Boosting Machines (GBM)
# Separar as features (input) e o target (output)
X = grouped_sales_q4_2017[['basecode', 'CUSTOMER_HIERARCHY_LVL2_CD']]  # Variáveis de entrada
y = grouped_sales_q4_2017['invoice_qty']  # Variável de saída (volume de vendas)

# Usar LabelEncoder para codificar basecode e CUSTOMER_HIERARCHY_LVL2_CD
le_basecode = LabelEncoder()
le_customer = LabelEncoder()

# Aplicar o LabelEncoder nas colunas de input usando `.loc[]`
X.loc[:, 'basecode'] = le_basecode.fit_transform(X['basecode'])
X.loc[:, 'CUSTOMER_HIERARCHY_LVL2_CD'] = le_customer.fit_transform(X['CUSTOMER_HIERARCHY_LVL2_CD'])

# Escalar os dados de entrada
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de Gradient Boosting Regressor
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo com MAE e RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"\nErro Médio Absoluto (MAE): {mae:.2f} toneladas")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f} toneladas")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Passo 4: Prever as vendas para o Q4 de 2018
# Prever as vendas para o Q4 de 2018
X_pred_2018 = promo_combinations_q4_2018.copy()

# Ajustar os nomes das colunas para garantir consistência
X_pred_2018.columns = ['basecode', 'CUSTOMER_HIERARCHY_LVL2_CD']

# Apenas usar categorias presentes no Q4 de 2017
X_pred_2018 = X_pred_2018[X_pred_2018['basecode'].isin(le_basecode.classes_) & X_pred_2018['CUSTOMER_HIERARCHY_LVL2_CD'].isin(le_customer.classes_)]

# Codificar os dados de 2018 usando `.loc[]`
X_pred_2018.loc[:, 'basecode'] = le_basecode.transform(X_pred_2018['basecode'])
X_pred_2018.loc[:, 'CUSTOMER_HIERARCHY_LVL2_CD'] = le_customer.transform(X_pred_2018['CUSTOMER_HIERARCHY_LVL2_CD'])

# Escalar os dados de entrada para previsão
X_pred_2018 = scaler.transform(X_pred_2018)

# Fazer a previsão para Q4 de 2018
y_pred_2018 = model.predict(X_pred_2018)

# Exibir o total previsto para Q4 de 2018
print(f"Previsão de vendas para o Q4 de 2018: {y_pred_2018.sum():.2f} toneladas")
