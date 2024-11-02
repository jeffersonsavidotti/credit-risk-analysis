# Importação de Bibliotecas
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Configuração do SparkSession
spark = SparkSession.builder \
    .appName("CreditRiskETL") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Definição do caminho para a camada gold
project_root = os.getcwd()
gold_file_path = os.path.join(project_root, 'pipeline_etl', 'etl_camadas', 'gold', 'credit_risk_gold.parquet')

# Carregando os dados da camada gold
if not os.path.exists(gold_file_path):
    print(f"Arquivo não encontrado: {gold_file_path}")
    spark.stop()
else:
    df_gold = spark.read.parquet(gold_file_path)
    print("Dados carregados com sucesso.")
    df_gold.show(5)

# Divisão dos Dados em Treino e Teste
train_data, test_data = df_gold.randomSplit([0.7, 0.3], seed=42)

# Treinamento do Modelo
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=100)
model = lr.fit(train_data)

# Avaliação do Modelo
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f"\nAcurácia do modelo: {accuracy:.2f}")

# Relatório de Classificação
predictions_pd = predictions.select('label', 'prediction').toPandas()
print("\nRelatório de Classificação:")
print(classification_report(predictions_pd['label'], predictions_pd['prediction']))

# Matriz de Confusão
cm = confusion_matrix(predictions_pd['label'], predictions_pd['prediction'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pagou', 'Default'], yticklabels=['Pagou', 'Default'])
plt.ylabel('Valor Real')
plt.xlabel('Previsão do Modelo')
plt.title('Matriz de Confusão')
plt.show()

# Calculando a AUC
binary_evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
auc_score = binary_evaluator.evaluate(predictions)
print(f"AUC: {auc_score:.2f}")

# Curva ROC
probabilities = predictions.select('probability', 'label').collect()
probs = np.array([row['probability'][1] for row in probabilities])
labels = np.array([row['label'] for row in probabilities])
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Análise dos Coeficientes do Modelo
coefficients = model.coefficients.toArray()
attrs = sorted(
    (attr["idx"], attr["name"]) 
    for attr in (train_data.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]
                 + train_data.schema["features"].metadata["ml_attr"]["attrs"].get("binary", [])
                 + train_data.schema["features"].metadata["ml_attr"]["attrs"].get("categorical", []))
)
features_names = [name for idx, name in attrs]
coeff_df = pd.DataFrame({'Feature': features_names, 'Coefficient': coefficients})
coeff_df['Abs_Coefficient'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Abs_Coefficient', ascending=False)

print("\nCoeficientes das Variáveis (Top 10):")
print(coeff_df[['Feature', 'Coefficient']].head(10))

# Importância das Variáveis
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df.head(10))
plt.title('Importância das Variáveis (Top 10)')
plt.xlabel('Coeficiente')
plt.ylabel('Variável')
plt.show()

# Visualização da Distribuição das Variáveis
df_original_gold = spark.read.parquet(gold_file_path)
expected_columns = ['loan_int_rate', 'label', 'person_age', 'person_income']
present_columns = [col for col in expected_columns if col in df_original_gold.columns]
missing_columns = [col for col in expected_columns if col not in df_original_gold.columns]

print("\nColunas presentes:", present_columns)
if missing_columns:
    print("Colunas faltantes:", missing_columns)
else:
    print("Todas as colunas esperadas estão presentes.")

df_original_pd = df_original_gold.sample(fraction=0.1, seed=42).toPandas()
if 'loan_int_rate' in df_original_pd.columns and 'label' in df_original_pd.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y='loan_int_rate', data=df_original_pd)
    plt.title('Distribuição da Taxa de Juros por Status do Empréstimo')
    plt.xlabel('Status do Empréstimo (0 = Pagou, 1 = Default)')
    plt.ylabel('Taxa de Juros (%)')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y='loan_percent_income', data=df_original_pd)
    plt.title('Percentual de Renda Comprometida por Status do Empréstimo')
    plt.xlabel('Status do Empréstimo (0 = Pagou, 1 = Default)')
    plt.ylabel('Percentual de Renda Comprometida')
    plt.show()
else:
    missing_for_viz = [col for col in ['loan_int_rate', 'label'] if col not in df_original_pd.columns]
    print(f"\nAs seguintes colunas necessárias para a visualização estão faltando: {missing_for_viz}")

# Finalizando a SparkSession
spark.stop()

# Conclusões e Insights
print("\nConclusões e Insights:")
print("Desempenho do Modelo:")
print(f"Acurácia: {accuracy * 100:.2f}%")
print(f"AUC: {auc_score:.2f}")
print("Principais Fatores de Risco:")
print(" - Taxa de Juros Elevada: Clientes com taxas de juros mais altas têm maior risco de default.")
print(" - Alto Comprometimento da Renda: Um alto percentual da renda comprometida com o empréstimo aumenta o risco.")
print(" - Curto Histórico de Crédito: Clientes com um histórico de crédito mais curto estão associados a um risco maior.")
print(" - Intenção do Empréstimo: Certos propósitos do empréstimo, como consolidação de dívidas, podem estar associados a maiores riscos.")
print("\nRecomendações:")
print(" - Avaliação Rigorosa: Realizar uma análise mais detalhada para clientes com taxas de juros altas e alto comprometimento de renda.")
print(" - Políticas de Crédito: Revisar as políticas para aprovar empréstimos com base nos fatores de risco identificados.")
print(" - Educação Financeira: Oferecer programas de educação financeira para clientes com histórico de crédito curto.")
