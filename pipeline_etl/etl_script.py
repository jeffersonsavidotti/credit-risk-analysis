# Importação das Bibliotecas Necessárias
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
import os

# Configuração do SparkSession
spark = SparkSession.builder \
    .appName("CreditRiskETL") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Função para criar diretórios, se não existirem
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Definindo o diretório base e criação das pastas
project_root = os.getcwd()
base_dir = os.path.join(project_root, 'pipeline_etl', 'etl_camadas')
bronze_dir = os.path.join(base_dir, 'bronze')
silver_dir = os.path.join(base_dir, 'silver')
gold_dir = os.path.join(base_dir, 'gold')

# Criar diretórios para as camadas de ETL
create_dir(bronze_dir)
create_dir(silver_dir)
create_dir(gold_dir)

# Caminho para o arquivo CSV original
raw_data_path = os.path.join(project_root, 'dataset', 'credit_risk_dataset.csv')

# Carregamento e Processamento da Camada Bronze
if not os.path.exists(raw_data_path):
    print(f"Arquivo CSV não encontrado: {raw_data_path}. Verifique o caminho e tente novamente.")
    spark.stop()  # Finaliza Spark se o arquivo estiver ausente
    exit()
else:
    # Carregando dados e criando camada bronze
    df_bronze = spark.read.csv(raw_data_path, header=True, inferSchema=True)
    df_bronze.show(5)
    df_bronze.printSchema()

    bronze_file_path = os.path.join(bronze_dir, 'credit_risk_bronze.parquet')
    df_bronze.write.mode('overwrite').parquet(bronze_file_path)
    print(f"\nCamada bronze salva com sucesso em: {bronze_file_path}")

    # Criando camada silver a partir da bronze
    df_silver = df_bronze

    # Tratamento de valores nulos e conversão de tipos na Camada Silver
    median_emp_length = df_silver.approxQuantile("person_emp_length", [0.5], 0.25)[0]
    df_silver = df_silver.na.fill({"person_emp_length": median_emp_length}).withColumn("person_age", col("person_age").cast(IntegerType())) \
        .withColumn("person_income", col("person_income").cast(DoubleType())) \
        .withColumn("loan_amnt", col("loan_amnt").cast(DoubleType())) \
        .withColumn("loan_int_rate", col("loan_int_rate").cast(DoubleType())) \
        .withColumn("loan_status", col("loan_status").cast(IntegerType())) \
        .withColumn("loan_percent_income", col("loan_percent_income").cast(DoubleType())) \
        .withColumn("cb_person_cred_hist_length", col("cb_person_cred_hist_length").cast(IntegerType()))
    df_silver.show(5)

    # Salvando a camada silver
    silver_file_path = os.path.join(silver_dir, 'credit_risk_silver.parquet')
    df_silver.write.mode('overwrite').parquet(silver_file_path)
    print(f"\nCamada silver salva com sucesso em: {silver_file_path}")

# Carregamento e Enriquecimento da Camada Gold
if not os.path.exists(silver_file_path):
    print(f"Arquivo silver não encontrado: {silver_file_path}")
else:
    df_silver = spark.read.parquet(silver_file_path)
    print("\nArquivo silver carregado com sucesso.")

    # Criando a camada gold a partir da camada silver
    df_gold = df_silver.na.fill({"person_home_ownership": "OTHER"}).withColumn(
        'is_home_owner', when(col('person_home_ownership').isin('OWN', 'MORTGAGE'), 1).otherwise(0)
    )
    df_gold = df_gold.na.fill({
        "person_age": 0, "person_income": 0.0, "person_emp_length": 0.0, "loan_amnt": 0.0,
        "loan_int_rate": 0.0, "loan_percent_income": 0.0, "cb_person_cred_hist_length": 0, "is_home_owner": 0
    })

    # Codificação de variáveis categóricas
    categorical_vars = ['loan_intent', 'loan_grade', 'cb_person_default_on_file']
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid='keep') for col in categorical_vars]
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec") for col in categorical_vars]
    pipeline = Pipeline(stages=indexers + encoders)
    df_gold = pipeline.fit(df_gold).transform(df_gold)

    # Criando o vetor de features
    assembler_inputs = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income', 'cb_person_cred_hist_length', 'is_home_owner'
    ] + [encoder.getOutputCol() for encoder in encoders]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")
    df_gold = assembler.transform(df_gold)

    # Selecionando as colunas necessárias para a camada gold
    df_gold = df_gold.select(
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income', 'cb_person_cred_hist_length', 'is_home_owner', 'features',
        col('loan_status').alias('label')
    )
    df_gold.show(5)
    df_gold.printSchema()

    # Salvando a camada gold
    gold_file_path = os.path.join(gold_dir, 'credit_risk_gold.parquet')
    df_gold.write.mode('overwrite').parquet(gold_file_path)
    print(f"\nCamada gold salva com sucesso em: {gold_file_path}")

# Finalizando a SparkSession
spark.stop()
