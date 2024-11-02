
---

# Análise de Risco de Crédito com Pipeline ETL em PySpark e Machine Learning

Este projeto consiste em uma análise de risco de crédito utilizando um pipeline ETL implementado em PySpark, seguido por um modelo de Machine Learning para prever a inadimplência de clientes. O objetivo é estimar as chances de um consumidor honrar um compromisso financeiro acordado, auxiliando instituições financeiras na tomada de decisões de crédito.

## Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Descrição dos Dados](#descrição-dos-dados)
- [Pipeline ETL](#pipeline-etl)
  - [Camada Bronze](#camada-bronze)
  - [Camada Silver](#camada-silver)
  - [Camada Gold](#camada-gold)
- [Modelagem de Machine Learning](#modelagem-de-machine-learning)
- [Dashboard Power BI](#dashboard-power-bi)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Execução](#execução)
  - [Executando o Pipeline ETL](#executando-o-pipeline-etl)
  - [Executando o Script de Machine Learning](#executando-o-script-de-machine-learning)
  - [Executando com Notebooks Jupyter](#executando-com-notebooks-jupyter)
- [Conclusão](#conclusão)
- [Licença](#licença)
- [Agradecimentos](#agradecimentos)

---

## Visão Geral

A análise de risco de crédito é fundamental para instituições financeiras, pois permite estimar a probabilidade de um cliente não cumprir com suas obrigações financeiras. Este projeto utiliza dados de clientes para construir um modelo preditivo que identifica os principais fatores associados à inadimplência, utilizando PySpark para processamento eficiente de dados em grandes volumes.

## Estrutura do Projeto

A estrutura do projeto está organizada da seguinte forma:

```
credit-risk-analysis/
├── business/
│   └── dashboard.pbix
├── dataset/
│   └── credit_risk_dataset.csv
├── env/
├── etl_pipeline/
│   ├── etl_camadas/
│   │   ├── bronze/
│   │   ├── silver/
│   │   └── gold/
│   ├── etl_script.ipynb
│   └── etl_script.py
└── ml/
    ├── ml_script.ipynb
    └── ml_script.py
```

- **business/**: Contém documentos relacionados ao negócio e o dashboard em Power BI (`dashboard.pbix`) com visualizações sobre o risco de crédito.
- **dataset/**: Armazena o arquivo CSV original com os dados brutos.
- **env/**: Contém o ambiente virtual com as dependências do projeto.
- **etl_pipeline/**: Contém o script do pipeline ETL e as camadas resultantes.
  - **etl_camadas/**: Diretório com as camadas do ETL (bronze, silver, gold).
  - **etl_script.py**: Script Python que executa o pipeline ETL usando PySpark.
  - **etl_script.ipynb**: Notebook Jupyter para execução interativa do ETL.
- **ml/**: Contém o script e notebook de Machine Learning.
  - **ml_script.py**: Script Python para modelagem preditiva.
  - **ml_script.ipynb**: Notebook Jupyter para execução interativa do modelo.

## Descrição dos Dados

O conjunto de dados utilizado contém informações demográficas e financeiras dos clientes, incluindo:

- **person_age**: Idade da pessoa.
- **person_income**: Renda anual da pessoa.
- **person_home_ownership**: Situação de moradia (ALUGUEL, PRÓPRIA, HIPOTECADA).
- **person_emp_length**: Tempo de emprego em anos.
- **loan_intent**: Intenção do empréstimo (PESSOAL, EDUCAÇÃO, DÍVIDA, etc.).
- **loan_grade**: Classificação de crédito do empréstimo.
- **loan_amnt**: Valor do empréstimo.
- **loan_int_rate**: Taxa de juros do empréstimo.
- **loan_status**: Status do empréstimo (0 = Pagou, 1 = Inadimplente).
- **loan_percent_income**: Percentual da renda comprometida com o empréstimo.
- **cb_person_default_on_file**: Histórico de inadimplência (Y = Sim, N = Não).
- **cb_person_cred_hist_length**: Duração do histórico de crédito em anos.

## Pipeline ETL

O pipeline ETL (Extract, Transform, Load) é implementado em PySpark e é composto por três camadas:

### Camada Bronze

- **Objetivo**: Armazenar os dados brutos exatamente como foram recebidos.
- **Processos**:
  - Carregamento do arquivo CSV original.
  - Armazenamento dos dados sem modificações.
- **Formato de Armazenamento**: Parquet.
- **Localização**: `etl_pipeline/etl_camadas/bronze/credit_risk_bronze.parquet`.

### Camada Silver

- **Objetivo**: Realizar a limpeza e transformação básica dos dados.
- **Processos**:
  - Tratamento de valores ausentes.
  - Conversão de tipos de dados.
  - Remoção de inconsistências.
- **Formato de Armazenamento**: Parquet.
- **Localização**: `etl_pipeline/etl_camadas/silver/credit_risk_silver.parquet`.

### Camada Gold

- **Objetivo**: Preparar os dados para a modelagem de Machine Learning.
- **Processos**:
  - Engenharia de características (feature engineering).
  - Codificação de variáveis categóricas utilizando `StringIndexer` e `OneHotEncoder`.
  - Montagem do vetor de features com `VectorAssembler`.
- **Formato de Armazenamento**: Parquet.
- **Localização**: `etl_pipeline/etl_camadas/gold/credit_risk_gold.parquet`.

## Modelagem de Machine Learning

O script de Machine Learning consome os dados da camada gold para treinar um modelo preditivo de risco de crédito.

- **Algoritmo Utilizado**: Regressão Logística implementada com PySpark MLlib.
- **Processos**:
  - Carregamento dos dados da camada gold.
  - Divisão dos dados em conjuntos de treinamento e teste.
  - Treinamento do modelo de Regressão Logística.
  - Avaliação do modelo utilizando métricas como acurácia, matriz de confusão e AUC-ROC.
  - Geração de gráficos para visualização dos resultados.
  - Análise dos coeficientes para identificar os principais fatores de risco.

## Dashboard Power BI

O projeto inclui um dashboard em Power BI localizado na pasta `business/` como `dashboard.pbix`. Esse dashboard fornece visualizações e insights sobre o risco de crédito dos clientes, facilitando a análise de tendências e padrões que podem impactar a decisão de crédito.

## Pré-requisitos

- **Python 3.8+**
- **Java (JDK 8 ou superior)**: Necessário para o funcionamento do PySpark.
- **Ambiente virtual**: Recomendado para isolamento das dependências.
- **Bibliotecas Python**:
  - `pyspark`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
- **Power BI Desktop**: Para abrir e visualizar o dashboard.

## Instalação

1. **Clonar o Repositório**:

   ```bash
   git clone https://github.com/seu_usuario/credit-risk-analysis.git
   ```

2. **Navegar até o Diretório do Projeto**:

   ```bash
   cd credit-risk-analysis
   ```

3. **Criar um Ambiente Virtual**:

   ```bash
   python -m venv env
   ```

4. **Ativar o Ambiente Virtual**:

   - **Linux/macOS**:

     ```bash
     source env/bin/activate
     ```

   - **Windows**:

     ```bash
     env\Scripts\activate
     ```

5. **Instalar as Dependências**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Verificar a Instalação do Java**:

   - Certifique-se de que o Java está instalado e configurado corretamente.
   - Verifique a versão do Java:

     ```bash
     java -version
     ```

## Execução

### Executando o Pipeline ETL

#### Opção 1: Executar o Script ETL

1. **Navegar até o Diretório ETL**:

   ```bash
   cd etl_pipeline
   ```

2. **Executar o Script ETL**:

   ```bash
   python etl_script.py
   ```

   **Descrição**: O script processa os dados brutos e os armazena nas camadas `bronze`, `silver` e `gold` em `etl_pipeline/etl_camadas/`.

#### Opção 2: Executar o Notebook ETL

1. **Abrir o Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. **Navegar até o Notebook**:

   No Jupyter, abra o arquivo `etl_script.ip

ynb` em `etl_pipeline/`.

3. **Executar o Notebook**:

   O notebook executará as mesmas etapas do script para processamento ETL interativo.

### Executando o Script de Machine Learning

#### Opção 1: Executar o Script ML

1. **Navegar até o Diretório de Machine Learning**:

   ```bash
   cd ../ml
   ```

2. **Executar o Script ML**:

   ```bash
   python ml_script.py
   ```

#### Opção 2: Executar o Notebook de Machine Learning

1. **Abrir o Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. **Navegar até o Notebook**:

   No Jupyter, abra o arquivo `ml_script.ipynb` em `ml/`.

3. **Executar o Notebook**:

   O notebook permite uma execução interativa das etapas de modelagem e avaliação.

### Visualização dos Gráficos

- Os gráficos são gerados automaticamente durante a execução do script e notebook de Machine Learning.
- Se estiver usando o Jupyter Notebook, os gráficos aparecerão embutidos.

## Conclusão

Este projeto foi desenvolvido por **Jefferson Savidotti** como parte de um processo de aprendizagem para se tornar um profissional da área de dados em TI. Foi orientado tecnicamente por **Vitor Hugo do Nascimento** e supervisionado pelo **Tech Leader Jilian Capri** da **Programmer's Beyond IT**. A estrutura modular do projeto facilita a manutenção e a escalabilidade, com visualizações adicionais fornecidas no dashboard Power BI para análise exploratória.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Agradecimentos

- **Comunidade Open Source**: Pelas bibliotecas utilizadas.
- **Programmer's Beyond IT**: Pelo apoio e orientação técnica.
- **Colaboradores**: Todos que contribuíram direta ou indiretamente para o sucesso deste projeto.

--- 
