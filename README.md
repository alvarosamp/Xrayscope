# Xrayscope

Projeto MLOps com fluxo CD4ML completo incluindo aplicação web com suporte de IA, deploy automatizado local e em cloud (AWS) usando IaC.

Resumo rápido
- Treinamento, versionamento e promoção de modelos com MLflow.
- Armazenamento de artefatos em S3 (MinIO local / AWS S3 cloud).
- Webapp para previsões (FastAPI + Gunicorn).
- Infraestrutura local com Docker Compose e deploy cloud com Terraform (ECS / ECR / ALB).
- Notebooks para experimentação e execução reprodutível (Jupyter / SageMaker).

## ✅ Objetivo do projeto
Construir um pipeline MLOps completo (CD4ML) que:
- Permita treinar, registrar e promover modelos automaticamente.
- Ofereça uma API web para inferência com o modelo em produção.
- Forneça ambientes replicáveis localmente e na AWS via IaC.
- Seja um template reutilizável para projetos de visão computacional (Chest X‑Ray / Pneumonia).

## 📌 Como os componentes se comunicam (visão prática)
- Jupyter / scripts de treinamento usam dados (bucket datasource) e disparam treinamento.
- O processo de treinamento registra métricas e artefatos no MLflow (tracking server).
- Artefatos do MLflow gravam em um bucket S3 (local via MinIO ou AWS S3).
- O webapp consulta o MLflow Registry (tracking URI) para baixar o modelo mais recente e servir previsões.
- Infraestrutura (terraform) cria recursos AWS para produção (ECR, ECS Fargate, ALB, S3).

## 🏗️ Arquitetura (resumo)
Local:
- MLflow: http://localhost:5000  
- Webapp: http://localhost:5001  
- MinIO (S3 local): http://localhost:9000  
- Jupyter: execução local

Cloud (AWS):
- MLflow, Webapp expostos via ALB/DNS configurados pelo Terraform
- Artefatos em buckets S3 reais
- Containers em ECS Fargate, imagens em ECR
- SageMaker para execução de notebooks se desejado

## 🚀 Execução Local (passos até onde você já foi)
Pré-requisitos:
- Docker & Docker Compose
- Python 3.8+ (apenas para notebooks/auxiliares)

1) Clonar e preparar ambiente
```bash
cd hm-mlflow
cp .env.example .env   # cria arquivo .env local a partir do template
```

2) Subir serviços (constrói imagens definidas pelos Dockerfiles)
```bash
docker compose up --build
```

3) Endpoints principais
- MLflow UI: http://localhost:5000  
- Webapp: http://localhost:5001  
- MinIO: http://localhost:9000

4) Treinar localmente via notebook
```bash
cd jupyter
python3 -m venv venv
source venv/bin/activate   # no Windows: .\venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
- Mantenha `PROFILE = "local"` no notebook para executar contra serviços locais.

## ☁️ Execução na Nuvem (visão geral / aviso)
- Requer credenciais AWS e Terraform instalado.
- O fluxo cria ECR, envia imagens, provisiona ECS/Fargate e ALB.
- Custo: executar em AWS gera custos — destrua recursos com `terraform destroy` quando terminar.

Principais passos (resumido):
- terraform init && terraform apply (infra)
- Build e push das imagens para ECR
- Atualizar variáveis do Terraform com o Account ID
- Executar tasks/services no ECS

## 🗂️ Estrutura do projeto
```
HM-mlflow/
├── infra/              # Terraform (AWS)
├── mlflow/             # MLflow server + Dockerfile-mlflow
├── webapp/             # Interface web + Dockerfile-webapp
├── model/              # Scripts de treinamento
├── minio/              # Scripts de criação de buckets locais
├── jupyter/            # Notebooks
├── source/             # Dados de treinamento
├── docker-compose.yaml # Orquestração local
└── README.md
```

## 🔧 Variáveis de ambiente (.env) — essenciais
Exemplos:
```
EXECUTION_ENVIRONMENT=local
MLFLOW_TRACKING_URI_CLOUD=http://mlflow.hm-mlflow.local
AWS_ACCESS_KEY_ID=<sua_key>
AWS_SECRET_ACCESS_KEY=<sua_secret>
MINIO_ROOT_USER=<user>
MINIO_ROOT_PASSWORD=<password>
```

## 📎 Dicas rápidas
- Use `pip freeze > requirements.txt` dentro do venv para gerar requirements.
- Se preferir não construir imagem customizada do MLflow, é possível usar a imagem oficial mlflow/mlflow no docker-compose.
- Sempre limpe recursos AWS com `terraform destroy` para evitar cobranças contínuas.

---

Se quiser, eu atualizo este README com:
- Um título de repo mais chamativo (ex.: xrayscope-ai) e badges;  
- Instruções completas de build/push para ECR com exemplos substituindo placeholders;  
- seção passo‑a‑passo para troubleshooting e comandos úteis (logs, update-service).  
Escolha o que deseja adicionar.