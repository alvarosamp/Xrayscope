# Data Review

Projeto MLOps com fluxo CD4ML completo incluindo aplicação web com suporte de IA, deploy automatizado local e em cloud (AWS) usando IaC

## 📋 Visão Geral

Este projeto implementa uma arquitetura de MLOps completa incluindo:

- **MLflow**: Tracking de experimentos e registro de modelos
- **Webapp**: Interface web para predições
- **Jupyter Notebook**: Ambiente para treinamento de modelos
- **Infraestrutura local**: Deploy automatizado com Docker Compose
- **Infraestrutura cloud AWS**: Deploy automatizado com Terraform
- **Containerização**: Docker para todos os componentes

## 🏗️ Arquitetura

### Local
- **MLflow**: http://localhost:5000
- **Webapp**: http://localhost:8080
- **MinIO**: http://localhost:9000 (S3 local)
- **Jupyter**: Execução local

### Cloud (AWS)
- **MLflow**: http://mlflow.hm-mlflow.local
- **Webapp**: http://app.hm-mlflow.local
- **S3**: Buckets AWS para dados e artefatos
- **ECS Fargate**: Containers gerenciados
- **SageMaker**: Notebooks na nuvem
- **ALB**: Load balancer para roteamento

### Base de dados
A base de dados utilizada no projeto é a Chest X-Ray Images (Pneumonia), que contém milhares de imagens de raios-X de tórax classificadas como normais ou com pneumonia, servindo para treinar e avaliar modelos de diagnóstico assistido por IA.
Disponível em: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## 🚀 Execução Local

### Pré-requisitos
- Docker e Docker Compose
##
sudo apt-get remove -y docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
#(Opcional) Executar Docker sem sudo
sudo groupadd -f docker
sudo usermod -aG docker $USER


- Python 3.8+

### Passos

1. **Baixe o projeto e configure o ambiente:**
```bash
cd hm-mlflow
cp .env.example .env
```

2. **Inicie os serviços:**
```bash
docker compose up --build
```

3. **Acesse os serviços:**
- MLflow UI: http://localhost:5000
- Webapp: http://localhost:5001
- MinIO: http://localhost:9000

4. **Crie uma venv, instale as dependências e execute o notebook de treinamento:**
```bash
cd jupyter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
- Mantenha `PROFILE = "local"` no notebook
- Não altere nenhuma outra variável, o notebook é programado para ser executado localmente por padrão
- Execute todas as células e analise os logs sobre o sucesso do registro da nova versão

## ☁️ Execução na Nuvem (AWS) - 
### ATENÇÃO - A implementação em núvem gera custos. Rode-a apenas se tiver plena consciência disso.

### Pré-requisitos

- [Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
```bash
# Atualiza pacotes
sudo apt-get update && sudo apt-get install -y gnupg software-properties-common
# Adiciona a chave GPG da HashiCorp
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
# Adiciona o repositório oficial
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
| sudo tee /etc/apt/sources.list.d/hashicorp.list
# Atualiza novamente
sudo apt-get update
# Instala Terraform
sudo apt-get install terraform
```


- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
```bash
# Instale unzip se necessário
sudo apt-get update && sudo apt-get install -y unzip
# Baixar o pacote oficial
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# Descompactar
unzip awscliv2.zip
# Instalar
sudo ./aws/install
```

- Credenciais AWS configuradas
```bash
aws configure

aws sts get-caller-identity
```

### Passo 1: Deploy da Infraestrutura

1. **Inicialize o Terraform:**
```bash
cd infra
terraform init
```

2. **Crie os repositórios ECR:**
```bash
terraform apply -target=aws_ecr_repository.mlflow -target=aws_ecr_repository.webapp -target=aws_ecr_repository.model
```

3. **Obtenha seu AWS Account ID e faça login no ECR:**
```bash
# Obter o Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Seu AWS Account ID: $AWS_ACCOUNT_ID"

# Login no ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

4. **Build e push das imagens:**
```bash
## Na raiz do projeto
/hm-mlflow

# Definir variável com o Account ID (se não definida no passo anterior)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"

# MLflow
docker build -t mlflow-image -f mlflow/Dockerfile-mlflow mlflow/
docker tag mlflow-image:latest $ECR_URI/hm-mlflow/mlflow:latest
docker push $ECR_URI/hm-mlflow/mlflow:latest

# Webapp
docker build -t webapp-image -f webapp/Dockerfile-webapp webapp/
docker tag webapp-image:latest $ECR_URI/hm-mlflow/webapp:latest
docker push $ECR_URI/hm-mlflow/webapp:latest

# Model
docker build -t model-image -f model/Dockerfile-model model/
docker tag model-image:latest $ECR_URI/hm-mlflow/model:latest
docker push $ECR_URI/hm-mlflow/model:latest
```

5. **Atualize as variáveis do Terraform:**
```bash
# Substitua <AWS_ACCOUNT_ID> pelo seu Account ID real no variables.tf
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
sed -i "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" infra/variables.tf
```
Ou edite manualmente `infra/variables.tf` substituindo `<AWS_ACCOUNT_ID>` pelo seu Account ID nas variáveis:
- `mlflow_image_uri`
- `webapp_image_uri` 
- `model_image_uri`

6. **Deploy completo:**
```bash
cd infra
terraform plan
terraform apply
```

### Passo 2: Configurar DNS Local

1. **Obtenha o IP do ALB:**
```bash
nslookup <ALB_DNS_NAME>
```
**Essa informação pode ser obtida com:**
```bash
cd infra && teraform output
```

2. **Adicione ao arquivo hosts:**
- **Tecla de atalho para o Executar**: win+R
- **Windows**: `C:\Windows\System32\drivers\etc\hosts`
- **Linux/macOS**: `/etc/hosts`
**Precisa ter privilégio de administrador para modificar o arquivo**
- **No Windows, após abrir o Executar, digite notepad e pressione ctrl+shift+enter**


```
<ALB_IP> mlflow.hm-mlflow.local app.hm-mlflow.local
```

### Passo 3: Executar Treinamento via ECS

```bash
# Obter valores do Terraform
SUBNET_ID=$(terraform output -json public_subnet_ids | jq -r '.[0]')
SG_ID=$(terraform output -raw ecs_security_group_id)

# Executar task
aws ecs run-task \
  --cluster hm-mlflow-cluster \
  --task-definition hm-mlflow-model-training \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region us-east-1
```

### Verifique o acesso
Acesse no seu navegador:
- **MLflow**: http://mlflow.hm-mlflow.local
- **Webapp**: http://app.hm-mlflow.local


### Atenção
# Caso as informações do modelo versão 1 estiverem aparecendo no rodapé, mas a previsão não estiver sendo feita, pode ter vencido o tempo de espera da webapp pra carregar o modelo. Após a finalização da task anterior de treinamento, rode o comando a seguir para atualizar o serviço de webapp que ele conseguirá fazer previsões
```bash
aws ecs update-service \
  --cluster hm-mlflow-cluster \
  --service hm-mlflow-webapp-service \
  --force-new-deployment \
  --region us-east-1
```

### Passo 4: Executar Treinamento no SageMaker

1. **Acesse o SageMaker:**
- Console AWS → SageMaker Studio → Notebook instances
- Abra `hm-mlflow-notebook`

2. **Faça upload do notebook:**
- Upload `jupyter/training_notebook.ipynb`

3. **Obtenha os valores necessários localmente:**
```bash
# No seu ambiente local, na pasta infra
cd infra
echo "ALB_DNS_NAME: $(terraform output -raw alb_dns_name)"
echo "DATASOURCE_BUCKET_NAME: $(terraform output -raw datasource_bucket_name)"
```

4. **Configure e execute no SageMaker:**
- Altere `PROFILE = "cloud"` na segunda célula
- Preencha as variáveis `ALB_DNS_NAME` e `DATASOURCE_BUCKET_NAME` com os valores obtidos no passo anterior
- Execute todas as células

5. **Destruição da infraestrutura na AWS (cloud)**
# Após finalizar seus trabalhos e validar todo funcionamento, você pode, quando quiser, remover TODA infraestrutura para não gerar gastos adicionais
```bash
# No seu ambiente local, na pasta infra
cd infra
terraform destroy
```

### ATENÇÃO
# A não remoção dos recursos da AWS geram custos permanentes até que você os remova. Caso você esteja apenas utilizando para estudo, não esqueça de limpar o ambiente para não sofrer cobranças indesejadas.

## 📊 Monitoramento

### Logs AWS
```bash
# Logs do MLflow
aws logs get-log-events --log-group-name "/ecs/hm-mlflow-mlflow" --log-stream-name "<STREAM>" --region us-east-1

# Logs do Webapp
aws logs get-log-events --log-group-name "/ecs/hm-mlflow-webapp" --log-stream-name "<STREAM>" --region us-east-1

# Logs do treinamento
aws logs get-log-events --log-group-name "/ecs/hm-mlflow-model-training" --log-stream-name "<STREAM>" --region us-east-1
```

### Teste dos Serviços
```bash
# Teste MLflow
curl -H "Host: mlflow.hm-mlflow.local" http://<ALB_DNS>

# Teste Webapp
curl -H "Host: app.hm-mlflow.local" http://<ALB_DNS>
```

## 🔧 Configuração

### Variáveis de Ambiente (.env)
```bash
EXECUTION_ENVIRONMENT=local  # ou "cloud"
MLFLOW_TRACKING_URI_CLOUD=http://mlflow.hm-mlflow.local
AWS_ACCESS_KEY_ID=<sua_key>
AWS_SECRET_ACCESS_KEY=<sua_secret>
```

### Notebook Configuration
- **Local**: `PROFILE = "local"`
- **Cloud**: `PROFILE = "cloud"`

## 📁 Estrutura do Projeto

```
HM-mlflow/
├── infra/              # Terraform (AWS)
├── mlflow/             # MLflow server
├── webapp/             # Interface web
├── model/              # Scripts de treinamento
├── minio/              # Scripts de criação do bucket S3 local
├── jupyter/            # Notebooks
├── source/             # Dados de treinamento
├── docker-compose.yaml # Orquestração local
└── README.md
```

## 🛠️ Tecnologias

- **MLflow**: Tracking e registro de modelos
- **FastAPI**: API do webapp
- **Scikit-learn**: Machine learning
- **OpenCV**: Processamento de imagens
- **Docker**: Containerização
- **Terraform**: Infrastructure as Code
- **AWS**: ECS, S3, SageMaker, ALB
- **MinIO**: S3 local para desenvolvimento

## 📝 Notas Importantes

- O notebook suporta execução local e na nuvem com a mesma base de código
- Modelos são automaticamente promovidos para "Production" após treinamento
- O webapp sempre usa o modelo mais recente em produção
- DNS local é necessário para acessar serviços na nuvem
- SageMaker configura DNS automaticamente quando `PROFILE = "cloud"`

## 🔍 Troubleshooting

### Problema: MLflow não acessível na nuvem
**Solução**: Verifique se o DNS está configurado corretamente no arquivo hosts

### Problema: Imagens não carregam
**Solução**: Verifique se os dados estão no bucket S3 correto

### Problema: Erro de permissão no SageMaker
**Solução**: Verifique se a IAM role tem permissões para S3 e MLflow

