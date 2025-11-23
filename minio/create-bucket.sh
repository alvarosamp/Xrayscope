#!/bin/sh
set -e

# wait for minio to be available
MC_ALIAS=local
MC_HOST=http://minio:9000
MAX_RETRIES=30
SLEEP=2

echo "Waiting for MinIO at ${MC_HOST}..."
TRIES=0
until curl -s ${MC_HOST} >/dev/null 2>&1 || [ ${TRIES} -ge ${MAX_RETRIES} ]; do
  TRIES=$((TRIES+1))
  echo "  still waiting (${TRIES}/${MAX_RETRIES})..."
  sleep ${SLEEP}
done

if [ ${TRIES} -ge ${MAX_RETRIES} ]; then
  echo "MinIO did not become available after ${MAX_RETRIES} attempts. Exiting."
  exit 1
fi

echo "Configuring mc alias..."
mc alias set ${MC_ALIAS} ${MC_HOST} "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"

for BUCKET in mlflow-artifacts datasource dev-models; do
  echo "Creating bucket: ${BUCKET} (if not exists)"
  mc mb --ignore-existing ${MC_ALIAS}/${BUCKET} || true
done

# If example data was mounted into /data/datasource, copy it into the datasource bucket
if [ -d /data/datasource ]; then
  echo "Copying local datasource to bucket datasource..."
  mc cp --recursive /data/datasource ${MC_ALIAS}/datasource || true
fi

echo "Bucket creation script finished."
